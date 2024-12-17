import os
import sys
import multiprocessing
import getopt

import trimesh
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import tensorflow as tf

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from cell_division.layers.custom_layers import ExtendedLSEPooling, extended_w_cel_loss
from cell_division.nets.transfer_learning import CNN

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.data.dataset_nuclei import NucleiDataset
from auxiliary.utils.timer import LoadingBar
from auxiliary.gpu.gpu_tf import (
    increase_gpu_memory,
    set_gpu_allocator
)


# Global variable for the model to be initialized in each worker
model = None


def setup_gpu(): # gpu_id
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print(f'{c.WARNING}Warning{c.ENDC}: No GPU found. Running on CPU.')
        return

    increase_gpu_memory()
    set_gpu_allocator()
    # if gpu_id == 0:
    #     set_mixed_precision()


def load_model():
    """
    Load the model with custom layers and loss function.
    This function will be called within each worker process.
    """
    model = CNN(
        base=tf.keras.applications.VGG16,
        input_shape=(100, 100, 3),
        n_classes=3
    )
    model.build_top(activation='softmax', b_type='CAM', pooling=ExtendedLSEPooling)
    model.compile(
        lr=.001,
        loss=extended_w_cel_loss()
    )

    model_dir = os.path.join(current_dir, os.pardir, 'models', 'cellular_division_models', 'vgg16_nuclei_under.h5')
    model.load(model_dir)
    return model


def process_cell(i, nuclei_ds): # gpu_id
    """
    Process a single cell, performing prediction using the global model.
    """
    global model
    if model is None:
        model = load_model()  # Load the model if not already loaded in this worker

    try:
        img, mask, cell_id = nuclei_ds[i]
        pred = model.predict3d(img, mask)
        return {
            'cell_id': cell_id,
            'cell_division': pred
        }
    except Exception as e:
        print(f'{c.WARNING}Warning{c.ENDC}: Error in cell {cell_id}: {e}')
        return {
            'cell_id': cell_id,
            'cell_division': np.nan
        }


def print_usage():
    """
    Print usage of script.
    """
    print(
        'usage: run_cell_division.py -l <level> -t <tissue> -s <specimen> -g <group> -e <segmentation_path> -r <raw_path> -o <path_out> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}-t, --tissue{c.ENDC}: Tissue to use. (Default: myocardium)\n'
        f'{c.BOLD}-s, --specimen{c.ENDC}: Specimen to process.\n'
        f'{c.BOLD}-g, --group{c.ENDC}: Group to process.\n'
        f'{c.BOLD}-a, --all{c.ENDC}: Run all specimens.\n'
        f'{c.BOLD}-v, --verbose{c.ENDC}: Verbosity level. (Default: 0)\n'
    )
    sys.exit(2)


def get_group(ds, specimen):
    for group_name, specimens in ds.specimens.items():
        if specimen in specimens:
            return group_name
    return None


if __name__ == '__main__':
    argv = sys.argv[1:]
    data_path = v.data_path
    tissue = 'myocardium'
    specimen = None
    group = None
    run_all = False
    verbose = 0

    try:
        opts, args = getopt.getopt(argv, 'hd:t:s:g:e:a:v:', [
            'help', 'data_path=', 'tissue=', 'specimen=', 'group=', 'all=', 'verbose='
        ])

        if len(opts) == 0:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-d', '--data_path'):
                data_path = arg_check(opt, arg, '-d', '--data_path', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-s', '--specimen'):
                specimen = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            elif opt in ('-a', '--all'):
                run_all = arg_check(opt, arg, '-a', '--all', bool, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()
    except getopt.GetoptError:
        print_usage()

    ds = HtDataset(data_path=data_path)

    # Determine specimens to process
    if specimen is not None:
        specimens = [specimen]
    elif group is not None:
        if group in ds.specimens:
            specimens = ds.specimens[group]
        else:
            print(f"{c.FAIL}Invalid group{c.ENDC}: {group}")
            sys.exit(2)
    else:
        if run_all:
            specimens = []
            for group_name, group_specimens in ds.specimens.items():
                specimens.extend(group_specimens)
        else:
            specimens = ds.check_features_complex('Nuclei', tissue, attr='cell_division', verbose=verbose)

    print(f'{c.OKGREEN}Running Cell Division{c.ENDC}')
    print(f'{c.OKBLUE}Tissue{c.ENDC}: {tissue}')

    # # Set multiprocessing start method to spawn
    # multiprocessing.set_start_method('spawn', force=True)
    #
    # num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    # n_jobs = 1 if num_gpus == 0 else num_gpus

    model = load_model()

    for spec in specimens:
        print(f'{c.BOLD}Specimen{c.ENDC}: {spec}')

        try:
            nuclei_ds = NucleiDataset(spec, tissue=tissue, verbose=verbose)
            features_path = ds.get_features(spec, 'Nuclei', tissue, only_path=True, filtered=True)
            features = pd.read_csv(features_path)

            if 'cell_id' not in features.columns:
                if 'original_labels' in features.columns:
                    features.rename(columns={'original_labels': 'cell_id'}, inplace=True)
                else:
                    raise ValueError('cell_id not found in features')

            # # Parallel processing of each cell
            # new_rows = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            #     delayed(process_cell)(i, nuclei_ds, gpu_id=i % num_gpus)
            #     for i in range(len(nuclei_ds.cell_ids))
            # )

            new_rows = []
            bar = LoadingBar(len(nuclei_ds.cell_ids))
            for i in range(len(nuclei_ds.cell_ids)):
                new_rows.append(process_cell(i, nuclei_ds))
                bar.update()
            bar.end()

            # Save results to DataFrame and CSV
            columns2overwrite = ['cell_division']
            if any([col in features.columns for col in columns2overwrite]):
                features.drop(columns=columns2overwrite, inplace=True)

            new_features = pd.DataFrame(new_rows, columns=['cell_id', 'cell_division'])
            merged_features = pd.merge(features, new_features, on='cell_id', how='left')
            merged_features.to_csv(features_path, index=False)

        except Exception as e:
            print(f'{c.FAIL}Error{c.ENDC}: {e}')
            if verbose:
                import traceback
                traceback.print_exc()
            continue



