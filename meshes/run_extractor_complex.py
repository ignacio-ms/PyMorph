import os
import sys
import getopt

import trimesh
import pandas as pd

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from utils.misc.bash import arg_check
from utils.misc.colors import bcolors as c
from utils.data.dataset_ht import HtDataset
from utils import values as v

from meshes.utils.features.extractor import MeshFeatureExtractor


def run(mesh_path, tissue_path, features_path, path_out=None, verbose=0, parallelize=True):
    cell_mesh = trimesh.load(mesh_path, file_type='ply')
    tissue_mesh = trimesh.load(tissue_path, file_type='ply')
    features = pd.read_csv(features_path)

    if verbose:
        print(f'{c.OKBLUE}Extracting features{c.ENDC} for {c.BOLD}{mesh_path}{c.ENDC}:')
        print(f'\t{c.BOLD}Perpendicularity')
        print(f'\t{c.BOLD}Sphericity')
        print(f'\t{c.BOLD}Columnarity')

    extractor = MeshFeatureExtractor(cell_mesh, tissue_mesh)
    new_features = extractor.extract(n_jobs=6 if parallelize else 1)

    if verbose:
        print(f'{c.OKGREEN}Features extracted{c.ENDC} [{len(new_features)} / {len(features)}]')

    # Rename original_labels -> cell_id if needed
    if 'original_labels' in features:
        features.rename(columns={'original_labels': 'cell_id'}, inplace=True)

    columns2overwrite = ['perpendicularity', 'sphericity', 'columnarity']
    if any([col in features.columns for col in columns2overwrite]):
        features.drop(columns=columns2overwrite, inplace=True)

    # Ensure data types
    features['cell_id'] = features['cell_id'].astype(int)
    new_features['cell_id'] = new_features['cell_id'].astype(int)

    merged_features = pd.merge(features, new_features, on='cell_id', how='left')
    merged_features.to_csv(
        features_path if path_out is None else path_out,
        index=False
    )


def print_usage():
    """
    Print usage of script.
    """
    print(
        'usage: run_extractor_complex.py -l <level> -t <tissue> -s <specimen> -g <group> -e <segmentation_path> -r <raw_path> -o <path_out> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}-l, --level{c.ENDC}: Level to use. (Default: Membrane)\n'
        f'{c.BOLD}-t, --tissue{c.ENDC}: Tissue to use. (Default: myocardium)\n'
        f'{c.BOLD}-s, --specimen{c.ENDC}: Specimen to process.\n'
        f'{c.BOLD}-g, --group{c.ENDC}: Group to process.\n'
        f'{c.BOLD}-e, --mesh_path{c.ENDC}: Path to segmentation data.\n'
        f'{c.BOLD}-r, --features_path{c.ENDC}: Path to features file (.csv).\n'
        f'{c.BOLD}-o, --path_out{c.ENDC}: Output path for the features file.\n'
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
    level = 'Membrane'
    tissue = 'myocardium'
    specimen = None
    group = None
    mesh_path = None
    features_path = None
    tissue_path = None
    path_out = None
    parallelize = True
    all_remaining = False
    verbose = 0

    try:
        opts, args = getopt.getopt(argv, 'hd:l:t:s:g:e:r:p:o:m:a:v:', [
            'help', 'data_path=', 'level=', 'tissue=', 'specimen=', 'group=', 'mesh_path=', 'features_path=', 'tissue_path=', 'path_out=', 'multiprocessing=', 'all=', 'verbose='
        ])

        if len(opts) == 0:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-d', '--data_path'):
                data_path = arg_check(opt, arg, '-d', '--data_path', str, print_usage)
            elif opt in ('-l', '--level'):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-s', '--specimen'):
                specimen = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-e', '--mesh_path'):
                mesh_path = arg_check(opt, arg, '-e', '--segmentation_path', str, print_usage)
            elif opt in ('-r', '--features_path'):
                features_path = arg_check(opt, arg, '-r', '--features_path', str, print_usage)
            elif opt in ('-p', '--tissue_path'):
                tissue_path = arg_check(opt, arg, '-p', '--tissue_path', str, print_usage)
            elif opt in ('-o', '--path_out'):
                path_out = arg_check(opt, arg, '-o', '--path_out', str, print_usage)
            elif opt in ('-m', '--multiprocessing'):
                parallelize = arg_check(opt, arg, '-m', '--multiprocessing', int, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            elif opt in ('-a', '--all'):
                all_remaining = arg_check(opt, arg, '-a', '--all', bool, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()
    except getopt.GetoptError:
        print_usage()

    ds = HtDataset(data_path=data_path)

    # If segmentation_path and raw_path are provided, process them directly
    if mesh_path is not None and features_path is not None and tissue_path is not None:
        if path_out is None:
            print(f"{c.FAIL}Output path not provided, overwriting input features file{c.ENDC}")
            sys.exit(2)
        print(f"{c.OKBLUE}Processing provided mesh{c.ENDC}")
        run(mesh_path, tissue_path, features_path, path_out, verbose, parallelize)
        sys.exit(0)

    # Else, proceed with specimens/groups
    if specimen is not None:
        specimens = [specimen]

    elif group is not None:
        if group in ds.specimens:
            specimens = ds.specimens[group]
        else:
            print(f"{c.FAIL}Invalid group{c.ENDC}: {group}")
            sys.exit(2)
    else:
        if all_remaining:
            # Process all specimens
            specimens = []
            for group_name, group_specimens in ds.specimens.items():
                specimens.extend(group_specimens)
        else:
            specimens = ds.check_features_complex(level, tissue, verbose)

    print(f'{c.OKBLUE}Level{c.ENDC}: {level}')
    print(f'{c.OKBLUE}Tissue{c.ENDC}: {tissue}')

    for spec in specimens:
        print(f'{c.BOLD}Specimen{c.ENDC}: {spec}')

        try:
            mesh_path = ds.get_mesh_cell(spec, level, tissue, verbose)
            tissue_path = ds.get_mesh_tissue(spec, tissue, verbose)
            features_path = ds.get_features(spec, level, tissue, verbose, only_path=True)

            run(mesh_path, tissue_path, features_path, path_out, verbose, parallelize)

        except Exception as e:
            print(f'{c.FAIL}Error{c.ENDC}: {e}')
            if verbose:
                import traceback
                traceback.print_exc()
            continue
