# Standard libraries
import os
import sys
import getopt

import torch

from scipy import ndimage
from skimage import exposure
from csbdeep.utils import normalize as deep_norm

import subprocess
import yaml

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.utils.timer import LoadingBar, timed
from auxiliary.data import imaging

from feature_extraction.feature_extractor import filter_by_margin, filter_by_volume

# Configurations
use_gpu = torch.cuda.is_available()
print(f"GPU activated: {use_gpu}")


def modify_yaml_path(new_path, file_path='membrane_segmentation/config.yaml'):
    subprocess.run(
        f"sed -i 's|^path:.*|path: {new_path}|' {file_path}",
        shell=True, check=True
    )


def print_usage():
    print(
        'usage: run_plantseg.py -p <path> -i <image> -s <specimen> -gr <group> -v <verbose>'
        f'\n\n{c.BOLD}Arguments{c.ENDC}:'
        f'\n{c.BOLD}<path>{c.ENDC}: Path to data directory.'
        f'\n{c.BOLD}<image>{c.ENDC}: Image to run prediction on.'
        f'\n{c.BOLD}<specimen>{c.ENDC}: Specimen to run prediction on.'
        f'\n{c.BOLD}<group>{c.ENDC}: Group of specimens to run prediction on.'
        f'\n{c.BOLD}<verbose>{c.ENDC}: Verbosity level.'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path, img, spec, group, verbose = None, None, None, None, 1

    try:
        opts, args = getopt.getopt(argv, 'hp:i:s:g:v:', [
            'help', 'path=', 'img=', 'spec=', 'group=', 'verbose='
        ])

        if len(opts) > 5:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-p', '--path'):
                data_path = arg_check(opt, arg, '-p', '--path', str, print_usage)
            elif opt in ('-i', '--img'):
                img = arg_check(opt, arg, '-i', '--img', str, print_usage)
            elif opt in ('-s', '--spec'):
                spec = arg_check(opt, arg, '-s', '--spec', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f'{c.FAIL}Invalid argument:{c.ENDC} {opt}')
                print_usage()

        if data_path is None:
            data_path = v.data_path

        dataset = HtDataset(data_path=data_path)

        if img is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on image{c.ENDC}: {img}')

            img_paths = [data_path + img]
            img_paths_out = [img_paths[0].replace('RawImages', 'Segmentation')]
            img_paths_out = [
                img_paths_out[0].replace(
                    '_mGFP_decon_0.5',
                    f'_mask'
                )
            ]

        elif group is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on group{c.ENDC}: {group}')

            dataset.check_segmentation(verbose=verbose)
            dataset.read_img_paths(type='RawImages')

            if group not in dataset.specimens.keys():
                print(f'{c.FAIL}Invalid group{c.ENDC}: {group}')
                print(f'{c.BOLD}Available groups{c.ENDC}: {list(dataset.specimens.keys())}')
                sys.exit(2)

            specimens = dataset.specimens[group]
            img_paths = dataset.raw_membrane_path
            img_paths = [
                img_path for img_path in img_paths if any(
                    specimen in img_path for specimen in specimens
                )
            ]

            img_paths_out = dataset.missing_membrane_out
            img_paths_out = [
                img_path_out for img_path_out in img_paths_out if any(
                    specimen in img_path_out for specimen in specimens
                )
            ]

        elif spec is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {spec}')

            img_path, img_path_out = dataset.read_specimen(spec, level='Membrane', verbose=verbose)
            img_paths = [img_path]
            img_paths_out = [img_path_out]

        else:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on all remaining images{c.ENDC}')

            dataset.check_segmentation(verbose=verbose)
            img_paths = dataset.missing_membrane
            img_paths_out = dataset.missing_membrane_out

        if len(img_paths) == 0:
            print(f'{c.FAIL}No images found{c.ENDC}')
            sys.exit(2)

        bar = LoadingBar(len(img_paths))
        for img_path, img_path_out in zip(img_paths, img_paths_out):

            img = imaging.read_image(img_path, verbose=verbose)

            print(f'{c.OKBLUE}Pre-processing image{c.ENDC}:')
            print(f'\t{c.OKBLUE}Histogram equalization{c.ENDC}...')
            img = exposure.equalize_hist(img)

            print(f'\t{c.OKBLUE}Normalization{c.ENDC}...')
            img = deep_norm(img, 1, 99.8, axis=(0, 1, 2))

            print(f'\t{c.OKBLUE}Median filter{c.ENDC}...')
            img = ndimage.median_filter(img, size=3)

            if verbose:
                print(f'{c.OKBLUE}Transforming image {c.BOLD} .nii.gz -> .h5{c.ENDC}: {img_path}')
            imaging.nii2h5(img, img_path.replace('.nii.gz', '.h5'), verbose=verbose)

            modify_yaml_path(img_path.replace('.nii.gz', '.h5'))
            subprocess.run(f'plantseg --config membrane_segmentation/config.yaml', shell=True, check=True)

            # Load segmented image and correct axes
            path = '/'.join(img_path.split('/')[:-1] + [
                'PreProcessing', 'confocal_3D_unet_sa_meristem_cells',
                'GASP', 'PostProcessing',
                img_path.split('/')[-1].replace('.nii.gz', '_predictions_gasp_average.tiff')
            ])

            masks = imaging.read_image(
                path,
                axes='ZYX', verbose=verbose
            )

            # Filter segmented image
            masks = filter_by_volume(masks, percentile=97, verbose=verbose)

            # Move output to correct location
            imaging.save_prediction(masks, img_path_out, verbose=verbose)

            # Remove temporary files and .h5 file
            file_to_remove = img_path.replace('.nii.gz', '.h5')
            directory_to_remove = '/'.join(file_to_remove.split('/')[:-1] + ['PreProcessing'])

            subprocess.run(f'rm -r {directory_to_remove}', shell=True, check=True)
            subprocess.run(f'rm {file_to_remove}', shell=True, check=True)

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
