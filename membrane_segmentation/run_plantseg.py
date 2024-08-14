# Standard libraries
import os
import sys
import getopt

import torch

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
            img_paths = dataset.raw_nuclei_path
            img_paths = [
                img_path for img_path in img_paths if any(
                    specimen in img_path for specimen in specimens
                )
            ]

        elif spec is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {spec}')

            img_path, _ = dataset.read_specimen(spec, verbose=verbose)
            img_paths = [img_path]

        else:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on all remaining images{c.ENDC}')

            dataset.check_segmentation(verbose=verbose)
            img_paths = dataset.missing_membrane

        if len(img_paths) == 0:
            print(f'{c.FAIL}No images found{c.ENDC}')
            sys.exit(2)

        bar = LoadingBar(len(img_paths))
        for img_path in img_paths:
            modify_yaml_path(img_path)
            subprocess.run(f'plantseg --config membrane_segmentation/config.yaml', shell=True, check=True)
            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
