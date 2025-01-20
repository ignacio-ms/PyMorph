# Standard libraries
import os
import subprocess
import sys
import getopt

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util import values as v
from util.misc.bash import arg_check
from util.misc.colors import bcolors as c
from util.data.dataset_ht import HtDataset, find_group
from util.misc.timer import LoadingBar


def run(
        source_mesh, target_mesh,
        source_landmarks, target_landmarks,
        specimen, data_path=None,
        init_method=2,
        n_pre_iters=200,
        n_main_iters=600,
        gpu_type='INTEL',
        verbose=0
):
    print(f'{c.OKBLUE}Running surface map computation{c.ENDC}...{specimen}')
    # if data_path is None or data_path == v.data_path:
    data_path = data_path + f'{specimen}/'
    data_path_out = data_path + f'{specimen}/map/'

    if verbose:
        print(f'\t{c.BOLD}Moving meshes and landmarks{c.ENDC}...')

    # Get just the file names
    source_mesh = source_mesh.split('/')[-1]
    target_mesh = target_mesh.split('/')[-1]
    source_landmarks = source_landmarks.split('/')[-1]
    target_landmarks = target_landmarks.split('/')[-1]

    current_dir = '/app/meshes/'

    print(f'\t{c.BOLD}Exporting environment variables{c.ENDC}...')
    subprocess.run(
        f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{current_dir}surface_map/{gpu_type}/libs/',
        shell=True, check=True
    )

    subprocess.run(
        f'echo $LD_LIBRARY_PATH',
        shell=True, check=True
    )

    if verbose:
        print(f'\t{c.BOLD}Running SurfaceMapComputation{c.ENDC}...')

    print(f'Data path: {data_path}')
    print(f'Current dir: {current_dir}')
    print(f'Source mesh: {source_mesh}')
    print(f'Target mesh: {target_mesh}')
    print(f'Source landmarks: {source_landmarks}')
    print(f'Target landmarks: {target_landmarks}')

    subprocess.run(
        f'/app/opt/SurfaceMapComputation/SurfaceMapComputation --path {data_path} '
        f'--init_method {init_method} --n_pre_iters {n_pre_iters} '
        f'--n_main_iters {n_main_iters} --shapeA {source_mesh} '
        f'--shapeB {target_mesh} --landmarksA {source_landmarks} '
        f'--landmarksB {target_landmarks}',
        shell=True, check=True
    )
    # subprocess.run(
    #     f'ldd /app/opt/SurfaceMapComputation/SurfaceMapComputation',
    #     shell=True, check=True
    # )

    if verbose:
        print(f'\t{c.BOLD}Moving output{c.ENDC}...')

    if verbose:
        print(f'{c.OKGREEN}Output moved{c.ENDC}')
        print(f'{c.OKGREEN}Surface map computation finished{c.ENDC}')


def print_usage():
    print(
        'usage: run_surface_map.py -p <path> -i <image> -s <specimen> -gr <group> -v <verbose>'
        f'\n\n{c.BOLD}Arguments{c.ENDC}:'
        f'\n{c.BOLD}<path>{c.ENDC}: Path to data directory.'
        f'\n{c.BOLD}<specimen>{c.ENDC}: Specimen to run prediction on.'
        f'\n{c.BOLD}<group>{c.ENDC}: Group of specimens to run prediction on.'
        f'\n{c.BOLD}<verbose>{c.ENDC}: Verbosity level.'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path = v.data_path
    spec = None
    group = None
    tissue = 'myocardium'
    verbose = 1

    try:
        opts, args = getopt.getopt(argv, 'hp:s:g:t:v:', [
            'help', 'path=', 'spec=', 'group=', 'tissue=', 'verbose='
        ])

        if len(opts) > 5:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-p', '--path'):
                data_path = arg_check(opt, arg, '-p', '--path', str, print_usage)
            elif opt in ('-s', '--spec'):
                spec = arg_check(opt, arg, '-s', '--spec', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f'{c.FAIL}Invalid argument:{c.ENDC} {opt}')
                print_usage()

        dataset = HtDataset(data_path=data_path)

        if spec is not None:
            specimens = [spec]

        elif group is not None:
            if group in dataset.specimens:
                specimens = dataset.specimens[group]
            else:
                print(f'{c.FAIL}Invalid group{c.ENDC}: {group}')
                sys.exit(2)

        else:
            specimens = []
            for group_name, group_specimens in dataset.specimens.items():
                specimens.extend(group_specimens)

        bar = LoadingBar(len(specimens))
        for s in specimens:
            gr = find_group(s)
            try:
                source_mesh = data_path + f'{s}/ATLAS_{gr}.ply'
                target_mesh = data_path + f'{s}/2019{s}Shape.ply'
                source_landmarks = data_path + f'{s}/ATLAS_{gr}_landmarks.pinned'
                target_landmarks = data_path + f'{s}/2019{s}_landmarks.pinned'

                run(
                    source_mesh, target_mesh,
                    source_landmarks, target_landmarks,
                    specimen=s, data_path=data_path,
                    verbose=verbose
                )
            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC}: {e}')

                import traceback
                traceback.print_exc()

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
