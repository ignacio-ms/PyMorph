# Standard libraries
import os
import sys
import getopt

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset, find_group
from auxiliary.utils.timer import LoadingBar

from meshes.utils.registration.surface_map_computation import run


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
                source_mesh = data_path + f'ATLAS/{tissue}/ATLAS_{gr}.ply'
                target_mesh = dataset.get_mesh_tissue(s, tissue)

                source_landmarks = data_path + f'Landmarks/ATLAS/ATLAS_{gr}_landmarks.pinned'
                target_landmarks = data_path + f'Landmarks/2019{s}_landmarks.pinned'

                aux = target_mesh.split('/')[:-1] + ['map', s]
                out_path = '/'.join(aux) + '/'
                # os.makedirs(out_path, exist_ok=True)

                run(
                    source_mesh, target_mesh,
                    source_landmarks, target_landmarks,
                    out_path,
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
