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

from util import values as v
from util.misc.bash import arg_check
from util.misc.colors import bcolors as c
from util.data.dataset_ht import HtDataset, find_group
from util.misc.timer import LoadingBar

from meshes.utils.registration.cell_map_computation import CellTissueMap


def print_usage():
    print(
        'usage: run_cell_map.py -p <path> -i <image> -s <specimen> -gr <group> -v <verbose>'
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
    level = 'Membrane'
    verbose = 1

    try:
        opts, args = getopt.getopt(argv, 'hp:s:g:t:l:v:', [
            'help', 'path=', 'spec=', 'group=', 'tissue=', 'level=', 'verbose='
        ])

        if len(opts) > 6:
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
            elif opt in ('-l', '--level'):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
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

            specimens = [s for s in specimens if s in v.specimens_to_analyze]

        if level not in ['Membrane', 'Nuclei']:
            print(f'{c.FAIL}Invalid level{c.ENDC}: {level} (Membrane | Nuclei)')
            sys.exit(2)

        bar = LoadingBar(len(specimens))
        for s in specimens:
            gr = find_group(s)
            print(f'{c.OKGREEN}Specimen{c.ENDC}: {s} ({gr}) - {tissue} - {level}')
            try:
                cell_map = CellTissueMap(s, tissue=tissue, verbose=verbose)

                cell_map.map_cells(type=level)
                cell_map.get_neighborhood(radius=50 if level == 'Membrane' else 40, type=level)

                # for feature in cell_map.cell_features.columns:
                #     if feature in ['cell_in_props', 'centroids', 'lines']:
                #         continue
                #     print(f'{c.OKBLUE}\tFeature{c.ENDC}: {feature}')
                #     try:
                #         _ = cell_map.color_mesh(feature, type='Membrane')
                #     except Exception as e:
                #         print(f'{c.FAIL}Error{c.ENDC}: {feature}')
                #         print(e)

            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC}: {e}')

                import traceback
                traceback.print_exc()

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
