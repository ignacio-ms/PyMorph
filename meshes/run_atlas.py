# Standard libraries
import os
import sys
import getopt

import trimesh


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

from meshes.utils.registration.feature_map_computation import FeatureMap
from meshes.utils.registration.cell_map_computation import CellTissueMap
from meshes.utils.visualize_analysis import save_mesh_views, create_feature_grid


def print_usage():
    print(
        'usage: run_atlas.py -p <path> -i <image> -s <specimen> -gr <group> -v <verbose>'
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
    group = None
    tissue = 'myocardium'
    level = 'Membrane'
    feature = None
    verbose = 1

    try:
        opts, args = getopt.getopt(argv, 'hp:g:t:l:f:v:', [
            'help', 'path=', 'group=', 'tissue=', 'level=', 'feature=', 'verbose='
        ])

        if len(opts) > 5:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-p', '--path'):
                data_path = arg_check(opt, arg, '-p', '--path', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-l', '--level'):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
            elif opt in ('-f', '--feature'):
                feature = arg_check(opt, arg, '-f', '--feature', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f'{c.FAIL}Invalid argument:{c.ENDC} {opt}')
                print_usage()

        dataset = HtDataset(data_path=data_path)

        if group is not None:
            if group in dataset.specimens:
                specimens = dataset.specimens[group]
            else:
                print(f'{c.FAIL}Invalid group{c.ENDC}: {group}')
                sys.exit(2)

        else:
            print(f'{c.FAIL}No group provided{c.ENDC}')
            sys.exit(2)

        if feature is None:
            print(f'{c.FAIL}No feature provided{c.ENDC}')
            sys.exit(2)

        feature_maps = []
        specimens = [s for s in specimens if s in v.specimens_to_analyze]

        for s in specimens:
            gr = find_group(s)
            print(f'{c.OKGREEN}Specimen{c.ENDC}: {s} ({gr})')
            try:
                print(f'\t{c.OKBLUE}Type{c.ENDC}: {level}')
                cell_map = CellTissueMap(s, tissue=tissue, verbose=verbose)

                check_color_mesh = dataset.get_feature_map(
                    s, level, tissue, feature,
                    verbose=verbose
                )

                if check_color_mesh is None:
                    if cell_map.mapping is None:
                        print(f'\t{c.WARNING}Waring{c.ENDC}: Mapping not found - computing')
                        cell_map.map_cells(type=level)
                        cell_map.get_neighborhood(radius=50 if level == 'Membrane' else 40, type=level)
                    else:
                        print(f'\t{c.OKGREEN}Mapping{c.ENDC}: Found - skipping')
                        # cell_map.init_vars(type=level)
                    color_mesh = cell_map.color_mesh(feature, type=level)

                else:
                    print(f'\t{c.OKGREEN}Color mesh{c.ENDC}: Found - skipping')
                    color_mesh = trimesh.load(check_color_mesh)

                feature_maps.append(color_mesh)

            except Exception as e:
                print(f'\t{c.FAIL}Error{c.ENDC}: {e}')

                import traceback
                traceback.print_exc()

        # Feature map computation
        out_path = v.data_path + f'{group}/3DShape/Tissue/{tissue}/map/{feature}/atlas_{feature}.ply'
        out_path_aux = v.data_path + f'ATLAS/{tissue}/Features/{level}/{feature}/{group}_atlas_{feature}.ply'
        feature_map = FeatureMap(
            group, specimens, feature_maps,
            feature, tissue, level,
            verbose=verbose
        )

        if not os.path.exists('/'.join(out_path.split('/')[:-1])):
            os.makedirs('/'.join(out_path.split('/')[:-1]), exist_ok=True)

        feature_map.color_atlas(out_path, type=level, out_path_aux=out_path_aux)

    except getopt.GetoptError:
        print_usage()

