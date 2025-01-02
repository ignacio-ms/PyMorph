# Standard packages
import getopt
import os
import sys

import pandas as pd

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary import values as v
from filtering.mesh_filtering import run
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.utils.timer import LoadingBar
from auxiliary.data import imaging


def print_usage():
    print(
        'usage: run_filter_tissue.py -l <level> -t <tissue> -s <specimen> -g <group> -m <mesh_path> -t <tissue_path> -f <features_path> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}-l, --level{c.ENDC}: Level to use. (Default: Membrane)\n'
        f'{c.BOLD}-t, --tissue{c.ENDC}: Tissue to use. (Default: myocardium)\n'
        f'{c.BOLD}-s, --specimen{c.ENDC}: Specimen to process.\n'
        f'{c.BOLD}-g, --group{c.ENDC}: Group to process.\n'
        f'{c.BOLD}-m, --mesh_path{c.ENDC}: Path to mesh data.\n'
        f'{c.BOLD}-t, --tissue_path{c.ENDC}: Path to tissue data.\n'
        f'{c.BOLD}-f, --features_path{c.ENDC}: Path to features data.\n'
        f'{c.BOLD}-v, --verbose{c.ENDC}: Verbosity level. (Default: 0)\n'
    )
    sys.exit(2)


def get_group(ds, specimen):
    for group_name, specimens in ds.specimens.items():
        if specimen in specimens:
            return group_name
    return None


if __name__ == "__main__":
    argv = sys.argv[1:]
    data_path = v.data_path
    level = 'Membrane'
    tissue = 'myocardium'
    specimen = None
    group = None
    mesh_path = None
    tissue_path = None
    features_path = None
    all_remaining = False
    verbose = 0

    try:
        opts, args = getopt.getopt(argv, "hdl:t:s:g:m:t:f:a:v:", [
            "help", 'data_path=', "level=", "tissue=", "specimen=", "group=", "mesh_path=", "tissue_path=", "features_path=", "all=", "verbose="
        ])

        if len(opts) == 0:
            print_usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_usage()
                sys.exit()
            elif opt in ("-d", "--data_path"):
                data_path = arg_check(opt, arg, "-d", "--data_path", str, print_usage)
            elif opt in ("-l", "--level"):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
            elif opt in ("-t", "--tissue"):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ("-s", "--specimen"):
                specimen = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ("-g", "--group"):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ("-m", "--mesh_path"):
                mesh_path = arg_check(opt, arg, '-m', '--mesh_path', str, print_usage)
            elif opt in ("-t", "--tissue_path"):
                tissue_path = arg_check(opt, arg, '-t', '--tissue_path', str, print_usage)
            elif opt in ("-f", "--features_path"):
                features_path = arg_check(opt, arg, '-f', '--features_path', str, print_usage)
            elif opt in ("-v", "--verbose"):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            elif opt in ("-a", "--all"):
                all_remaining = arg_check(opt, arg, '-a', '--all', bool, print_usage)

            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()

    except getopt.GetoptError:
        print_usage()

    ds = HtDataset(data_path=data_path)

    # If mesh_path, tissue_path, and features_path are provided, process them directly
    if mesh_path is not None and tissue_path is not None and features_path is not None:
        if verbose > 0:
            print(f"{c.OKBLUE}Processing mesh data{c.ENDC}")

        run(mesh_path, tissue_path, features_path, verbose=verbose)

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
            specimens = [specimen for sublist in ds.specimens.values() for specimen in sublist]
        else:
            specimens = ds.check_meshes(level, tissue, verbose, filtered=True)

    print(f"{c.OKBLUE}Level{c.ENDC}: {level}")
    print(f"{c.OKBLUE}Tissue{c.ENDC}: {tissue}")

    for spec in specimens:
        try:
            if verbose > 0:
                print(f"{c.OKBLUE}Processing specimen{c.ENDC}: {spec}")

            mesh_path = ds.get_mesh_cell(spec, level, tissue, verbose, filtered=False)
            tissue_path = ds.get_mesh_tissue(spec, tissue, verbose)
            features_path = ds.get_features(spec, level, tissue, verbose, only_path=True, filtered=False)

            _, intersecting_cell_ids, non_intersecting_cell_ids = run(
                mesh_path, tissue_path, distance_threshold=25.0
            )

            # Remove non-intersecting cells from the features
            if features_path is not None:
                features = pd.read_csv(features_path)
                features = features[features['cell_id'].isin(intersecting_cell_ids)]

                path_split = features_path.split('/')
                path_split[-1] = f"Filtered/{path_split[-1]}"
                path_split = '/'.join(path_split)

                if not os.path.exists('/'.join(path_split.split('/')[:-1])):
                    os.makedirs('/'.join(path_split.split('/')[:-1]))

                features.to_csv(path_split, index=False)

                if verbose:
                    print(f"{c.OKGREEN}Filtered features saved to{c.ENDC}: {path_split}")

        except Exception as e:
            print(f"{c.FAIL}Error processing specimen{c.ENDC}: {spec}")
            print(e)
            if verbose:
                import traceback
                traceback.print_exc()
            continue
