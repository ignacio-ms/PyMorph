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
from auxiliary.data.dataset_ht import HtDataset

from meshes.mesh_reconstruction import run
from filtering.cardiac_region import filter_by_tissue


def print_usage():
    """
    Print usage of script.
    """
    print(
        'usage: run_mesh_reconstruction.py -l <level> -t <tissue> -s <specimen> -g <group> -e <segmentation_path> -r <raw_path> -o <path_out> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}-l, --level{c.ENDC}: Level to use. (Default: Membrane)\n'
        f'{c.BOLD}-t, --tissue{c.ENDC}: Tissue to use. (Default: myocardium)\n'
        f'{c.BOLD}-s, --specimen{c.ENDC}: Specimen to process.\n'
        f'{c.BOLD}-g, --group{c.ENDC}: Group to process.\n'
        f'{c.BOLD}-e, --segmentation_path{c.ENDC}: Path to segmentation data.\n'
        f'{c.BOLD}-r, --raw_path{c.ENDC}: Path to raw data.\n'
        f'{c.BOLD}-o, --path_out{c.ENDC}: Output path for the mesh file.\n'
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
    level = 'Membrane'
    tissue = 'myocardium'
    specimen = None
    group = None
    segmentation_path = None
    raw_path = None
    path_out = None
    verbose = 0

    try:
        opts, args = getopt.getopt(argv, 'hl:t:s:g:e:r:o:v:', [
            'help', 'level=', 'tissue=', 'specimen=', 'group=', 'segmentation_path=', 'raw_path=', 'path_out=', 'verbose='
        ])

        if len(opts) == 0:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-l', '--level'):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-s', '--specimen'):
                specimen = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-e', '--segmentation_path'):
                segmentation_path = arg_check(opt, arg, '-e', '--segmentation_path', str, print_usage)
            elif opt in ('-r', '--raw_path'):
                raw_path = arg_check(opt, arg, '-r', '--raw_path', str, print_usage)
            elif opt in ('-o', '--path_out'):
                path_out = arg_check(opt, arg, '-o', '--path_out', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()
    except getopt.GetoptError:
        print_usage()

    ds = HtDataset()

    # If segmentation_path and raw_path are provided, process them directly
    if segmentation_path is not None and raw_path is not None:
        if path_out is None:
            print(f"{c.FAIL}Output path is required when using segmentation and raw paths directly.{c.ENDC}")
            sys.exit(2)
        print(f"{c.OKBLUE}Processing provided segmentation and raw paths{c.ENDC}")
        lines_path = None  # Adjust as needed if lines_path is relevant
        run(segmentation_path, path_out, raw_path, lines_path, tissue, level, verbose=verbose)
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
        # Process all specimens
        specimens = []
        for group_name, group_specimens in ds.specimens.items():
            specimens.extend(group_specimens)

    print(f'{c.OKBLUE}Level{c.ENDC}: {level}')
    print(f'{c.OKBLUE}Tissue{c.ENDC}: {tissue}')

    for spec in specimens:
        print(f'{c.BOLD}Specimen{c.ENDC}: {spec}')

        try:
            img_path, _ = ds.read_specimen(
                spec, level, 'Segmentation',
                verbose=verbose
            )

            img_path_raw, _ = ds.read_specimen(spec, level, type='RawImages', verbose=verbose)
            lines_path, _ = ds.read_line(spec, verbose=verbose)

            # Get group name if not provided
            group_name = group if group is not None else get_group(ds, spec)
            if group_name is None:
                print(f'{c.FAIL}Cannot determine group for specimen{c.ENDC}: {spec}')
                continue

            # Use provided path_out if available, else construct default path
            if path_out is not None:
                output_path = path_out
            else:
                output_path = os.path.join(v.data_path, f'{group_name}/3DShape/{level}/{tissue}/2019{spec}_{tissue}.ply')

            run(img_path, output_path, img_path_raw, lines_path, tissue, level, verbose=verbose)

        except Exception as e:
            print(f'{c.FAIL}Error{c.ENDC}: {e}')
            if verbose:
                import traceback
                traceback.print_exc()
            continue
