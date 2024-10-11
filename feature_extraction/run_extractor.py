# Standard packages
import os
import re
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
from auxiliary.utils.timer import timed
from auxiliary.data import imaging

from auxiliary.data.dataset_ht import HtDataset, find_group

from filtering import cardiac_region as cr
from feature_extraction.feature_extractor import extract, filter_by_volume


@timed
def run(ds, s, type, tissue=None, verbose=0):
    """
    Run feature extraction.
    :param ds: HtDataset object.
    :param s: Specimen.
    :param type: Type of image. (Nuclei, Membrane)
    :param verbose: Verbosity level. (default: 0)
    :return: DataFrame with features.
    """
    path_lines, _ = ds.read_line(s, verbose=verbose)
    path_raw, _ = ds.read_specimen(s, type, 'RawImages', verbose=verbose)
    path_seg, _ = ds.read_specimen(s, type, 'Segmentation', verbose=verbose)

    lines = imaging.read_image(path_lines, verbose=verbose)
    raw_img = imaging.read_image(path_raw, verbose=verbose)
    seg_img = imaging.read_image(path_seg, verbose=verbose)

    seg_img = filter_by_volume(seg_img, verbose=verbose)
    if tissue:
        seg_img = cr.filter_by_tissue(seg_img, lines, tissue, 1, verbose=verbose)

    return extract(
        seg_img, raw_img, lines, path_raw,
        f_type=type,
        verbose=verbose
    )


def print_usage():
    print(
        'Usage: run_extractor.py -d <data_path> -s <specimen> -g <group> -t <type> -v <verbose>\n'
        f'\n\n{c.BOLD}Options:{c.ENDC}\n'
        f'{c.BOLD}<data_path>{c.ENDC}: Path to data directory.\n'
        f'{c.BOLD}<specimen>{c.ENDC}: Specimen to run prediction on.\n'
        f'{c.BOLD}<group>{c.ENDC}: Group to run prediction on.\n'
        f'{c.BOLD}<type>{c.ENDC}: Type of image (Nuclei, Membrane).\n'
        f'{c.BOLD}<tissue>{c.ENDC}: Tissue to filter by.\n'
        f'{c.BOLD}<verbose>{c.ENDC}: Verbosity level.\n'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path, spec, group, type, tissue, verbose = None, None, None, None, None, 0

    try:
        opts, args = getopt.getopt(argv, 'hd:s:g:t:l:v:', [
            'help', 'data_path=', 'specimen=', 'group=', 'type=', 'tissue=', 'verbose='
        ])

        if len(opts) == 0 or len(opts) > 5:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-d', '--data_path'):
                data_path = arg_check(opt, arg, '-d', '--data_path', str, print_usage)
            elif opt in ('-s', '--specimen'):
                spec = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-t', '--type'):
                type = arg_check(opt, arg, '-t', '--type', str, print_usage)
            elif opt in ('-l', '--tissue'):
                tissue = arg_check(opt, arg, '-l', '--tissue', str, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f'Invalid option: {opt}')
                print_usage()

        if data_path is None:
            data_path = v.data_path

        if type is None:
            print(f'{c.BOLD}Type not provided{c.ENDC}: Running with default type (Nuclei)')
            type = 'Nuclei'

        if tissue is not None and tissue not in v.lines.keys():
            print(f'{c.FAIL}Invalid tissue{c.ENDC}: {tissue}')
            print_usage()

        if group is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on group{c.ENDC}: {group}')

            ds = HtDataset(data_path=data_path)
            todo_specimens, todo_out_paths = ds.check_features(verbose=verbose, type=type, tissue=tissue)

            specimens = v.specimens[group]

            # Filter by group with its corresponding out paths
            idx = [i for i, s in enumerate(specimens) if s in todo_specimens]
            todo_specimens = [specimens[i] for i in idx]
            todo_out_paths = [todo_out_paths[i] for i in idx]

            if len(todo_specimens) == 0:
                print(f'{c.OKGREEN}All features extracted for group: {c.BOLD}{c.ENDC}')
                sys.exit(0)

            for s in todo_specimens:
                if verbose:
                    print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {s}')

                try:
                    features = run(ds, s, type, tissue=tissue, verbose=verbose)
                    features.to_csv(todo_out_paths[todo_specimens.index(s)], index=False)
                except Exception as e:
                    print(f'{c.FAIL}Error{c.ENDC}: {e}')

        elif spec is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {spec}')

            ds = HtDataset(data_path=data_path)
            features = run(ds, spec, type, tissue=tissue, verbose=verbose)

            out_dir = os.path.join(
                v.data_path, find_group(spec), 'Features',
                f'2019{spec}_cell_properties_radiomics_{type}_{tissue}.csv'
            )

            features.to_csv(out_dir, index=False)

        else:
            print(f'{c.FAIL}No group or specimen provided, running all{c.ENDC}')

            ds = HtDataset(data_path=data_path)
            todo_specimens, todo_out_paths = ds.check_features(
                verbose=verbose, type=type, tissue=tissue
            )

            if len(todo_specimens) == 0:
                print(f'{c.OKGREEN}All features extracted{c.ENDC}')
                sys.exit(0)

            for s in todo_specimens:
                if verbose:
                    print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {s}')

                try:
                    features = run(ds, s, type, tissue=tissue, verbose=verbose)
                    features.to_csv(todo_out_paths[todo_specimens.index(s)], index=False)
                except Exception as e:
                    print(f'{c.FAIL}Error{c.ENDC}: {e}')

    except getopt.GetoptError:
        print_usage()
