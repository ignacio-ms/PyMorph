# Standard packages
import os
import sys

import getopt

import numpy as np
from csbdeep.utils import normalize as csb_normalize
from skimage.exposure import rescale_intensity

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util import values as v
from util.misc.bash import arg_check
from util.misc.colors import bcolors as c
from util.misc.timer import timed
from util.data import imaging

from util.data.dataset_ht import HtDataset, find_group

from filtering import cardiac_region as cr
from feature_extraction.feature_extractor import extract, filter_connected_components_with_size


@timed
def run(ds, s, type, tissue=None, norm=True, from_filtered=False, verbose=0):
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
    metadata, _ = imaging.load_metadata(path_raw)
    seg_img = imaging.read_image(path_seg, verbose=verbose).astype(np.uint32)

    if norm:
        print(f'{c.OKBLUE}Normalizing image...{c.ENDC}')
        # raw_img = csb_normalize(raw_img, 1, 99.8, axis=(0, 1))

        for z in range(raw_img.shape[2]):
            min_v, max_v = np.min(raw_img[..., z]), np.max(raw_img[..., z])

            raw_img[..., z] = rescale_intensity(
                raw_img[..., z], in_range=(min_v, max_v), out_range=(0, 1)
            )

    if from_filtered:
        try:
            path_seg, _ = ds.read_specimen(s, type, 'Segmentation', filtered=True, verbose=verbose)
            seg_img = imaging.read_image(path_seg, verbose=verbose).astype(np.uint32)
        except Exception as e:
            print(f'{c.WARNING}Warning{c.ENDC}: No filtered segmentation found. Filtering without connected components analysis.')
            path_seg, _ = ds.read_specimen(s, type, 'Segmentation', filtered=False, verbose=verbose)
            seg_img = imaging.read_image(path_seg, verbose=verbose).astype(np.uint32)

            if tissue:
                seg_img = cr.filter_by_tissue(seg_img, lines, tissue, 2, verbose=verbose)

            path_split = path_seg.split('/')
            path_split[-1] = f'Filtered/{path_split[-1].replace(".nii.gz", f"_{tissue}.nii.gz")}'
            path_seg = '/'.join(path_split)
            if not os.path.exists('/'.join(path_split[:-1])):
                os.makedirs('/'.join(path_split[:-1]), exist_ok=True)

            imaging.save_nii(
                seg_img, path_seg,
                verbose=verbose
            )

    else:
        if tissue:
            seg_img = cr.filter_by_tissue(seg_img, lines, tissue, 2, verbose=verbose)

        seg_img = filter_connected_components_with_size(
            seg_img, min_size=20, max_size=4500,
            verbose=verbose
        )

        path_split = path_seg.split('/')
        path_split[-1] = f'Filtered/{path_split[-1].replace(".nii.gz", f"_{tissue}.nii.gz")}'
        path_seg = '/'.join(path_split)
        if not os.path.exists('/'.join(path_split[:-1])):
            os.makedirs('/'.join(path_split[:-1]), exist_ok=True)

        imaging.save_nii(
            seg_img, path_seg,
            verbose=verbose
        )

    return extract(
        seg_img, raw_img, lines, path_raw,
        metadata=metadata,
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

    data_path, spec, group, type, tissue, norm, verbose = None, None, None, None, None, True, 0

    try:
        opts, args = getopt.getopt(argv, 'hd:s:g:t:l:n:v:', [
            'help', 'data_path=', 'specimen=', 'group=', 'type=', 'tissue=', 'norm=', 'verbose='
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
            elif opt in ('-n', '--norm'):
                norm = arg_check(opt, arg, '-n', '--norm', str, print_usage)
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
            features = run(ds, spec, type, tissue=tissue, verbose=verbose, from_filtered=True)

            out_dir = os.path.join(
                data_path, find_group(spec), 'Features',
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
                    out_path = todo_out_paths[todo_specimens.index(s)]
                    features = run(ds, s, type, tissue=tissue, norm=True, verbose=verbose)

                    features.to_csv(out_path, index=False)
                    print(f'{c.OKGREEN}Features saved{c.ENDC}: {out_path}')
                except Exception as e:
                    # print(f'{c.FAIL}Error{c.ENDC}: {e}')
                    continue

    except getopt.GetoptError:
        print_usage()
