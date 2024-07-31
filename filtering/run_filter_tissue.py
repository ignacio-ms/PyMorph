# Standard packages
import getopt
import os
import sys

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary import values as v
from filtering.cardiac_region import filter_by_tissue
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.utils.timer import LoadingBar
from auxiliary.data import imaging


def run():
    """
    Filter segmented 3D image by tissue.
    :return:
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hp:s:l:i:o:g:t:v:", [
            "help", 'data_path=', 'specimen=', 'level=', "img=", "output=", 'group=', "tissue=", "verbose="
        ])

        if len(opts) == 0 or len(opts) > 7:
            print_usage()
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    data_path, spec, img, output_path, group, tissue, level, verbose = None, None, None, None, None, None, None, None

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_usage()
        elif opt in ('-p', '--data_path'):
            data_path = arg_check(opt, arg, "-p", "--data_path", str, print_usage)
        elif opt in ('-s', '--specimen'):
            spec = arg_check(opt, arg, "-s", "--specimen", str, print_usage)
        elif opt in ('-i', '--img'):
            img = arg_check(opt, arg, "-i", "--img", str, print_usage)
        elif opt in ('-o', '--output'):
            output_path = arg_check(opt, arg, "-o", "--output", str, print_usage)
        elif opt in ('-g', '--group'):
            group = arg_check(opt, arg, "-g", "--group", str, print_usage)
        elif opt in ('-t', '--tissue'):
            tissue = arg_check(opt, arg, "-t", "--tissue", str, print_usage)
        elif opt in ('-l', '--level'):
            level = arg_check(opt, arg, "-l", "--level", str, print_usage)
        elif opt in ('-v', '--verbose'):
            verbose = arg_check(opt, arg, "-v", "--verbose", int, print_usage)
        else:
            print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
            print_usage()

    if tissue is None:
        print(f'{c.BOLD}Tissue not provided{c.ENDC}: Filtering by default tissue (myocardium)')
        tissue = 'myocardium'

    if level is None:
        print(f'{c.BOLD}Level not provided{c.ENDC}: Filtering by default level (Nuclei)')
        level = 'Nuclei'

    if data_path is None:
        data_path = v.data_path

    ds = HtDataset(data_path=data_path)

    if img is not None:
        if verbose:
            print(f'{c.OKBLUE}Filtering image:{c.ENDC} {img}')

        img_paths = [img]
        img_paths_out = [
            output_path if output_path else img
            .replace('.nii.gz', f'_{tissue}.nii.gz')
            .replace(f'{level}', f'{level}/{tissue}')
            .replace('.tif', f'_{tissue}.tif')
        ]

        aux = img.split('/')[-1].split('_')[0].split('2019')[-1]
        aux_e = img.split('/')[-1].split('_')[1]
        spec = f'{aux}_{aux_e}'
        line_paths = [ds.read_line(spec, verbose=verbose)[0]]

    elif spec is not None:
        if verbose:
            print(f'{c.OKBLUE}Filtering specimen:{c.ENDC} {spec}')

        img_path, _ = ds.read_specimen(spec, type='Segmentation', level=level, verbose=verbose)
        img_paths = [img_path]
        img_paths_out = [
            img_path
            .replace('.nii.gz', f'_{tissue}.nii.gz')
            .replace(f'{level}', f'{level}/{tissue}')
            .replace('.tif', f'_{tissue}.tif')
        ]

        line_paths = [ds.read_line(spec, verbose=verbose)[0]]

    elif group is not None:
        if verbose:
            print(f'{c.OKBLUE}Filtering group:{c.ENDC} {group}')

        ds.read_img_paths(type='Segmentation')

        if group not in ds.specimens.keys():
            print(f'{c.FAIL}Invalid group{c.ENDC}: {group}')
            print(f'{c.BOLD}Available groups{c.ENDC}: {list(ds.specimens.keys())}')
            sys.exit(2)

        specimens = ds.specimens[group]
        img_paths = ds.segmentation_nuclei_path if level == 'Nuclei' else ds.segmentation_membrane_path
        img_paths = [
            img_path for img_path in img_paths if any(
                specimen in img_path for specimen in specimens
            )
        ]

        img_paths_out = [
            img_path_out.replace(f'{level}', f'{level}/{tissue}')
            .replace('.nii.gz', f'_{tissue}.nii.gz')
            .replace('.tif', f'_{tissue}.tif')
            for img_path_out in img_paths
        ]

        line_paths = [ds.read_line(spec, verbose=verbose)[0] for spec in specimens]

    if len(img_paths) == 0 or len(img_paths_out) == 0:
        print(f'{c.FAIL}No images found{c.ENDC}')
        sys.exit(2)

    if len(img_paths) != len(img_paths_out):
        print(f'{c.FAIL}Number of input and output images must be the same{c.ENDC}')
        sys.exit(2)

    bar = LoadingBar(len(img_paths))
    for img_path, img_path_out, line_path in zip(img_paths, img_paths_out, line_paths):
        img = imaging.read_nii(img_path, verbose=verbose) if img_path.endswith('.nii.gz') else imaging.read_tiff(img_path, verbose=verbose)
        lines = imaging.read_nii(line_path, verbose=verbose)

        filtered_img = filter_by_tissue(
            img, lines, tissue_name=tissue,
            dilate=2, dilate_size=3,
            verbose=verbose
        )

        bar.update()

        try:
            imaging.save_prediction(filtered_img, img_path_out, verbose=verbose)
        except FileNotFoundError:
            print(f'\n{c.WARNING}Folder not found{c.ENDC}: Creating directory for {c.BOLD}{tissue}{c.ENDC} filtered images')

            if group is None:
                group = img_path.split('/')[-4]
            tissue_folder = data_path + f'{group}/Segmentation/{level}/{tissue}/'

            os.makedirs(tissue_folder, exist_ok=True)
            imaging.save_prediction(filtered_img, img_path_out, verbose=verbose)


def print_usage():
    print(
        "Usage: python run_filter_tissue.py -i <img> -t <tissue> -l <level> -o <output> -p <data_path> -s <specimen> -g <group> -v <verbose>\n"
        "Options:\n"
        '<img> Input segmented image path (nii.gz or tiff).\n'
        '<tissue> Tissue to filter by. (Default: myocardium)\n'
        '<level> Nuclei level or Membrane level. (Default: Nuclei)\n'
        '<output> Output path for filtered image. (Default: input image path with tissue name)\n'
        '<data_path> Path to data directory. (Default: v.data_path)\n'
        '<specimen> Specimen to filter. (Default: None)\n'
        '<group> Group of specimens to filter. (Default: None)\n'
        '<verbose> Verbosity level. (Default: 0)\n'
    )
    sys.exit(2)


if __name__ == "__main__":
    run()
