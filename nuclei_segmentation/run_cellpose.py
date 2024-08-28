# Standard libraries
import os
import sys
import getopt

from scipy import ndimage
from skimage import exposure
import numpy as np

from cellpose import models, core
from csbdeep.utils import normalize as deep_norm

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset, find_specimen
from auxiliary.utils.timer import LoadingBar, timed
from auxiliary.data import imaging

from filtering.cardiac_region import get_margins, crop_img, restore_img
from feature_extraction.feature_extractor import filter_by_volume, filter_by_margin

# Configurations
use_gpu = core.use_gpu()
print(f"GPU activated: {use_gpu}")

from cellpose.io import logger_setup
logger_setup();


def load_img(img_path, equalize_img=True, normalize_img=True, verbose=0):
    """
    Load and normalize image.
    :param img_path: Path to image.
    :param normalize_img: Normalize image. (Default: True)
    :param equalize_img: Perform histogram equalization on image. (Default: True)
    :param verbose: Verbosity level.
    :return: Image.
    """
    img = imaging.read_image(img_path, axes='ZYX', verbose=verbose)
    if verbose:
        print(f'{c.OKBLUE}Loaded image{c.ENDC}: {img_path}')
        print(f'{c.BOLD}Image shape{c.ENDC}: {img.shape}')

    if equalize_img:
        if verbose:
            print(f'{c.OKBLUE}Equalizing image{c.ENDC}...')

        # Rescale image data to range [0, 1]
        img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))
        img = (img - img.min()) / (img.max() - img.min())

        img = exposure.equalize_hist(img)

        vmin, vmax = np.percentile(img, q=(5, 95))
        img = exposure.rescale_intensity(img, in_range=(vmin, vmax))

    if normalize_img:
        if verbose:
            print(f'{c.OKBLUE}Normalizing image{c.ENDC}...')

        img = deep_norm(img, 2, 98, axis=(0, 1, 2))

    # Median filter
    if verbose:
        print(f'{c.OKBLUE}Applying median filter{c.ENDC}...')

    img = ndimage.median_filter(img, size=(3, 3, 3))
    # img = ndimage.median_filter(img, size=(3, 3, 3))

    return img


def load_model(model_type='nuclei', model_path=None):
    """
    Load cellpose model.
    :param model_type: Type of model to load. (nuclei, cyto)
        -nuclei: nuclei model
        -(cyto, cyto2, cyto3): cytoplasm model
        -tissuenet_cp3: tissuenet dataset.
        -livecell_cp3: livecell dataset
        -yeast_PhC_cp3: YEAZ dataset
        -yeast_BF_cp3: YEAZ dataset
        -bact_phase_cp3: omnipose dataset
        -bact_fluor_cp3: omnipose dataset
        -deepbacs_cp3: deepbacs dataset
        -cyto2_cp3: cellpose dataset
    :param model_path: Path to model. (Default: None)
    :return: Model.
    """
    # Not implemented
    if model_path is None:
        model_path = '../models/cellpose_models/'

    print(f'{c.OKBLUE}Loading model{c.ENDC}: {model_type}')
    if model_type in ['nuclei', 'cyto', 'cyto2', 'cyto3']:
        return models.Cellpose(gpu=use_gpu, model_type=model_type)

    return models.CellposeModel(model_type, diam_mean=17)

def run(
        model, img,
        diameter=None, channels=None,
        anisotropy=1,
        verbose=0
):
    """
    Run cellpose on image.
    :param model: Cellpose model.
    :param img: Image.
    :param diameter: Diameter of nuclei. (Default: 0)
    :param channels: Channels to use. (Default: [0, 0])
    :param anisotropy: Anisotropy of image for sampling difference between XY and Z. (Default: 1)
    :param verbose: Verbosity level.
    :return: Masks.
    """

    if diameter is None:
        diameter = 17

    if channels is None:
        channels = [0, 0]

    if isinstance(model, models.Cellpose):
        masks, _, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            normalize=False,
            anisotropy=anisotropy,
            do_3D=True,
            cellprob_threshold=.25,
            # stitch_threshold=.45,
            # flow_threshold=.35,
        )
    else:
        masks, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            normalize=False,
            anisotropy=anisotropy,
            do_3D=True,
            cellprob_threshold=.25,
            # stitch_threshold=.5,
            # flow_threshold=.35,
        )

    if verbose:
        print(f'{c.OKGREEN}Masks shape{c.ENDC}: {masks.shape}')

    return masks


@timed
def predict(
    img_path, img_path_out,
    model,
    normalize, equalize,
    diameter, channels,
    tissue, verbose
):
    """
    Predict nuclei on image.
    :param img_path: Path to image.
    :param img_path_out: Path to save masks.
    :param model: Model to use.
    :param normalize: Normalize image.
    :param equalize: Equalize image.
    :param diameter: Diameter of nuclei.
    :param channels: Channels to use.
    :param tissue: Tissue to crop image.
    :param verbose: Verbosity level
    """
    img = load_img(
        img_path,
        equalize_img=equalize,
        normalize_img=normalize,
        verbose=verbose
    )
    model = load_model(model_type=model)

    # Set anisotropy
    metadata, _ = imaging.load_metadata(img_path)
    anisotropy = metadata['x_res'] / metadata['z_res']
    print(
        f'{c.OKBLUE}Image resolution{c.ENDC}: \n'
        f'X: {metadata["x_res"]} um/px\n'
        f'Y: {metadata["y_res"]} um/px\n'
        f'Z: {metadata["z_res"]} um/px'
    )
    print(f'{c.OKBLUE}Anisotropy{c.ENDC}: {anisotropy}')

    # Crop img by tissue
    specimen = find_specimen(img_path)
    lines_path, _ = dataset.read_line(specimen)

    if tissue is not None:
        margins = get_margins(
            line_path=lines_path, img_path=img_path,
            tissue=tissue, verbose=verbose
        )

        # Set margins to ZYX
        margins = (
            (margins[0][2], margins[0][1], margins[0][0]),
            (margins[1][2], margins[1][1], margins[1][0])
        )
        img = crop_img(img, margins, verbose=verbose)

    # Run segmentation
    masks = run(
        model, img,
        diameter=diameter, channels=channels,
        anisotropy=anisotropy,
        verbose=verbose
    )

    masks = filter_by_volume(masks, percentile=99, verbose=verbose)

    if tissue is not None:
        masks = filter_by_margin(masks, verbose=verbose)

        # Restore original shape
        masks = restore_img(
            masks, margins,
            depth=metadata['z_size'], resolution=metadata['x_size'],
            axes='ZYX', verbose=verbose
        )

    if isinstance(tissue, list):
        tissue = '_'.join(tissue)

    img_path_out = img_path_out.replace('.nii.gz', f'_{tissue}.nii.gz')
    imaging.save_nii(masks, img_path_out, verbose=verbose, axes='ZYX')


def print_usage():
    """
    Print usage of script.
    """
    print(
        'usage: run_cellpose.py -i <image> -s <specimen> -gr <group> -m <model> -n <normalize> -e <equalize> -d <diameter> -c <channels> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}<image>{c.ENDC}: Path to image.\n'
        f'{c.BOLD}<specimen>{c.ENDC}: Specimen to predict.\n'
        f'\tIf <image> is not provided, <specimen> is used.\n'
        f'\tSpecimen must be in the format: XXXX_EY\n'
        f'{c.BOLD}<group>{c.ENDC}: Group to predict all remaining images.\n'
        f'\tIf <group> is not provided, <image> is used.\n'
        f'\tIn not <group> nor <image> nor <specimen> is provided, all remaining images are predicted.\n'
        f'{c.BOLD}<model>{c.ENDC}: Model to use. (Default: nuclei)\n'
        f'{c.BOLD}<normalize>{c.ENDC}: Normalize image. (Default: True)\n'
        f'{c.BOLD}<equalize>{c.ENDC}: Histogram equalization over image. (Default: True)\n'
        f'{c.BOLD}<diameter>{c.ENDC}: Diameter of nuclei. (Default: 17)\n'
        f'{c.BOLD}<channels>{c.ENDC}: Channels to use. (Default: [0, 0])\n'
        f'\tChannels must be a list of integers.\n'
        f'\t0 = Grayscale - 1 = Red - 2 = Green - 3 = Blue\n'
        f'{c.BOLD}<tissue>{c.ENDC}: Tissue to crop image. (Default: Myocardium)\n'
        f'{c.BOLD}<verbose>{c.ENDC}: Verbosity level. (Default: 0)\n'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path, img, spec, group, model, normalize, equalize, diameter, channels, tissue, verbose = None, None, None, None, 'nuclei', True, True, None, None, None,1

    try:
        opts, args = getopt.getopt(argv, "hp:i:s:g:m:n:e:d:c:t:v:", [
            'help', "data_path=", "image=", "specimen=", "group=", "model=", "normalize=", "equalize=", "diameter=", "channels=", "tissue=", "verbose="
        ])

        if len(opts) == 0 or len(opts) > 8:
            print_usage()

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_usage()
            elif opt in ("-p", "--data_path"):
                data_path = arg_check(opt, arg, "-p", "--data_path", str, print_usage)
            elif opt in ("-i", "--image"):
                img = arg_check(opt, arg, "-i", "--image", str, print_usage)
            elif opt in ("-s", "--specimen"):
                spec = arg_check(opt, arg, "-s", "--specimen", str, print_usage)
            elif opt in ("-g", "--group"):
                group = arg_check(opt, arg, "-g", "--group", str, print_usage)
            elif opt in ("-m", "--model"):
                model = arg_check(opt, arg, "-m", "--model", str, print_usage)
            elif opt in ("-n", "--normalize"):
                normalize = arg_check(opt, arg, "-n", "--normalize", bool, print_usage)
            elif opt in ("-e", "--equalize"):
                equalize = arg_check(opt, arg, "-e", "--equalize", bool, print_usage)
            elif opt in ("-d", "--diameter"):
                diameter = arg_check(opt, arg, "-d", "--diameter", int, print_usage)
            elif opt in ("-c", "--channels"):
                channels = arg_check(opt, arg, "-c", "--channels", list, print_usage)
            elif opt in ("-t", "--tissue"):
                tissue = arg_check(opt, arg, "-t", "--tissue", str, print_usage)
            elif opt in ("-v", "--verbose"):
                verbose = arg_check(opt, arg, "-v", "--verbose", int, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()

        if data_path is None:
            data_path = v.data_path

        if normalize is None:
            normalize = True

        if equalize is None:
            equalize = True

        if model is None:
            print(f'{c.BOLD}Model not provided{c.ENDC}: Running with default model (nuclei)')
            model = 'nuclei'

        if tissue is None:
            print(f'{c.BOLD}Tissue not provided{c.ENDC}: Running with default tissues (myocardium, splanchnic)')
            # tissue = ['myocardium', 'splanchnic']

        dataset = HtDataset(data_path=data_path)

        if group is not None:
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

            img_paths_out = dataset.missing_nuclei_out
            img_paths_out = [
                img_path_out for img_path_out in img_paths_out if any(
                    specimen in img_path_out for specimen in specimens
                )
            ]

        elif img is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on image{c.ENDC}: {img}')

            img_paths = [data_path + img]
            img_paths_out = [img_paths[0].replace('RawImages', 'Segmentation')]
            img_paths_out = [
                img_paths_out[0].replace(
                    '_DAPI_decon_0.5',
                    f'_{model}_mask'
                )
            ]

        elif spec is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {spec}')

            img_path, img_path_out = dataset.read_specimen(spec, verbose=verbose)
            img_paths = [img_path]
            img_paths_out = [img_path_out]

        else:
            if verbose:
                print(f'Running prediction for all remaining images')

            dataset.check_segmentation(verbose=verbose)
            img_paths = dataset.missing_nuclei
            img_paths_out = dataset.missing_nuclei_out

        if len(img_paths) == 0:
            print(f'{c.WARNING}No images to process{c.ENDC}')
            sys.exit(2)

        bar = LoadingBar(len(img_paths), length=50)
        for img_path, img_path_out in zip(img_paths, img_paths_out):
            predict(
                img_path, img_path_out,
                model=model,
                normalize=normalize, equalize=equalize,
                diameter=diameter, channels=channels,
                tissue=tissue, verbose=verbose
            )

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
