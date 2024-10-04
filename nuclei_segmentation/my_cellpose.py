from scipy import ndimage

from cellpose import models, core
from cellpose.io import logger_setup

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset, find_specimen
from auxiliary.utils.timer import LoadingBar, timed
from auxiliary.data import imaging

from filtering.cardiac_region import get_margins, crop_img, restore_img
from feature_extraction.feature_extractor import filter_by_volume, filter_by_margin
from nuclei_segmentation import preprocessing

# Configurations
use_gpu = core.use_gpu()
print(f"GPU activated: {use_gpu}")
logger_setup()


def load_img(img_path, pipeline=None, test_name=None, verbose=0, **kwargs):
    """
    Load image with preprocessing.
    :param img_path: Path to image.
    :param pipeline: Preprocessing pipeline.
        Possible steps:
        normalization, equalization, anisodiff, bilateral,
        isotropy, gaussian, median, gamma, rescale_intensity
    :param test_name:
    :param verbose:
    :param kwargs:
    :return:
    """
    preprocess = preprocessing.Preprocessing(pipeline=pipeline)
    return preprocess.run(
        img_path, test_name=test_name,
        verbose=verbose, **kwargs
    )


def load_model(model_type='nuclei'):
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
    :return: Model.
    """
    print(f'{c.OKBLUE}Loading model{c.ENDC}: {model_type}')
    if model_type in ['nuclei', 'cyto', 'cyto2', 'cyto3']:
        return models.Cellpose(gpu=use_gpu, model_type=model_type)

    return models.CellposeModel(model_type, diam_mean=17)


def run(
        model, img,
        diameter=None, channels=None,
        anisotropy=1,
        do_3D=False,
        stitch_threshold=.6,
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
            anisotropy=1,
            do_3D=do_3D,
            # cellprob_threshold=.5,
            stitch_threshold=stitch_threshold,
            # flow_threshold=.45,
        )

    else:
        masks, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            normalize=False,
            anisotropy=anisotropy,
            do_3D=True,
            # cellprob_threshold=.25,
            stitch_threshold=.6,
            # flow_threshold=.35,
        )

    if verbose:
        print(f'{c.OKGREEN}Masks shape{c.ENDC}: {masks.shape}')

    return masks


@timed
def predict(
    img_path, img_path_out,
    model, dataset,
    diameter, channels,
    tissue, verbose,
    **kwargs
):
    if 'test_name' in kwargs:
        img_path_out = img_path_out.replace('.nii.gz', f'_{kwargs["test_name"]}.nii.gz')

    img = load_img(
        img_path,
        test_name=kwargs['test_name'] if 'test_name' in kwargs else None,
        pipeline=kwargs['pipeline'] if 'pipeline' in kwargs else None,
        verbose=verbose
    )
    model = load_model(model_type=model)

    # Set anisotropy
    metadata, _ = imaging.load_metadata(img_path)
    anisotropy = metadata['z_res'] / metadata['x_res']
    inverse_anisotropy = 1 / anisotropy
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
        diameter=diameter,
        channels=channels,
        do_3D=kwargs['do_3D'] if 'do_3D' in kwargs else False,
        stitch_threshold=kwargs['stitch_threshold'] if 'stitch_threshold' in kwargs else .6,
        verbose=verbose
    )

    masks = filter_by_volume(masks, percentile=97, verbose=verbose)

    # Anisotropic recosntruction
    masks = ndimage.zoom(masks, (inverse_anisotropy, 1, 1), order=0)

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

    img_path_out = img_path_out.replace(
        '.nii.gz', f'_{tissue if tissue is not None else "all"}.nii.gz'
    )
    imaging.save_nii(masks, img_path_out, verbose=verbose, axes='ZYX')
