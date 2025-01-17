import torch

from scipy import ndimage
from skimage import exposure
from csbdeep.utils import normalize as deep_norm

import subprocess

from utils.misc.colors import bcolors as c
from utils.data.dataset_ht import find_specimen
from utils.data import imaging

from feature_extraction.feature_extractor import filter_by_margin
from filtering.cardiac_region import get_margins, crop_img, restore_img
from nuclei_segmentation.processing import postprocessing

# Configurations
use_gpu = torch.cuda.is_available()
print(f"GPU activated: {use_gpu}")


def modify_yaml_path(new_path, file_path='membrane_segmentation/config.yaml'):
    subprocess.run(
        f"sed -i 's|^path:.*|path: {new_path}|' {file_path}",
        shell=True, check=True
    )


def predict(img_path, img_path_out, tissue, dataset, verbose=0):
    # Image preprocessing
    img = imaging.read_image(img_path, verbose=verbose)

    print(f'{c.OKBLUE}Pre-processing image{c.ENDC}:')
    print(f'\t{c.OKBLUE}Histogram equalization{c.ENDC}...')
    img = exposure.equalize_hist(img)

    print(f'\t{c.OKBLUE}Normalization{c.ENDC}...')
    img = deep_norm(img, 1, 99.8, axis=(0, 1, 2))

    print(f'\t{c.OKBLUE}Median filter{c.ENDC}...')
    img = ndimage.median_filter(img, size=3)

    # Crop image by tissue margin
    metadata, _ = imaging.load_metadata(img_path)

    specimen = find_specimen(img_path)
    lines_path, _ = dataset.read_line(specimen)

    margins = get_margins(
        line_path=lines_path, img_path=img_path,
        tissue=tissue, verbose=verbose
    )
    img = crop_img(img, margins, verbose=verbose)

    # Pipeline
    if verbose:
        print(f'{c.OKBLUE}Transforming image {c.BOLD} .nii.gz -> .h5{c.ENDC}: {img_path}')
    imaging.nii2h5(img, img_path.replace('.nii.gz', '.h5'), verbose=verbose)

    modify_yaml_path(img_path.replace('.nii.gz', '.h5'))
    subprocess.run(f'plantseg --config membrane_segmentation/config.yaml', shell=True, check=True)

    # Load segmented image and correct axes
    path = '/'.join(img_path.split('/')[:-1] + [
        'PreProcessing', 'confocal_3D_unet_sa_meristem_cells',
        'GASP', 'PostProcessing',
        img_path.split('/')[-1].replace('.nii.gz', '_predictions_gasp_average.tiff')
    ])

    masks = imaging.read_image(
        path,
        axes='ZYX', verbose=verbose
    )

    # Filter segmented image
    pp = postprocessing.PostProcessing(pipeline=[
        'remove_small_objects',
        'remove_large_objects'
    ])
    masks = pp.run(masks, verbose=verbose)

    masks = filter_by_margin(masks, verbose=verbose)

    # Restore image to original size
    masks = restore_img(
        masks, margins,
        depth=metadata['z_size'], resolution=metadata['x_size'],
        verbose=verbose
    )

    # Move output to correct location
    img_path_out = img_path_out.replace('.tif', '.nii.gz')
    imaging.save_nii(masks, img_path_out, verbose=verbose)

    # Remove temporary files and .h5 file
    file_to_remove = img_path.replace('.nii.gz', '.h5')
    directory_to_remove = '/'.join(file_to_remove.split('/')[:-1] + ['PreProcessing'])

    subprocess.run(f'rm -r {directory_to_remove}', shell=True, check=True)
    subprocess.run(f'rm {file_to_remove}', shell=True, check=True)
