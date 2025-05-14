import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from skimage.restoration import denoise_bilateral
from skimage.morphology import white_tophat, disk, opening, closing
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from cellpose import denoise
from csbdeep.utils import normalize as deep_norm

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from feature_extraction.feature_extractor import filter_by_margin
from nuclei_segmentation.processing.intensity_calibration import compute_z_profile_no_mask, fit_inverted_logistic, \
    correction_factor, compute_z_profile
from filtering.cardiac_region import get_margins, crop_img, restore_img
from nuclei_segmentation.processing import postprocessing
from util.data import imaging
from util.misc.colors import bcolors as c
from membrane_segmentation.my_plantseg import predict


use_gpu = torch.cuda.is_available()
print(f"GPU activated: {use_gpu}")

base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/2025/'
raw_dir = os.path.join(base_dir, 'raw_to_segment_cells', 'decon')
tissue_seg_dir = os.path.join(base_dir, 'binary_masks')
results_dir = os.path.join(base_dir, 'results')

skip_existing = True

params = {
    'size_threshold': 55000,
}

pipeline = [
    'intensity_calibration',
    # 'isotropy',
    'cellpose_denoising',
    'bilateral'
]

def columnarity():
    pass

def volume(mask):
    vol_dict = {}
    cell_ids = np.unique(mask)

    for cell in cell_ids:
        if cell == 0:
            continue
        vol_dict[cell] = (mask == cell).sum()
    return vol_dict

def vox2micron(metadata):
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    return x_res * y_res * z_res

def annotate_segmentation():
    pass

def modify_yaml_path(new_path, file_path='membrane_segmentation/config.yaml'):
    subprocess.run(
        f"sed -i 's|^path:.*|path: {new_path}|' {file_path}",
        shell=True, check=True
    )

def intensity_calibration(img, **kwargs):
    default_kwargs = {
        'mask': None,
        'p0': None,
        'maxfev': 50000,
        'z_ref': 0,
    }
    default_kwargs.update(kwargs)

    z_slices, z_indices = img.shape[2], np.arange(img.shape[2])

    # Compute the intensity profile along the z-axis
    zprofile = compute_z_profile_no_mask if default_kwargs['mask'] is None else compute_z_profile
    fg_means, bg_means = zprofile(img)

    # Fit inverted logistic function
    popt = fit_inverted_logistic(
        z_indices, fg_means,
        p0=default_kwargs['p0'], maxfev=default_kwargs['maxfev']
    )
    L, U, k, x0 = popt

    if 'verbose' in kwargs:
        print(f'{c.OKBLUE}Intensity calibration fitted parameters{c.ENDC}:')
        print(f'\tL: {L}\n\tU: {U}\n\tk: {k}\n\tx0: {x0}')

    # Calibrate image
    calibration_factors = [
        correction_factor(z, popt, z_ref=default_kwargs['z_ref'])
        for z in z_indices
    ]

    img_calibrated = np.zeros_like(img, dtype=img.dtype)
    for z, factor in zip(range(z_slices), calibration_factors):
        img_calibrated[..., z] = img[..., z] * factor

    # Transpose back to ZYX
    return img_calibrated

def cellpose_denoising(img, **kwargs):
    default_kwargs = {
        'diameter': 30,
        'channels': [0, 0],
        'model_type': 'denoise_cyto3',
    }
    default_kwargs.update(kwargs)

    model = denoise.DenoiseModel(model_type='denoise_cyto3', gpu=use_gpu)
    denoised = np.swapaxes(np.swapaxes([
        model.eval(img[..., z], **kwargs)
        for z in range(img.shape[-1])
    ], 0, 1), 1, 2)

    if denoised.ndim == 4:
        denoised = denoised[..., 0]

    return denoised

def bilateral(img, **kwargs):
    default_kwargs = {
        'win_size': 3,
        'sigma_color': None,
        'sigma_spatial': 7
    }
    default_kwargs.update(kwargs)

    return np.swapaxes(np.swapaxes([
        denoise_bilateral(img[..., z], **default_kwargs)
        for z in range(img.shape[-1])
    ], 0, 1), 1, 2)

def tophat(img, **kwargs):
    default_kwargs = {
        'disk_size': 7
    }

    default_kwargs.update(kwargs)

    img_tophat = np.swapaxes(np.swapaxes([
        white_tophat(
            img[..., z],
            footprint=disk(default_kwargs['disk_size'])
        )
        for z in range(img.shape[-1])
    ], 0, 1), 1, 2)

    thr = threshold_otsu(img_tophat)

    init_mask = img_tophat > thr
    mask_closed = np.swapaxes(np.swapaxes([
        closing(init_mask[..., z], disk(5))
        for z in range(init_mask.shape[-1])
    ], 0, 1), 1, 2)

    return img * mask_closed

def resample(img, **kwargs):
    default_kwargs = {
        'spacing': (.5, .5, 1.0),
        'order': 3,
        'mode': 'edge',
        'clip': True,
        'anti_aliasing': False,
        'preserve_range': True,
    }
    default_kwargs.update(kwargs)

    img_resampled = rescale(
        img, default_kwargs['spacing'],
        order=default_kwargs['order'],
        mode=default_kwargs['mode'],
        clip=default_kwargs['clip'],
        anti_aliasing=default_kwargs['anti_aliasing'],
        preserve_range=default_kwargs['preserve_range']
    )

    return img_resampled

def preprocess(img_path, verbose=0):
    img = imaging.read_image(img_path)

    if img.shape[0] == 2048:
        img = resample(img)
        print(f'{c.OKBLUE}Resampled image{c.ENDC}: {img.shape}')

    img_denoised = intensity_calibration(img)
    # img_denoised = tophat(img)
    # img_denoised = cellpose_denoising(img)
    img_denoised = bilateral(img_denoised)
    img_denoised = deep_norm(img_denoised, 1, 99, axis=(0, 1, 2))

    imaging.save_tiff_imagej_compatible(
        f'{img_path.replace(".tif", "_preprocessed.tif")}',
        img_denoised, axes='XYZ',
    )
    return img_denoised

def filter_by_tissue(img, lines, erode=0, erode_size=3, verbose=0):
    if img.ndim == 4:
        img = img[..., 0]
    filtered = np.zeros_like(img)

    print(f'lines shape: {lines.shape}')
    print(f'img shape: {img.shape}')

    if erode and erode_size:
        if verbose:
            print(f'{c.BOLD}Dilating mask{c.ENDC}: ({erode_size}x{erode_size}) {erode} times...')

        ds = erode_size if erode_size % 2 else erode_size + 1
        kernel = np.ones((ds, ds), np.uint8)

        if lines.ndim == 2:
            lines = cv2.erode(lines, kernel, iterations=erode)
        else:
            for z in range(lines.shape[-1]):
                lines[..., z] = cv2.erode(lines[..., z], kernel, iterations=erode)

    cell_ids = np.unique(img[lines > 0])

    for z in range(lines.shape[-1]):
        mask = np.isin(img[..., z], cell_ids)
        filtered[..., z] = np.where(mask, img[..., z], 0)

    return filtered

def predict(img_path, out_path, lines_path, verbose=0):
    img = preprocess(img_path, verbose=verbose)

    print(img.shape)

    # Crop image by tissue margin
    metadata, _ = imaging.load_metadata(img_path)

    margins = get_margins(
        line_path=lines_path, img_path=img_path,
        tissue=None, verbose=verbose
    )
    print(img.shape)
    img = crop_img(img, margins, verbose=verbose)
    print(img.shape)

    # Pipeline
    if verbose:
        print(f'{c.OKBLUE}Transforming image {c.BOLD} .tif -> .h5{c.ENDC}: {img_path}')
    imaging.nii2h5(img, img_path.replace('.tiff', '.h5'), verbose=verbose)

    modify_yaml_path(img_path.replace('.tiff', '.h5'))
    print(img_path.replace('.tiff', '.h5'))
    subprocess.run(f'plantseg --config membrane_segmentation/config.yaml', shell=True, check=True)

    # Load segmented image and correct axes
    path = '/'.join(img_path.split('/')[:-1] + [
        'PreProcessing', 'confocal_3D_unet_sa_meristem_cells',
        'GASP', 'PostProcessing',
        img_path.split('/')[-1].replace('.tiff', '_predictions_gasp_average.tiff')
    ])

    masks = imaging.read_image(
        path,
        axes='XYZ', verbose=verbose
    )

    # Filter segmented image
    pp = postprocessing.PostProcessing(pipeline=[
        'remove_large_objects'
    ])
    masks = pp.run(masks, verbose=verbose, percentile=98)

    masks = filter_by_margin(masks, verbose=verbose)

    # Reorder margins to XYZ
    margins = (
        (margins[0][2], margins[0][1], margins[0][0]),
        (margins[1][2], margins[1][1], margins[1][0])
    )
    masks = restore_img(
        masks, margins,
        depth=metadata['z_size'], resolution=metadata['x_size'],
        verbose=verbose, axes='ZYX'
    )


    lines = imaging.read_image(lines_path, axes='ZYX', verbose=verbose)
    masks = filter_by_tissue(masks, lines, erode=3, erode_size=5, verbose=verbose)

    # Move output to correct location
    imaging.save_prediction(masks, out_path, verbose=verbose)

    # Remove temporary files and .h5 file
    file_to_remove = img_path.replace('.tiff', '.h5')
    directory_to_remove = '/'.join(file_to_remove.split('/')[:-1] + ['PreProcessing'])

    # subprocess.run(f'rm -r {directory_to_remove}', shell=True, check=True)
    # subprocess.run(f'rm {file_to_remove}', shell=True, check=True)

def process_image(raw_path, out_path, lines_path):
    """
    Run preprocess + predict on one file and save the mask.
    Intended to be called in a short-lived child process so that
    GPU / RAM is released when the process ends.
    """
    predict(raw_path, out_path, lines_path, verbose=1)
    print(f'{c.OKGREEN}Mask saved{c.ENDC}: {out_path}')

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--one-image", action="store_true",
                        help="Internal flag: process a single image then exit.")
    parser.add_argument("raw_path",    nargs="?", default=None)
    parser.add_argument("out_path",    nargs="?", default=None)
    parser.add_argument("lines_path",  nargs="?", default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # CHILD MODE – run exactly one image then exit (return-code==0 on success)
    # ------------------------------------------------------------------
    if args.one_image:
        if not (args.raw_path and args.out_path and args.lines_path):
            print("Error: raw_path, out_path and lines_path are required with --one-image")
            sys.exit(1)
        try:
            process_image(args.raw_path, args.out_path, args.lines_path)
        except Exception as e:
            # Let the parent know the child failed (return-code ≠ 0)
            print(f'{c.FAIL}Child process failed on {args.raw_path}: {e}{c.ENDC}')
            sys.exit(2)

    # ------------------------------------------------------------------
    # PARENT MODE – loop over many images, spawning a child each time
    # ------------------------------------------------------------------
    for raw_name in os.listdir(raw_dir):
        try:
            if (not raw_name.endswith('.tiff')) or ('preprocessed' in raw_name):
                continue

            if raw_name in [
                'E5_decon.tiff',
                'CE3_decon.tiff',
            ]:
                continue

            raw_path   = os.path.join(raw_dir,  raw_name)
            out_path   = os.path.join(results_dir,
                                      raw_name.replace('.tiff', '_mask.tif'))
            lines_path = os.path.join(tissue_seg_dir,
                                      raw_name.replace('_decon.tiff',
                                                       '_SHF_segmentation.tif'))

            if skip_existing and os.path.exists(out_path):
                print(f'{c.OKGREEN}Skipping existing mask{c.ENDC}: {out_path}')
                continue

            print(f'{c.BOLD}Predicting masks{c.ENDC}: {raw_path}')

            # >>> spawn a child python so a failure/OOM won't kill the loop
            cmd = [
                sys.executable,   # current Python interpreter
                __file__,         # this very script
                "--one-image",
                raw_path,
                out_path,
                lines_path
            ]
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f'{c.FAIL}Skipping {raw_name} (child returned {result.returncode}){c.ENDC}')
                continue  # proceed with the next image

        except Exception as e:
            print(f'{c.FAIL}Error in parent loop for {raw_name}: {e}{c.ENDC}')
            continue


if __name__ == '__main__':
    main()

