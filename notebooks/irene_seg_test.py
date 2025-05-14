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

base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/Ignacio/LabMT/IreneLegAngles/'
raw_dir = os.path.join(base_dir, 'dapi')
results_dir = os.path.join(base_dir, 'segmentation')

skip_existing = True

params = {
    'size_threshold': 55000,
}

# pipeline = [
#     'intensity_calibration',
#     # 'isotropy',
#     'cellpose_denoising',
#     'bilateral'
# ]
#
# def columnarity():
#     pass

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
        'sigma_color': 50,
        # 'sigma_spatial': 7
    }
    default_kwargs.update(kwargs)

    return np.swapaxes(np.swapaxes([
        denoise_bilateral(img[..., z], **default_kwargs)
        for z in range(img.shape[-1])
    ], 0, 1), 1, 2)

def tophat(img, **kwargs):
    default_kwargs = {
        'disk_size': 3
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
        closing(init_mask[..., z], disk(1))
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


def isotropy(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

    if 'image' in kwargs:
        img = kwargs['image']

    metadata = kwargs['metadata']
    resampling_factor = metadata['z_res'] / metadata['x_res']

    img_iso = rescale(
        img, (1, 1, resampling_factor),
        order=0, mode='constant',
        anti_aliasing=False,
        preserve_range=True
    )

    print(
        f'\t{c.OKBLUE}Image resolution{c.ENDC}: \n'
        f'\tX: {metadata["x_res"]} um/px\n'
        f'\tY: {metadata["y_res"]} um/px\n'
        f'\tZ: {metadata["z_res"]} um/px'
    )
    print(f'\tResampling factor: {resampling_factor}')
    print(f'\tOriginal shape: {img.shape}')
    print(f'\tResampled shape: {img_iso.shape}')

    return img_iso


def preprocess(img_path, metadata, verbose=0):
    img = imaging.read_image(img_path)

    if img.shape[0] == 2048:
        img = resample(img)
        metadata['x_size'] *= 2
        metadata['y_size'] *= 2
        print(f'{c.OKBLUE}Resampled image{c.ENDC}: {img.shape}')

    # img = intensity_calibration(img)
    img = isotropy(img, metadata=metadata)
    img = tophat(img)
    # img = cellpose_denoising(img)
    img = bilateral(img)
    # img = deep_norm(img, 1, 99, axis=(0, 1, 2))

    imaging.save_tiff_imagej_compatible(
        f'{img_path.replace(".tif", "_preprocessed.tif")}',
        img, axes='XYZ',
    )
    return img

def predict(img_path, out_path, verbose=0): # , lines_path
    metadata, _ = imaging.load_metadata(img_path, z_res=1.0399826)
    img = preprocess(img_path, metadata=metadata, verbose=verbose)
    print(img.shape)

    # Swap from XYZ to YZX
    # img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    # print(img.shape)

    if verbose:
        print(f'{c.OKBLUE}Transforming image {c.BOLD} .tif -> .h5{c.ENDC}: {img_path}')
    imaging.nii2h5(img, img_path.replace('.tif', '.h5'), verbose=verbose)

    modify_yaml_path(img_path.replace('.tif', '.h5'))
    print(img_path.replace('.tif', '.h5'))
    subprocess.run(f'plantseg --config membrane_segmentation/config.yaml', shell=True, check=True)

    # Load segmented image and correct axes
    path = '/'.join(img_path.split('/')[:-1] + [
        'PreProcessing', 'confocal_3D_unet_sa_meristem_cells',
        'GASP', 'PostProcessing',
        img_path.split('/')[-1].replace('.tif', '_predictions_gasp_average.tiff')
    ])

    masks = imaging.read_image(
        path,
        axes='ZYX', verbose=verbose
    )

    # Filter segmented image
    pp = postprocessing.PostProcessing(pipeline=[
        'remove_large_objects'
    ])
    masks = pp.run(masks, verbose=verbose, percentile=99)

    # Move output to correct location
    imaging.save_nii(masks, out_path, verbose=verbose)

    # Remove temporary files and .h5 file
    file_to_remove = img_path.replace('.tif', '.h5')
    directory_to_remove = '/'.join(file_to_remove.split('/')[:-1] + ['PreProcessing'])

    # subprocess.run(f'rm -r {directory_to_remove}', shell=True, check=True)
    # subprocess.run(f'rm {file_to_remove}', shell=True, check=True)

def process_image(raw_path, out_path): #, lines_path):
    """
    Run preprocess + predict on one file and save the mask.
    Intended to be called in a short-lived child process so that
    GPU / RAM is released when the process ends.
    """
    predict(raw_path, out_path, verbose=1) # , lines_path
    print(f'{c.OKGREEN}Mask saved{c.ENDC}: {out_path}')

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--one-image", action="store_true",
                        help="Internal flag: process a single image then exit.")
    parser.add_argument("raw_path",    nargs="?", default=None)
    parser.add_argument("out_path",    nargs="?", default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # CHILD MODE
    # ------------------------------------------------------------------
    if args.one_image:
        if not (args.raw_path and args.out_path): # and args.lines_path):
            print("Error: raw_path, out_path and lines_path are required with --one-image")
            sys.exit(1)
        try:
            process_image(args.raw_path, args.out_path) #, args.lines_path)
        except Exception as e:
            # Let the parent know the child failed (return-code ≠ 0)
            print(f'{c.FAIL}Child process failed on {args.raw_path}: {e}{c.ENDC}')
            sys.exit(2)

    # ------------------------------------------------------------------
    # PARENT MODE
    # ------------------------------------------------------------------
    for raw_name in os.listdir(raw_dir):
        try:
            if (not raw_name.endswith('.tif')) or ('preprocessed' in raw_name):
                continue

            if '2024' in raw_name:
                print(f'{c.OKBLUE}Skipping{c.ENDC}: {raw_name}')
                continue

            raw_path   = os.path.join(raw_dir,  raw_name)
            out_path   = os.path.join(results_dir, raw_name.replace('.tif', '_mask.nii.gz'))

            if skip_existing and os.path.exists(out_path):
                print(f'{c.OKGREEN}Skipping existing mask{c.ENDC}: {out_path}')
                continue

            print(f'{c.BOLD}Predicting masks{c.ENDC}: {raw_path}')

            cmd = [
                sys.executable,   # current Python interpreter
                __file__,         # this very script
                "--one-image",
                raw_path,
                out_path,
            ]
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f'{c.FAIL}Skipping {raw_name} (child returned {result.returncode}){c.ENDC}')
                continue

        except Exception as e:
            print(f'{c.FAIL}Error in parent loop for {raw_name}: {e}{c.ENDC}')
            continue


if __name__ == '__main__':
    main()

