import os
import subprocess
import sys

import numpy as np
import torch
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity
from skimage.transform import rescale
from cellpose import denoise, models
# from stardist.models import StarDist3D, StarDist2D


try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

# from util.gpu.gpu_tf import increase_gpu_memory, set_gpu_allocator
from nuclei_segmentation.processing.intensity_calibration import compute_z_profile_no_mask, fit_inverted_logistic, \
    correction_factor, compute_z_profile
from util.data import imaging
from util.misc.colors import bcolors as c
# from membrane_segmentation.my_plantseg import predict


os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/"
use_gpu = torch.cuda.is_available()
print(f"GPU activated: {use_gpu}")

_skip_existing = True
_type = 'FPN enh7-3' # EBI TFP
base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CovaNacho/'
raw_dir = os.path.join(base_dir, _type)
out_dir = os.path.join(base_dir, 'Segmentation', _type)

params = {
    'model_type': 'cellpose',
    'do_3d': True,
    'diameter': 17,
    'channels': [0, 0],
    'flow_threshold': 0.4,
    'cellprob_threshold': 0.4,
    'stitch_threshold': 0.4,
}

def load_model(model_type='nuclei'):
    print(f'{c.OKBLUE}Loading model{c.ENDC}: {model_type}')
    if model_type in ['nuclei', 'cyto', 'cyto2', 'cyto3']:
        return models.Cellpose(gpu=use_gpu, model_type=model_type)

    return models.CellposeModel(model_type, diam_mean=17)

# def load_stardist_model(do_3D):
#     try:
#         increase_gpu_memory()
#         set_gpu_allocator()
#     except Exception as e:
#         print(f'{c.WARNING}GPU mem. growth not available{c.ENDC}')
#     if do_3D:
#         _model_name = 'n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)'
#         _model_path = os.path.join(current_dir, "..", "models", "stardist_models")
#         return StarDist3D(None, name=_model_name, basedir=_model_path)
#     _model_name = '2D_versatile_fluo'
#     return StarDist2D.from_pretrained(_model_name)

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


def isotropy(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

    if 'image' in kwargs:
        img = kwargs['image']

    metadata = kwargs['metadata']
    resampling_factor = metadata['z_res'] / metadata['x_res']

    img_iso = rescale(
        img, (1, 1, resampling_factor),
        order=3, mode='edge',
        clip=True,
        anti_aliasing=False,
        preserve_range=True
    )

    if 'verbose' in kwargs:
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


def norm_percentile(img, **kwargs):
    default_kwargs = {
        'low': 1,
        'high': 99
    }

    default_kwargs.update(kwargs)
    low, high = np.percentile(img, (default_kwargs['low'], default_kwargs['high']))

    print(f'\t{c.OKBLUE}Normalizing image{c.ENDC}: {low} - {high}')
    return rescale_intensity(
        img, in_range=(low, high),
        out_range=(0.0, 1.0)
    )

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

    spacing = default_kwargs['spacing']

    prev_shape = img.shape
    img = rescale(
        img, spacing,
        order=default_kwargs['order'],
        mode=default_kwargs['mode'],
        clip=default_kwargs['clip'],
        anti_aliasing=default_kwargs['anti_aliasing'],
        preserve_range=default_kwargs['preserve_range']
    )

    if 'verbose' in kwargs:
        print(f'{c.OKBLUE}Resampling image with spacing:{c.ENDC} {spacing}')
        print(f'\tOriginal shape: {prev_shape}')
        print(f'\tResampled shape: {img.shape}')

    return img

def preprocess(img_path, verbose=0):
    # pipeline = {
    #     'rescale',  # 1024 & 16bit
    #     'intensity_calibration',
    #     'cellpose_denoising',
    #     'isotropy',  # Tricubic interpolation & ¬Reconstruct
    #     'norm_percentile',  # .1
    #     'bilateral'  # 3/.1/10
    # }
    img = imaging.read_image(img_path).astype(np.float16)
    metadata, _ = imaging.load_metadata(img_path)

    img_preprocessed = resample(img, spacing=(0.5, 0.5, 1.0), verbose=verbose)
    metadata['x_res'] *= 2
    metadata['y_res'] *= 2

    print(f'{c.OKBLUE}Image shape{c.ENDC}: {img_preprocessed.shape}')
    print(f'{c.OKBLUE}Meta: {c.ENDC} {metadata}')

    img_preprocessed = intensity_calibration(img_preprocessed, verbose=verbose)
    img_preprocessed = cellpose_denoising(img_preprocessed)
    img_preprocessed = isotropy(img_preprocessed, metadata=metadata, verbose=verbose)
    img_preprocessed = norm_percentile(img_preprocessed, low=1, high=99, verbose=verbose)
    img_preprocessed = bilateral(img_preprocessed, win_size=3, sigma_color=.1, sigma_spatial=10)

    imaging.save_tiff_imagej_compatible(
        f'{img_path.replace(".tif", "_preprocessed.tif")}',
        img_preprocessed, axes='XYZ',
    )
    return img_preprocessed

def predict(img_path, out_path, verbose=0):
    img = preprocess(img_path, verbose=verbose)
    if params['model_type'] == 'cellpose':
        img = np.swapaxes(img, 0, 2)

    if params['model_type'] == 'stardist':
        # model = load_stardist_model(params['do_3d'])
        # mask, _ = model.predict_instances(
        #     img, axes='XYZ',
        #     n_tiles=(1, 1, 1),
        #     verbose=verbose
        # )
        raise NotImplementedError('Stardist model not implemented yet.')
    else:
        model = load_model(model_type='nuclei')
        mask, _, _, _ = model.eval(
            img,
            diameter=params['diameter'],
            channels=params['channels'],
            normalize=False,
            do_3D=params['do_3d'],
            cellprob_threshold=params['cellprob_threshold'],
            stitch_threshold=params['stitch_threshold'],
            flow_threshold=params['flow_threshold'],
        )

    if verbose:
        print(f'{c.OKGREEN}Masks shape{c.ENDC}: {mask.shape}')
    # Save mask
    imaging.save_tiff_imagej_compatible(
        out_path,
        mask, axes='ZYX' if params['model_type'] == 'cellpose' else 'XYZ',
    )


def process_image(raw_path, out_path):
    """Runs the whole pipeline (preprocess + predict) on one image."""
    predict(raw_path, out_path, verbose=1)
    print(f'{c.OKGREEN}Mask saved{c.ENDC}: {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-image", action="store_true", help="Process only one image, then exit.")
    parser.add_argument("raw_path", nargs="?", default=None)
    parser.add_argument("out_path", nargs="?", default=None)
    args = parser.parse_args()

    if args.one_image:
        # We are in "child" mode. Just process one image and return
        if not args.raw_path or not args.out_path:
            print("Error: Must provide raw_path and out_path with --one-image")
            sys.exit(1)
        process_image(args.raw_path, args.out_path)
        sys.exit(0)

    # Otherwise we do the multi-image loop (the "parent" mode).
    for raw_name in os.listdir(raw_dir):
        try:
            if not raw_name.endswith('.tif') or 'preprocessed' in raw_name:
                continue
            if raw_name in [
                'FPN393_1_dapi.tif',
                'FPN405_2_dapi.tif',
                'FPN408_2_dapi.tif',
                'FPN393_2_dapi.tif',
                'FPN405_5_dapi.tif',
                'FPN408_3_dapi.tif',
                'FPN393_4_dapi.tif',
                'FPN405_6_dapi.tif',
                'FPN408_4_dapi.tif',
                'FPN400_2_dapi.tif',
                'FPN406_1_dapi.tif',
                'FPN408_6_dapi.tif',
                'FPN400_3_dapi.tif',
                'FPN406_3_dapi.tif',
                'FPN418_1_dapi.tif',
                'FPN400_5_dapi.tif',
                'FPN406_4_dapi.tif',
                'FPN418_2_dapi.tif',
                'FPN400_6_dapi.tif',
                'FPN406_5_dapi.tif',
                'FPN418_3_dapi.tif',
                'FPN400_8_dapi.tif',
                'FPN408_1_dapi.tif',
                'FPN418_4_dapi.tif',
            ]:
                continue

            raw_path = os.path.join(raw_dir, raw_name)
            out_path = os.path.join(out_dir, raw_name.replace('.tif', '_mask.tif'))

            if os.path.exists(out_path) and _skip_existing:
                print(f'{c.WARNING}Skipping existing{c.ENDC}: {out_path}')
                continue

            print(f'{c.BOLD}Predicting masks{c.ENDC}: {raw_path}')

            # Spawn a subprocess (another python) to process raw_path
            cmd = [
                sys.executable,  # same python interpreter
                __file__,        # this same script
                "--one-image",
                raw_path,
                out_path
            ]
            result = subprocess.run(cmd)
            # If the child was killed by OOM or crashed, returncode != 0
            if result.returncode != 0:
                print(f'{c.FAIL}Memory or other error on {raw_name}, skipping{c.ENDC}')
                continue

        except Exception as e:
            print(f'{c.FAIL}Error processing {raw_name}: {e}{c.ENDC}')
            continue


if __name__ == '__main__':
    main()

