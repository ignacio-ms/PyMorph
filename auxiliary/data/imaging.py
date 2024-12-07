import glob
import os

import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tifffile import imread
import tifffile as tiff
import h5py
# import SimpleITK as sitk

from csbdeep.io import save_tiff_imagej_compatible

from auxiliary.utils.colors import bcolors as c
# from filtering import cardiac_region as cr


def read_nii(path, axes='XYZ', verbose=0):
    """
    Read single NIfTI file.
    :param path: Path to NIfTI file.
    :param axes: Axes of the image. (Default: XYZ)
    :param verbose: Verbosity level.
    :return: Image as numpy array.
    """
    if verbose:
        print(f'{c.OKBLUE}Reading NIfTI{c.ENDC}: {path}')

    try:
        img = nib.load(path).get_fdata()
        if axes == 'ZXY':
            img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif axes == 'ZYX':
            img = np.swapaxes(img, 0, 2)
        elif axes == 'XZY':
            img = np.swapaxes(img, 0, 1)
        elif axes not in ['XYZ', 'ZXY', 'ZYX']:
            print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - NIfTI')

    except FileNotFoundError:
        print(f'File not found: {path}')
        return None

    if img.ndim == 4:
        img = img[:, :, :, 0]

    return img


def read_nii_batch(paths):
    """
    Read all NIfTI files in the given path.
    :param paths: Path to NIfTI files.
    :return: List of images as numpy arrays.
    """
    X_names = sorted(glob(paths + '*.nii.gz'))
    X = list(map(nib.load, X_names))
    X = [x.get_fdata() for x in X]
    return X


def load_metadata(path):
    """
    Load metadata from (NIfTI | Tif) file without loading the image.
    :param path: Path to image.
    :return: Metadata.
    """
    if path.endswith('.nii.gz'):
        proxy = nib.load(path)
        return {
            'x_size': proxy.header['dim'][1],
            'y_size': proxy.header['dim'][2],
            'z_size': proxy.header['dim'][3],
            'x_res': np.round(proxy.header['pixdim'][1], 6),
            'y_res': np.round(proxy.header['pixdim'][2], 6),
            'z_res': np.round(proxy.header['pixdim'][3], 6)
        }, proxy.affine.copy()
    else:  # Tiff file
        with tiff.TiffFile(path) as tif:
            first_page = tif.pages[0]
            y_size, x_size = first_page.shape
            z_size = len(tif.pages)

            resolution = first_page.tags.get('XResolution'), first_page.tags.get('YResolution')
            if resolution[0] is not None and resolution[1] is not None:
                x_res = 1.0 / resolution[0].value[0] if resolution[0].value[0] != 0 else None
                y_res = 1.0 / resolution[1].value[0] if resolution[1].value[0] != 0 else None
                z_res = 1.0
            else:
                x_res, y_res, z_res = 1.0, 1.0, 1.0

        return {
            'x_size': x_size,
            'y_size': y_size,
            'z_size': z_size,
            'x_res': x_res,
            'y_res': y_res,
            'z_res': z_res
        }, None


# def resample_img(img, mask, raw_img_path):
#     """
#     Resample the image and mask. The resampling is done based on the XYZ resolutions.
#     :param img: Image.
#     :param mask: Mask.
#     :param raw_img_path: Raw image path.
#     :param verbose: Verbosity level. (default: 0)
#     :return: Resampled image and mask.
#     """
#     metadata, _ = cr.load_metadata(raw_img_path)
#     spacing = [
#         float(metadata['x_res']),
#         float(metadata['y_res']),
#         float(metadata['z_res'])
#     ]
#
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     resampler.SetSize([
#         int(round(img.GetSize()[0] * spacing[0])),
#         int(round(img.GetSize()[1] * spacing[1])),
#         int(round(img.GetSize()[2] * spacing[2]))
#     ])
#
#     img = resampler.Execute(img)
#     mask = resampler.Execute(mask)
#
#     return img, mask


def read_tiff(path, axes='XYZ', verbose=0):
    """
    Read single TIFF file.
    :param path: Path to TIFF file.
    :param axes: Axes of the image. (Default: XYZ)
    :param verbose: Verbosity level.
    :return: Image as numpy array.
    """
    if verbose:
        print(f'{c.OKBLUE}Reading TIFF{c.ENDC}: {path}')

    img = np.array(imread(path))
    if img.ndim == 2:
        return img

    if axes == 'XYZ':
        img = np.swapaxes(img, 0, 2)
    elif axes == 'ZXY':
        img = np.swapaxes(img, 1, 2)
    elif axes not in ['XYZ', 'ZXY', 'ZYX']:
        print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - TIFF')

    if img.ndim == 4:
        img = img[:, :, :, 0]

    return img


def read_image(path, axes='XYZ', verbose=0):
    """
    Read single image.
    :param path: Path to image.
    :param axes: Axes of the image. (Default: XYZ)
    :param verbose: Verbosity level.
    :return: Image as numpy array.
    """
    if path.endswith('.nii.gz'):
        return read_nii(path, axes, verbose)
    elif path.endswith('.tif') or path.endswith('.tiff'):
        return read_tiff(path, axes, verbose)
    else:
        print(f'{c.FAIL}Invalid image format{c.ENDC}: {path}')


def save_prediction(labels, out_path, axes='XYZ', verbose=0):
    """
    Save prediction as tiff image.
    :param axes: Axes of the image. If the image is 2D, the axis must be XY or YX (Default: XYZ)
    :param labels: Labels.
    :param out_path: Path to save prediction.
    :param verbose: Verbosity level.
    """

    if out_path.endswith('.nii.gz'):
        out_path = out_path.replace('.nii.gz', '.tif')

    # Check axes (test)
    if labels.shape[0] < labels.shape[1]:
        labels = np.swapaxes(labels, 0, 2)

    if labels.ndim == 2:
        labels = np.expand_dims(labels, axis=2)

    save_tiff_imagej_compatible(out_path, labels, axes=axes)

    if verbose:
        print(f'\n{c.OKGREEN}Saving prediction{c.ENDC}: {out_path}')


def save_nii(labels, out_path, axes='XYZ', affine=None, metadata=None, verbose=0):
    """
    Save prediction as NIfTI image.
    :param labels: Labels.
    :param out_path: Path to save prediction.
    :param axes: Axes of the image. (Default: XYZ)
    :param affine: Affine transformation matrix. (Default: None)
    :param verbose: Verbosity level.
    """
    if out_path.endswith('.tif') or out_path.endswith('.tiff'):
        out_path = out_path.replace('.tif', '.nii.gz')

    if affine is None:
        affine = np.eye(4)

    if axes == 'ZXY':
        labels = np.swapaxes(np.swapaxes(labels, 0, 2), 1, 2)
    elif axes == 'ZYX':
        labels = np.swapaxes(labels, 0, 2)
    elif axes not in ['XYZ', 'ZXY', 'ZYX']:
        print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - NIfTI')

    img = nib.Nifti1Image(labels, affine)

    if metadata:
        header = img.header
        header['pixdim'][1] = metadata.get('x_res', header['pixdim'][1])
        header['pixdim'][2] = metadata.get('y_res', header['pixdim'][2])
        header['pixdim'][3] = metadata.get('z_res', header['pixdim'][3])

        img = nib.Nifti1Image(labels, affine, header=header)

    nib.save(img, out_path)

    if verbose:
        print(f'\n{c.OKGREEN}Saving prediction{c.ENDC}: {out_path}')


def nii2h5(img, out_path, axes='XYZ', verbose=0):
    """
    Save image as HDF5 file.
    :param img: Image.
    :param out_path: Path to save image.
    :param axes: Axes of the image. (Default: XYZ)
    :param verbose: Verbosity level.
    """
    if out_path.endswith('.nii.gz'):
        out_path = out_path.replace('.nii.gz', '.h5')

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping conversion.")
        return

    if axes == 'ZXY':
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    elif axes == 'ZYX':
        img = np.swapaxes(img, 0, 2)
    elif axes not in ['XYZ', 'ZXY', 'ZYX']:
        print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - HDF5')

    with h5py.File(out_path, 'a') as hf:
        _ = hf.create_dataset('raw', data=img)

    if verbose:
        print(f'\n{c.OKGREEN}Saving image{c.ENDC}: {out_path}')


def crop(img, x1, x2, y1, y2, z1, z2):
    """
    Crop image.
    :param img: Image.
    :param x1: Start of x-axis.
    :param x2: End of x-axis.
    :param y1: Start of y-axis.
    :param y2: End of y-axis.
    :param z1: Start of z-axis.
    :param z2: End of z-axis.
    :return: Cropped image.
    """
    return img[x1:x2, y1:y2, z1:z2]


def iqr_filter(img, get_params=False, verbose=0):
    """
    Apply IQR filter to image intensities to remove z-slices with low intensity values.
    :param img: Image.
    :param get_params: Get intensities and threshold. (Default: False)
    :param verbose: Verbosity level.
    :return: Filtered image.
    """
    img_copy = img.copy().astype(np.uint8)
    intensities = np.mean(img_copy, axis=(0, 1))

    q1 = np.percentile(intensities, 35)
    q3 = np.percentile(intensities, 95)
    iqr = q3 - q1

    threshold = q1 - .1 * iqr
    if verbose:
        print(f'{c.BOLD}IQR Threshold{c.ENDC}: {threshold}')
        print(f'Removing {np.sum(intensities < threshold)} z-slices')

    if get_params:
        return img[:, :, intensities > threshold], intensities, threshold

    return img[:, :, intensities > threshold]


def update_affine(affine, original_res, new_res):
    """
    Update the affine matrix based on the new resolutions.
    :param affine: Original affine matrix.
    :param original_res: Tuple of original (x_res, y_res, z_res).
    :param new_res: Tuple of new (x_res, y_res, z_res).
    :return: Updated affine matrix.
    """
    if affine is None:
        affine = np.eye(4)

    scaling_factors = np.array(new_res) / np.array(original_res)
    updated_affine = affine.copy()
    updated_affine[:3, :3] *= scaling_factors
    return updated_affine


def resize_xyz_05(img_path):
    """
    Resize x and y dimensions by 0.5.
    Metadata res x and y should be updated. (res_x / 2)
    :param img_path: Path to image.
    :return: Resized image.
    """
    img = read_image(img_path)
    metadata, affine = load_metadata(img_path)

    original_x, original_y, original_z = img.shape
    new_x, new_y, new_z = original_x // 2, original_y // 2, original_z // 2

    zoom_factors = (0.5, 0.5, 0.5)
    resized_img = zoom(img, zoom_factors, order=1)

    # Update metadata
    metadata['x_size'], metadata['y_size'], metadata['z_size'] = new_x, new_y, new_z
    metadata['x_res'] /= 2
    metadata['y_res'] /= 2
    metadata['z_res'] /= 2

    affine = update_affine(
        affine,
        (metadata['x_res'] * 2, metadata['y_res'] * 2, metadata['z_res'] * 2),
        (metadata['x_res'], metadata['y_res'], metadata['z_res'])
    )

    img_path = img_path.replace('.tif', '_0.5.nii.gz')
    save_nii(
        resized_img.astype(np.uint8), img_path,
        affine=affine, metadata=metadata, verbose=1
    )

    print(f'{c.OKGREEN}Resized image{c.ENDC}: {img_path}')
    print(f'{c.BOLD}Metadata{c.ENDC}: {metadata}')
    print(f'{c.BOLD}Affine{c.ENDC}: {affine}')
    print(f'{c.OKBLUE}Saving resized image{c.ENDC}: {img_path}')
