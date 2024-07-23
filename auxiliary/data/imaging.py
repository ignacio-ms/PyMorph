import glob

import numpy as np
import nibabel as nib
from tifffile import imread
import tifffile as tiff

from csbdeep.io import save_tiff_imagej_compatible

from auxiliary.utils.colors import bcolors as c


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
        elif axes not in ['XYZ', 'ZXY', 'ZYX']:
            print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - NIfTI')
    except FileNotFoundError:
        print(f'File not found: {path}')
        return None
    return img.astype(np.uint8)


def read_nii_batch(paths):
    """
    Read all NIfTI files in the given path.
    :param paths: Path to NIfTI files.
    :return: List of images as numpy arrays.
    """
    X_names = sorted(glob(paths + '*.nii.gz'))
    X = list(map(nib.load, X_names))
    X = [x.get_fdata() for x in X]
    return [x.astype(np.uint8) for x in X]


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
        }
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
                x_res, y_res, z_res = None, None, None

        return {
            'x_size': x_size,
            'y_size': y_size,
            'z_size': z_size,
            'x_res': x_res,
            'y_res': y_res,
            'z_res': z_res
        }


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
    if axes == 'XYZ':
        img = np.swapaxes(img, 0, 2)
    elif axes == 'ZXY':
        img = np.swapaxes(img, 1, 2)
    elif axes not in ['XYZ', 'ZXY', 'ZYX']:
        print(f'{c.FAIL}Invalid axes{c.ENDC}: {axes} (XYZ, ZXY, ZYX) - TIFF')

    return img


def save_prediction(labels, out_path, verbose=0):
    """
    Save prediction as tiff image.
    :param labels: Labels.
    :param out_path: Path to save prediction.
    :param verbose: Verbosity level.
    """

    if out_path.endswith('.nii.gz'):
        out_path = out_path.replace('.nii.gz', '.tif')

    # Check axes (test)
    if labels.shape[0] < labels.shape[1]:
        labels = np.swapaxes(labels, 0, 2)

    save_tiff_imagej_compatible(out_path, labels, axes='XYZ')

    if verbose:
        print(f'{c.OKGREEN}Saving prediction{c.ENDC}: {out_path}')
