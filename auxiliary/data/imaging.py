import glob

import numpy as np
import nibabel as nib

from csbdeep.io import save_tiff_imagej_compatible

from auxiliary.utils.colors import bcolors as c


def read_nii(path):
    """
    Read single NIfTI file.
    :param path: Path to NIfTI file.
    :return: Image as numpy array.
    """
    try:
        img = nib.load(path)
        img = img.get_fdata()
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
    Load metadata from NIfTI file without loading the image.
    :param path: Path to NIfTI file.
    :return: Metadata.
    """
    proxy = nib.load(path)
    return {
        'x_size': proxy.header['dim'][1],
        'y_size': proxy.header['dim'][2],
        'z_size': proxy.header['dim'][3],
        'x_res': np.round(proxy.header['pixdim'][1], 6),
        'y_res': np.round(proxy.header['pixdim'][2], 6),
        'z_res': np.round(proxy.header['pixdim'][3], 6)
    }


def save_prediction(labels, out_path, verbose=0):
    """
    Save prediction as tiff image.
    :param labels: Labels.
    :param out_path: Path to save prediction.
    :param verbose: Verbosity level.
    """

    save_tiff_imagej_compatible(out_path, labels, axes='XYZ')

    if verbose:
        print(f'{c.OKGREEN}Saving prediction{c.ENDC}: {out_path}')
