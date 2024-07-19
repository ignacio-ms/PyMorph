import os
import glob
import auxiliary.values as v

import numpy as np
import pandas as pd

import nibabel as nib


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
