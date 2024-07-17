import os
import glob
import auxiliary.values as v

import numpy as np
import pandas as pd

import nibabel as nib


def get_refs():
    ref_path = os.path.join(v.data_path, 'refs.csv')
    table = pd.read_csv(ref_path)

    return {c: table['ref_name'][table['cluster'] == c].values for c in table['cluster'].unique()}


def read_nii(path):
    img = nib.load(path)
    img = img.get_fdata()
    return img.astype(np.uint8)


def read_nii_batch(paths):
    X_names = sorted(glob(paths + '*.nii.gz'))
    X = list(map(nib.load, X_names))
    X = [x.get_fdata() for x in X]
    return [x.astype(np.uint8) for x in X]
