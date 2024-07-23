#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import numpy as np
from importlib import reload
import nibabel as nib
import pandas as pd
import importlib
from skimage import morphology


sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
import cardiac_region
import importlib

importlib.reload(cardiac_region)
import cardiac_region as c



especimens = [
    "20190504_E1",
    "20190404_E2",
    "20190520_E4",
    "20190516_E3",
    "20190806_E3",
    "20190520_E2",
    "20190401_E3",
    "20190517_E1",
    "20190520_E1",
    "20190401_E1",
]

for E in especimens:
    print(E)
    gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{E}_mGFP_XYZ_predictions_GASP.nii.gz"
    nuclei = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/DECON_05/DAPI/{E}_DAPI_decon_0.5.nii.gz"
    DAPI = nib.load(nuclei).get_fdata()
    DAPI = DAPI[:, :, :, 0]
    print(DAPI.shape)
    pred_mem = nib.load(gasp_mem).get_fdata()
    mask_mem = np.where(pred_mem != 0, True, False)
    mask_nuclei = mask_mem * DAPI
    c.saveNifti(
        mask_nuclei,
        c.load3D_metadata(nuclei),
        f"/Users/dvarelat/Documents/MASTER/TFM/methods/division/INFERENCE/{E}_predmask.nii.gz",
    )
    
    
    
        