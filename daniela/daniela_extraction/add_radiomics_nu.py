#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import sys
import os
import radiomics
import os
import SimpleITK as sitk
import nibabel as nib
import porespy as ps
import pandas as pd
import numpy as np
from skimage import morphology
from radiomics import shape, firstorder

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

FOLDERS = [
    "1_20190504_E1",
    "7_20190404_E2",
    "10_20190520_E4",
    "3_20190516_E3",
    "4_20190806_E3",
    "2_20190520_E2",
    "5_20190401_E3",
    "8_20190517_E1",
    "9_20190520_E1",
    "6_20190401_E1",
]

keys = [
    "Elongation",
    "Flatness",
    "LeastAxisLength",
    "MajorAxisLength",
    "Maximum2DDiameterColumn",
    "Maximum2DDiameterRow",
    "Maximum2DDiameterSlice",
    "Maximum3DDiameter",
    "MeshVolume",
    "MinorAxisLength",
    "Sphericity",
    "SurfaceArea",
    "SurfaceVolumeRatio",
    "VoxelVolume",
    "cell",
    "10Percentile",
    "90Percentile",
    "Energy",
    "Entropy",
    "InterquartileRange",
    "Kurtosis",
    "Maximum",
    "MeanAbsoluteDeviation",
    "Mean",
    "Median",
    "Minimum",
    "Range",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "TotalEnergy",
    "Uniformity",
    "Variance",
]

d_empty = dict(zip(keys, [-1 for k in range(len(keys))]))
for i, ESPECIMEN in enumerate(especimens):
    print(ESPECIMEN)
    FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{FOLDERS[i]}/cell_properties_radiomics.csv"
    gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    cellpose_nu = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/nuclei/{ESPECIMEN}_MASK_EQ_XYZ_decon.nii.gz"
    nuclei = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/DECON_05/DAPI/{ESPECIMEN}_DAPI_decon_0.5.nii.gz"
    pred_mem = nib.load(gasp_mem).get_fdata()
    mask_mem = np.where(pred_mem != 0, True, False)
    pred_nu = nib.load(cellpose_nu).get_fdata()
    mask_on_nuclei = mask_mem * pred_nu
    props_nu = ps.metrics.regionprops_3D(morphology.label(mask_on_nuclei))
    DAPI = nib.load(nuclei).get_fdata()
    DAPI = DAPI[:, :, :, 0]
    spacing = [
        float(c.load3D_metadata(nuclei)["x_res"]),
        float(c.load3D_metadata(nuclei)["y_res"]),
        float(c.load3D_metadata(nuclei)["z_res"]),
    ]
    results = []
    df = pd.read_csv(FILE)
    print(df.shape)
    print(df[df.nuclei_cell_in_props != -1].shape)
    for cell in df.nuclei_cell_in_props:
        if cell == -1:
            result = d_empty
        else:
            img = np.swapaxes(np.swapaxes(DAPI[props_nu[cell].slice], 0, 2), 1, 2)
            m = np.swapaxes(np.swapaxes(props_nu[cell].mask, 0, 2), 1, 2)
            sitk_img = sitk.GetImageFromArray(img)
            sitk_img = sitk.JoinSeries(sitk_img)[:, :, :, 0]
            sitk_img.SetSpacing(spacing)
            sitk_mask = sitk.GetImageFromArray(m.astype("uint16"))
            sitk_mask = sitk.JoinSeries(sitk_mask)[:, :, :, 0]
            sitk_mask.SetSpacing(spacing)
            shapeFeatures = shape.RadiomicsShape(sitk_img, sitk_mask)
            shapeFeatures.enableAllFeatures()
            result = shapeFeatures.execute()
            result["cell"] = str(cell)
            firstOrderFeatures = firstorder.RadiomicsFirstOrder(sitk_img, sitk_mask)
            result_fo = firstOrderFeatures.execute()
            result.update(result_fo)
            result.update(
                {k: v.tolist() for k, v in result.items() if isinstance(v, np.ndarray)}
            )
        results.append(result)

    dfnew = pd.DataFrame(results).rename(columns=lambda x: "NU" + x, inplace=False)
    df_ = pd.concat([df, dfnew], axis=1)
    print(df_.shape)
    df_.to_csv(FILE, index=False, header=True)
    print("---------------------")
