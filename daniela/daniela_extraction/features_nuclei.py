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
from radiomics import shape, firstorder, glcm, glszm
import pickle
from collections import Counter
import warnings

warnings.simplefilter("ignore", DeprecationWarning)


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
for ESPECIMEN in especimens:
    print(ESPECIMEN)
    cellpose_nu = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/nuclei/{ESPECIMEN}_MASK_EQ_XYZ_decon.nii.gz"
    gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    # pickle_div = f"/Users/dvarelat/Documents/MASTER/TFM/methods/daniela_division/LISTS_NUCLEI/{ESPECIMEN}_list_nuclei.pkl"
    nuclei = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/DECON_05/DAPI/{ESPECIMEN}_DAPI_decon_0.5.nii.gz"
    FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/NUCLEI/{ESPECIMEN}_NUCLEI_properties.csv"
    # linefile = (
    #     f"/Users/dvarelat/Documents/MASTER/TFM/DATA/LINES/line_{ESPECIMEN}.nii.gz"
    # )
    # lines = nib.load(linefile).get_fdata()
    DAPI = nib.load(nuclei).get_fdata()
    DAPI = DAPI[:, :, :, 0]
    pred_nu = nib.load(cellpose_nu).get_fdata()
    pred_mem = nib.load(gasp_mem).get_fdata()
    mask_mem = np.where(pred_mem != 0, True, False)
    mask_on_nuclei = mask_mem * pred_nu
    props_nu = ps.metrics.regionprops_3D(morphology.label(mask_on_nuclei))
    print(len(props_nu))

    df = pd.read_csv(FILE)
    print(df.shape)
    # with open(pickle_div, "rb") as f:
    #     list_div = pickle.load(f)
    # print(len(list_div))
    centroids_nu = [[round(i) for i in p["centroid"]] for p in props_nu]
    NU_original_labels_centroids = [pred_nu[c[0], c[1], c[2]] for c in centroids_nu]
    ### LINES EXTRACTION - TOUCHING
    # df = pd.DataFrame(
    #     {
    #         "cell_in_props": range(len(props_nu)),
    #         "volumes": [p.volume for p in props_nu],
    #         "original_labels": NU_original_labels_centroids,
    #         "centroids": centroids_nu,
    #     }
    # )
    # df = df[df.original_labels != 0]
    # med = np.median(df.volumes)
    # print(0.2 * med)
    # print(10 * med)
    # df = df[df.volumes > 0.2 * med]
    # df = df[df.volumes < 10 * med]
    # print(df.shape)
    # most_communs = []
    # for cell_index in list(df.cell_in_props):
    #     p = props_nu[cell_index]
    #     b = Counter(lines[p.slices].flatten())
    #     if len(list(b)) == 1:  ## si es solo 1, ese es
    #         m = list(b)[0]
    #     else:  # si hay varios. coger mayor diff cero
    #         d = {key: val for key, val in dict(b).items() if key != 0}
    #         m = max(d, key=d.get)
    #     most_communs.append(m)
    # df["lines"] = most_communs

    spacing = [
        float(c.load3D_metadata(nuclei)["x_res"]),
        float(c.load3D_metadata(nuclei)["y_res"]),
        float(c.load3D_metadata(nuclei)["z_res"]),
    ]
    results = []
    print("Running pyradiomics")
    for cell in df.cell_in_props:
        img = np.swapaxes(np.swapaxes(DAPI[props_nu[cell].slice], 0, 2), 1, 2)
        m = np.swapaxes(np.swapaxes(props_nu[cell].mask, 0, 2), 1, 2)
        sitk_img = sitk.GetImageFromArray(img)
        sitk_img = sitk.JoinSeries(sitk_img)[:, :, :, 0]
        sitk_img.SetSpacing(spacing)
        sitk_mask = sitk.GetImageFromArray(m.astype("uint16"))
        sitk_mask = sitk.JoinSeries(sitk_mask)[:, :, :, 0]
        sitk_mask.SetSpacing(spacing)
        # shapeFeatures = shape.RadiomicsShape(sitk_img, sitk_mask)
        # shapeFeatures.enableAllFeatures()
        # result = shapeFeatures.execute()
        # result["cell"] = cell
        # firstOrderFeatures = firstorder.RadiomicsFirstOrder(sitk_img, sitk_mask)
        # result_fo = firstOrderFeatures.execute()
        # result.update(result_fo)
        # result.update(
        #     {k: v.tolist() for k, v in result.items() if isinstance(v, np.ndarray)}
        # )
        # results.append(result)
        textureFeatures = glcm.RadiomicsGLCM(sitk_img, sitk_mask)
        textureFeatures.enableAllFeatures()
        result_1 = textureFeatures.execute()
        result_1["cell"] = cell
        textureFeatures2 = glszm.RadiomicsGLSZM(sitk_img, sitk_mask)
        textureFeatures2.enableAllFeatures()
        result_2 = textureFeatures2.execute()
        result_1.update(result_2)
        result_1.update(
            {k: v.tolist() for k, v in result_1.items() if isinstance(v, np.ndarray)}
        )
        results.append(result_1)

    dfnew = pd.DataFrame(results)
    df_ = df.merge(dfnew, left_on="cell_in_props", right_on="cell")
    print(df_.shape)
    # div = [1 if i in list_div else 0 for i in list(df_.cell_in_props)]
    # df_["div"] = div
    df_.to_csv(FILE, index=False, header=True)
    print(FILE)
    print("-------------")
