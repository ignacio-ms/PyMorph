import os
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import importlib
import json


# f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)


# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")

sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region
import importlib

importlib.reload(cardiac_region)
import cardiac_region as c

FOLDERS = [
    element
    for sublist in [
        [f"{i[-1]}_2019" + e for e in data[i]] for i in ["stage2", "stage3", "stage4"]
    ]
    for element in sublist
]
for i, folder in enumerate(FOLDERS):
    print(folder)
    ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
    gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    df = pd.read_csv(
        f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
    )
    new = f"/homedtic/dvarela/EXTRACTION/features/filtered/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    print(df.shape)
    pred_mem = nib.load(gasp_mem).get_fdata()
    ### filtrar los que estén en el DF
    mask = np.isin(pred_mem, list(df.original_labels))
    filt_pred_mem = mask * pred_mem
    ### filtrar myocardio
    mask_myo = np.isin(pred_mem, list(df[df.lines == 1].original_labels))
    filt_myo = mask_myo * pred_mem
    print(filt_myo.shape)
    mask_spl = np.isin(pred_mem, list(df[df.lines == 5].original_labels))
    filt_spl = mask_spl * pred_mem
    print(filt_spl.shape)
    # mask_som = np.isin(pred_mem, list(df[df.lines == 4].original_labels))
    # filt_som = mask_som * pred_mem
    # print(filt_som.shape)
    # mask_pro = np.isin(pred_mem, list(df[df.lines == 6].original_labels))
    # filt_pro = mask_pro * pred_mem
    # print(filt_pro.shape)
    c.saveNifti(
        filt_pred_mem,
        c.load3D_metadata(gasp_mem),
        new.replace("predictions_GASP.nii", "GASP_filtered.nii"),
    )
    c.saveNifti(
        filt_myo,
        c.load3D_metadata(gasp_mem),
        new.replace("predictions_GASP.nii", "GASP_filtered_myo.nii"),
    )
    c.saveNifti(
        filt_spl,
        c.load3D_metadata(gasp_mem),
        new.replace("predictions_GASP.nii", "GASP_filtered_spl.nii"),
    )
    # c.saveNifti(
    #     filt_som,
    #     c.load3D_metadata(gasp_mem),
    #     new.replace("predictions_GASP.nii", "GASP_filtered_som.nii"),
    # )
    # c.saveNifti(
    #     filt_pro,
    #     c.load3D_metadata(gasp_mem),
    #     new.replace("predictions_GASP.nii", "GASP_filtered_pro.nii"),
    # )
    print("-----------")


for i in os.listdir("/homedtic/dvarela/EXTRACTION/features"):
    if "csv" in i:
        df = pd.read_csv(os.path.join("/homedtic/dvarela/EXTRACTION/features", i))
        print(i)
        print(df.shape)


for i in os.listdir("/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features"):
    if "csv" in i:
        print(i)
        df = pd.read_csv(
            os.path.join(
                "/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features", i
            )
        )
        print(df.shape)
