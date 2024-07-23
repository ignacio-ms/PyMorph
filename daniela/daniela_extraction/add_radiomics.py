
import pandas as pd
import ast
import sys
import os
import logging
import six
import radiomics
import os
import SimpleITK as sitk
import nibabel as nib
from radiomics import featureextractor, getFeatureClasses, shape
import porespy as ps
import pandas as pd
import numpy as np
from skimage import morphology

import json


f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
#f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)


sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
#sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region
import importlib

importlib.reload(cardiac_region)
import cardiac_region as cR


FOLDERS = [
        element
        for sublist in [
            [f"{i[-1]}_2019" + e for e in data[i]]
            for i in ["stage1","stage2","stage3", "stage4"]
        ]
        for element in sublist
    ]
FOLDERS = ["3_20190516_E3"]
for i, folder in enumerate(FOLDERS):
    print(folder)
    ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
    # mem = f"/homedtic/dvarela/DECON_05/MGFP/mem/{ESPECIMEN}_mGFP_decon_0.5.nii.gz"
    # FILE = f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
    # gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/CNIC/paraDaniela/mem/{ESPECIMEN}_mGFP_decon_0.5.nii.gz"
    FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
    linefile = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/CNIC/paraDaniela/lines/line_{ESPECIMEN}.nii.gz"
    gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        
    MEM = nib.load(mem).get_fdata()
    MEM = MEM[:, :, :, 0]
    print(MEM.shape)
    pred_mem = nib.load(gasp_mem).get_fdata()
    print(pred_mem.shape)
    df = pd.read_csv(FILE)
    print(df.shape)
    print(df.head())
    
    props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
    print(len(props_mem))
    results = []
    for cell in df.cell_in_props:
        #print(cell)
        img = np.swapaxes(np.swapaxes(MEM[props_mem[cell].slice], 0, 2), 1, 2)
        m = np.swapaxes(np.swapaxes(props_mem[cell].mask, 0, 2), 1, 2)
        spacing = [
            float(cR.load3D_metadata(mem)["x_res"]),
            float(cR.load3D_metadata(mem)["y_res"]),
            float(cR.load3D_metadata(mem)["z_res"]),
        ]
        #print(spacing)
        sitk_img = sitk.GetImageFromArray(img)
        sitk_img = sitk.JoinSeries(sitk_img)[:, :, :, 0]
        sitk_img.SetSpacing(spacing)
        sitk_mask = sitk.GetImageFromArray(m.astype("uint16"))
        sitk_mask = sitk.JoinSeries(sitk_mask)[:, :, :, 0]
        sitk_mask.SetSpacing(spacing)
        shapeFeatures = shape.RadiomicsShape(
            sitk_img,
            sitk_mask,
        )
        shapeFeatures.enableAllFeatures()
        result = shapeFeatures.execute()
        results.append(result)
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    print(df.shape)
    print(df.head())
    
    # #### IMPROVE LINES
    # print("Improving lines---")
    # margenesXYZ = cR.crop_line(linefile, gasp_mem, escala2048=False, ma=1)
    # crop_m = cR.crop_embryo(margenesXYZ, gasp_mem)

    # im = np.ones(shape=pred_mem.shape)
    # im[
    #         margenesXYZ[0][0] : margenesXYZ[1][0],
    #         margenesXYZ[0][1] : margenesXYZ[1][1],
    #         margenesXYZ[0][2] : margenesXYZ[1][2],
    #     ] = 0
    # mask_borders = im * pred_mem
    # labels = np.unique(mask_borders)
    # df["improved_lines_corr"] = df.apply(lambda x: 0 if x["original_labels"] in labels else x["improved_lines"], axis=1)
    # dfnoback = df[df.lines !=0]
    # dfnoback = dfnoback.reset_index(drop=True)
    # dfback = df[df.lines==0]
    # new_lines = {}
    # d = 45
    # for i in dfback.cell_in_props:
    #     c = props_mem[i].centroid
    #     distances = dfnoback["centroids"].apply(lambda x: np.linalg.norm(np.array(ast.literal_eval(x))-np.array(c)))
    #     index = np.argsort(distances)[0]
    #     #print(f"Distance --> {distances[index]}")
    #     if distances[index] > d:
    #         new_lines[i] = 0
    #     else:
    #         new_lines[i] = dfnoback.loc[index, ["lines"]].values[0]
    # df["improved_lines"] = df.apply(lambda x: x["lines"] if x["lines"] !=0 else new_lines[x["cell_in_props"]], axis=1)
    # df.to_csv(FILE, index=False, header=True)
    # print(df.shape)
    print(FILE)
    df.to_csv(FILE, index=False, header=True)
    print("---------------------")
    