#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Break
from unittest import skip
import numpy as np
import sys
import os
import numpy as np
from importlib import reload
import nibabel as nib
import pandas as pd
import importlib
from skimage import morphology
import porespy as ps
from collections import Counter
from math import sqrt
import json


f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
# f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)


def eccentricity_3D(prop):
    l1, l2, l3 = prop.inertia_tensor_eigvals
    if l1 == 0:
        return 0
    return sqrt(1 - l2 / l1)


if __name__ == "__main__":
    flatten_list = [
        element
        for sublist in [data[i] for i in ["stage1", "stage2", "stage3", "stage4"]]
        for element in sublist
    ]
    mems = "/homedtic/dvarela/CardiacRegion/all/mem"
    # mems = "/Users/dvarelat/Documents/MASTER/TFM/DATA/CNIC/paraDaniela/mem"
    FOLDERS = [
        element
        for sublist in [
            [f"{i[-1]}_2019" + e for e in data[i]]
            for i in ["stage2", "stage3", "stage4"]
        ]
        for element in sublist
    ]
    for folder in ["3_20190516_E3"]:
        print(folder)
        ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
        # FOLDER = f"/homedtic/dvarela/EXTRACTION/{folder}"
        # FILE = f"/homedtic/dvarela/EXTRACTION/{folder}/{ESPECIMEN}_cell_properties_radiomics.csv"
        FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        linefile = (
            f"/Users/dvarelat/Documents/MASTER/TFM/DATA/LINES/line_{ESPECIMEN}.nii.gz"
        )
        gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        # linefile = f"/homedtic/dvarela/LINES/line_{ESPECIMEN}.nii.gz"
        # gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"

        ## LEER ARCHIVOS
        pred_mem = nib.load(gasp_mem).get_fdata()
        print(pred_mem.shape)

        lines = nib.load(linefile).get_fdata()
        print(lines.shape)

        ## props membrane
        props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
        centroids_mem = [[round(i) for i in p["centroid"]] for p in props_mem]
        original_labels_centroids = [pred_mem[c[0], c[1], c[2]] for c in centroids_mem]
        print("YA CASI")
        ### LINES EXTRACTION - TOUCHING
        most_communs = []
        for p in props_mem:
            b = Counter(lines[p.slices].flatten())
            if len(list(b)) == 1:  ## si es solo 1, ese es
                m = list(b)[0]
            else:  # si hay varios. coger mayor diff cero
                d = {key: val for key, val in dict(b).items() if key != 0}
                m = max(d, key=d.get)
            most_communs.append(m)

        l = []
        for p in props_mem:
            try:
                l.append(p.axis_minor_length)
            except:
                l.append(0)

        df = pd.DataFrame(
            {
                "cell_in_props": range(len(props_mem)),
                "volumes": [p.volume for p in props_mem],
                "sphericities": [p.sphericity for p in props_mem],
                "original_labels": original_labels_centroids,
                "centroids": centroids_mem,
                "lines": most_communs,
                "axis_major_length": [p.axis_major_length for p in props_mem],
                "axis_minor_length": l,
                "solidity": [p.solidity for p in props_mem],
                # "feret_diameter_max": [p.feret_diameter_max for p in props_mem],
            }
        )
        f = []
        for i, p in enumerate(props_mem):
            try:
                f.append(p.feret_diameter_max)
            except:
                f.append(0)
        df["feret_diameter_max"] = f

        df = df[df.original_labels != 0]
        df = df[df.volumes < 1.5 * np.median(df.volumes)]
        df = df[df.volumes > 0.2 * np.median(df.volumes)]
        df = df[df.feret_diameter_max != 0]
        print(df.shape)
        df.to_csv(FILE, index=False, header=True)
        print(FILE)
        print("-------------")

        ### centroid approach for nuclei
        ## mask on nuclei
        cellpose_nu = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/nuclei/{ESPECIMEN}_MASK_EQ_XYZ_decon.nii.gz"
        pred_nu = nib.load(cellpose_nu).get_fdata()
        print(pred_nu.shape)
        mask_mem = np.where(pred_mem != 0, True, False)
        mask_on_nuclei = mask_mem * pred_nu
        props_nu = ps.metrics.regionprops_3D(morphology.label(mask_on_nuclei))
        centroids_nu = [[round(i) for i in p["centroid"]] for p in props_nu]
        NU_original_labels_centroids = [pred_nu[c[0], c[1], c[2]] for c in centroids_nu]
        labels_centroid_nu_in_mem = [pred_mem[c[0], c[1], c[2]] for c in centroids_nu]

        dict_nuclei_membrane_centroids = dict(
            zip(labels_centroid_nu_in_mem, NU_original_labels_centroids)
        )
        dict_memlabel_cellnumber_props_nuclei = dict(
            zip(labels_centroid_nu_in_mem, range(len(centroids_nu)))
        )

        df["nuclei_label_cent"] = [
            dict_nuclei_membrane_centroids[label]
            if label in labels_centroid_nu_in_mem
            else -1
            for label in df.original_labels
        ]
        df["nuclei_cell_in_props"] = [
            dict_memlabel_cellnumber_props_nuclei[label]
            if label in labels_centroid_nu_in_mem
            else -1
            for label in df.original_labels
        ]
        df.to_csv(FILE, index=False, header=True)
