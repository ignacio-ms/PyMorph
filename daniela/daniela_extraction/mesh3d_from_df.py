#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### SCRIPT TO BE RUN ON PORESPY CONDA ENV

import numpy as np
import scipy
import skimage
import porespy as ps
import mcubes
import trimesh
import csv
from skimage import morphology
from skimage.morphology import disk
import nibabel as nib
import pandas as pd
import time
import ast
import pickle
import sys
import importlib
import json
import os

# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as cR


def Median3D_Array(NumpyArray, disk_size):
    # disk_size: el tamaño de los objetos que el median filter elimina
    # objetos aislados de menor o igual tamaño se los ventila
    x_height, y_height, z_depth = (
        NumpyArray.shape[0],
        NumpyArray.shape[1],
        NumpyArray.shape[2],
    )

    # Hay arrays que se guardan como una cuarta dimension de valor =1 que no se porque lo hace...
    # Con esto lo quito
    if len(NumpyArray.shape) == 4:
        NumpyArray = NumpyArray[:, :, :, 0]

    return scipy.ndimage.median_filter(NumpyArray, size=disk_size)
    print("NO LLEGA")

    # Bluring over the {xy} plane
    for i in range(0, z_depth):
        NumpyArray[:, :, i] = skimage.filters.median(
            NumpyArray[:, :, i], disk(disk_size)
        )
    # Bluring over the {xz} plane
    for i in range(0, y_height):
        NumpyArray[:, i, :] = skimage.filters.median(
            NumpyArray[:, i, :], disk(disk_size)
        )
    # Bluring over the {yz} plane
    for i in range(0, x_height):
        NumpyArray[i, :, :] = skimage.filters.median(
            NumpyArray[i, :, :], disk(disk_size)
        )

    return NumpyArray


# f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
# f = open("/homedtic/dvarela/specimens.json")
# data = json.load(f)

# flatten_list = [
#     element for sublist in [data[i] for i in ["stage6"]] for element in sublist
# ]
flatten_list = ["0806_E3", "0401_E3", "0401_E1"]
if __name__ == "__main__":
    dict_bads = {}
    for e in flatten_list:
        ESPECIMEN = f"2019{e}"

        print(ESPECIMEN)
        ######### INPUTS ----------------------------------------------------------------------------
        # gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        # FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        # file_out = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_MYO_lines_corr.pkl"
        # line_mesh = f"/Users/dvarelat/Documents/MASTER/TFM/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        # bad_json = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/pickles_myo_{ESPECIMEN}.json"

        # CLUSTER
        gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        FILE = f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        file_out = f"/homedtic/dvarela/EXTRACTION/features/list_meshes/{ESPECIMEN}_MYO_lines_corr.pkl"
        line_mesh = f"/homedtic/dvarela/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        bad_json = f"/homedtic/dvarela/EXTRACTION/features/list_meshes/pickles_myo_{ESPECIMEN}.json"
        #### --------------------------------------

        df = pd.read_csv(FILE)
        print(f"Features shape {df.shape}")
        pred_mem = nib.load(gasp_mem).get_fdata()
        dim_info = cR.load3D_metadata(gasp_mem)

        print(f"Segmentation shape {pred_mem.shape}")
        props = ps.metrics.regionprops_3D(morphology.label(pred_mem))
        df = df[df.myo == 1]
        indices_props = list(df.cell_in_props.values)
        print(f"Numero cells - a partir de DF {len(indices_props)}")
        disk_size = 3
        meshes = []
        runtime = time.time()
        bad = []
        for i, cell_indice in enumerate(indices_props):
            print(i, "/", len(indices_props))
            prop = props[cell_indice]
            coords = prop.mask * 1
            add = 10
            aux = np.zeros(shape=tuple(np.asarray(coords.shape) + add), dtype="uint8")
            aux[
                add // 2 : -add // 2, add // 2 : -add // 2, add // 2 : -add // 2
            ] = coords.copy()
            coords = aux.copy()
            del aux
            coords = Median3D_Array(coords.copy(), disk_size)
            vert, trian = mcubes.marching_cubes(mcubes.smooth(coords), 0)
            vert -= vert.mean(axis=0)
            vert += prop.centroid
            vert *= np.asarray(
                [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
            )
            if len(vert) > 0 and len(trian) > 0:
                m_cell = trimesh.Trimesh(vert, trian, process=False)
                trimesh.smoothing.filter_taubin(
                    m_cell, lamb=0.5, nu=-0.5, iterations=20
                )
                m_cell = m_cell.simplify_quadratic_decimation(int(100))  ## BAJARLE
                meshes.append(m_cell)
            else:
                print("NO")
                bad.append(str(cell_indice))
        print(bad)
        dict_bads[ESPECIMEN] = bad

        runtime = time.time() - runtime
        print(f"Meshing took {runtime:.2f} s")
        print(f"Number elements {len(meshes)}")
        print(file_out)
        with open(file_out, "wb") as f:
            pickle.dump(meshes, f)
        print("-------------")
        print(dict_bads)
        with open(bad_json, "w") as outfile:
            json.dump(dict_bads, outfile)
