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


f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)

FOLDERS = [
    element
    for sublist in [
        [f"{i[-1]}_2019" + e for e in data[i]]
        for i in ["stage1", "stage2", "stage3", "stage4"]
    ]
    for element in sublist
]
if __name__ == "__main__":
    dict_bads = {}
    for folder in FOLDERS:
        ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
        print(ESPECIMEN)
        ##LOCAL
        # gasp = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        # FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{folder}/{ESPECIMEN}_cell_properties_radiomics.csv"
        # file_out = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_SPLmesh_lines.pkl"

        # CLUSTER
        gasp = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        FILE = f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        file_out = f"/homedtic/dvarela/EXTRACTION/features/list_meshes/{ESPECIMEN}_MYO_lines_corr.pkl"
        line_mesh = f"/homedtic/dvarela/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        # line_mesh = f"/homedtic/dvarela/lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply"
        if os.path.isfile(line_mesh):
            print(f"Sí existe --> {line_mesh}")
            df_clean = pd.read_csv(FILE)
            print(f"Features shape {df_clean.shape}")
            # df_clean = df_clean[df_clean.lines == 5]
            # print(f"Features SPLACHN {df_clean.shape}")
            pred_mem = nib.load(gasp).get_fdata()
            print(f"Segmentation shape {pred_mem.shape}")
            props = ps.metrics.regionprops_3D(morphology.label(pred_mem))
            print(f"LEN PROPS {len(props)}")

            ### extraer cells que entrarán
            dim_info = cR.load3D_metadata(gasp)
            mesh = trimesh.load_mesh(line_mesh, process=False)
            final_number_faces = 18000
            mesh = mesh.simplify_quadratic_decimation(int(final_number_faces))
            vertices_location = np.floor(
                mesh.vertices
                / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
            ).astype("uint16")
            vertices_location = np.array(
                [
                    v
                    for v in vertices_location
                    if (
                        v[0] < dim_info["x_size"]
                        and v[1] < dim_info["y_size"]
                        and v[2] < dim_info["z_size"]
                    )
                ]
            )

            vertices_labels = pred_mem[
                vertices_location[:, 0],
                vertices_location[:, 1],
                vertices_location[:, 2],
            ]
            indices_lines5 = list(df_clean[df_clean.lines == 1].cell_in_props.values)
            # print(f"Shape lines 5 --> {df_clean[df_clean.lines == 5].shape}")
            print(f"Shape lines 1 --> {df_clean[df_clean.lines == 1 ].shape}")
            indices_vertices_surface = list(
                df_clean[
                    df_clean.original_labels.isin(vertices_labels)
                ].cell_in_props.values
            )
            print(
                f"Shape vertices --> {df_clean[df_clean.original_labels.isin(vertices_labels)].shape}"
            )
            indices_props = list(set(indices_vertices_surface) | set(indices_lines5))
            print(f"Final number of cells {len(indices_props)}")
            ### -----------------
            # df_clean = df_clean[df_clean.cell_in_props.isin(indices_props)]
            ### -----------------
            # dict_label_cell = dict(
            #     zip(
            #         list(df_clean.original_labels),
            #         list(df_clean.cell_in_props),
            #     )
            # )
            disk_size = 3
            meshes = []
            runtime = time.time()
            bad = []
            for i, cell_indice in enumerate(indices_props):
                print(i, "/", len(indices_props))
                prop = props[cell_indice]
                coords = prop.mask * 1
                add = 10
                aux = np.zeros(
                    shape=tuple(np.asarray(coords.shape) + add), dtype="uint8"
                )
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
                    bad.append(cell_indice)
            print(bad)
            dict_bads[ESPECIMEN] = bad

            runtime = time.time() - runtime
            print(f"Meshing took {runtime:.2f} s")
            print(file_out)
            with open(file_out, "wb") as f:
                pickle.dump(meshes, f)
            print("-------------")
        else:
            print(f"NOT FOUND --> {line_mesh}")
    print(dict_bads)
    with open("pickles2.json", "w") as outfile:
        json.dump(dict_bads, outfile)
