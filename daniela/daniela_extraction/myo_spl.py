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
import porespy as ps
import math
import ast
from skimage import morphology
import json
import trimesh

# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region
import importlib

importlib.reload(cardiac_region)
import cardiac_region as cR

f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)
flatten_list = [
    element for sublist in [data[i] for i in ["stage6"]] for element in sublist
]
if __name__ == "__main__":
    for e in flatten_list:
        ESPECIMEN = f"2019{e}"
        print(ESPECIMEN)
        ######### INPUTS ----------------------------------------------------------------------------
        # gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        # DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        # mesh_myo = f"/Users/dvarelat/Documents/MASTER/TFM/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        # mesh_spl = f"/Users/dvarelat/Documents/MASTER/TFM/lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply"

        mesh_spl = f"/homedtic/dvarela/lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply"
        mesh_myo = f"/homedtic/dvarela/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        DFFILE = f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"

        ######### READ DATA ----------------------------------------------------------------------------
        myo = trimesh.load_mesh(mesh_myo, process=False)
        spl = trimesh.load_mesh(mesh_spl, process=False)
        df = pd.read_csv(DFFILE)

        dim_info = cR.load3D_metadata(gasp_mem)
        print(dim_info)
        pred_mem = nib.load(gasp_mem).get_fdata()
        print(pred_mem.shape)
        vertices_location_spl = np.floor(
            spl.vertices / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
        ).astype("uint16")
        ### por si los indices se salen del tamaño (0523_E1)
        vertices_location_spl = np.array(
            [
                v
                for v in vertices_location_spl
                if (
                    v[0] < dim_info["x_size"]
                    and v[1] < dim_info["y_size"]
                    and v[2] < dim_info["z_size"]
                )
            ]
        )
        vertices_labels_spl = pred_mem[
            vertices_location_spl[:, 0],
            vertices_location_spl[:, 1],
            vertices_location_spl[:, 2],
        ]
        vertices_location_myo = np.floor(
            myo.vertices / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
        ).astype("uint16")
        vertices_labels_myo = pred_mem[
            vertices_location_myo[:, 0],
            vertices_location_myo[:, 1],
            vertices_location_myo[:, 2],
        ]

        indices_lines5_spl = list(df[df.lines == 5].cell_in_props.values)

        indices_vertices_surface_spl = list(
            df[df.original_labels.isin(vertices_labels_spl)].cell_in_props.values
        )

        indices_lines1_myo = list(df[df.lines == 1].cell_in_props.values)

        indices_vertices_surface_myo = list(
            df[df.original_labels.isin(vertices_labels_myo)].cell_in_props.values
        )
        indices_props_myo = list(
            set(indices_vertices_surface_myo) | set(indices_lines1_myo)
        )
        indices_props_spl = list(
            set(indices_vertices_surface_spl) | set(indices_lines5_spl)
        )
        df["myo"] = df["cell_in_props"].apply(
            lambda x: 1 if x in indices_props_myo else 0
        )
        df["spl"] = df["cell_in_props"].apply(
            lambda x: 1 if x in indices_props_spl else 0
        )
        print(
            f"Final number of cells {len(indices_props_myo)} and {len(indices_props_spl)}"
        )
        df.to_csv(DFFILE, index=False, header=True)
        print(DFFILE)
        print("-------------")
