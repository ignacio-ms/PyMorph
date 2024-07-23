import trimesh
import numpy as np
import sys
import importlib
import pandas as pd
import nibabel as nib
import porespy as ps

import importlib
import mcubes
import pickle
from numpy import dot
from skimage import morphology
import json

sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
# sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as cR


def get_mainaxis(mesh):
    cov = np.cov(mesh.vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigen_i = np.where(np.asarray(eigenvalues == eigenvalues.max()))[0][0]
    return eigenvectors[:, eigen_i]


def calculate_elongation(mesh):
    """Igual a como se calcula en pyradiomics"""
    cov = np.cov(mesh.vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenValues = eigenvalues[idx]
    return np.sqrt(eigenValues[1] / eigenValues[0])  ## minor/major


# f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
# # f = open("/homedtic/dvarela/specimens.json")
# data = json.load(f)

# FOLDERS = [
#     element
#     for sublist in [
#         [f"{i[-1]}_2019" + e for e in data[i]]
#         for i in ["stage1", "stage2", "stage3", "stage4"]
#     ]
#     for element in sublist
# ]

if __name__ == "__main__":
    # f = open("/homedtic/dvarela/specimens.json")
    # data = json.load(f)
    # flatten_list = [
    #     element for sublist in [data[i] for i in ["stage6"]] for element in sublist
    # ]
    flatten_list = ["0401_E2"]
    for e in flatten_list:
        ESPECIMEN = f"2019{e}"
        print(ESPECIMEN)
        ######### INPUTS ----------------------------------------------------------------------------
        pickle_spl = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_MYO_lines_corr.pkl"
        DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        line_mesh = f"/Users/dvarelat/Documents/MASTER/TFM/lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply"
        gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        OUTFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/orientation/{ESPECIMEN}_angles_myo.csv"
        # cluster
        # pickle_spl = f"/homedtic/dvarela/EXTRACTION/features/list_meshes/{ESPECIMEN}_SPL_lines_corr.pkl"
        # gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        # DFFILE = f"/homedtic/dvarela/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        # line_mesh = f"/homedtic/dvarela/lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply"
        # OUTFILE = f"/homedtic/dvarela/EXTRACTION/features/orientation/{ESPECIMEN}_angles_spl.csv"

        #### --------------------------------------
        mesh = trimesh.load_mesh(line_mesh, process=False)
        final_number_faces = 18000
        mesh = mesh.simplify_quadratic_decimation(int(final_number_faces))
        v_normals = mesh.vertex_normals
        dim_info = cR.load3D_metadata(gasp_mem)
        df = pd.read_csv(DFFILE)
        with open(pickle_spl, "rb") as f:
            readMESHES = pickle.load(f)

        # SPLANCHNIC or MYO
        df = df[df.myo == 1]
        if len(readMESHES) == df.shape[0]:
            print(f"MISMO SIZE {len(readMESHES)}")
            pred_mem = nib.load(gasp_mem).get_fdata()
            props = ps.metrics.regionprops_3D(morphology.label(pred_mem))
            indices_props = list(df.cell_in_props.values)
            print(f"Final number of cells {len(indices_props)}")
            props_set = [props[i] for i in indices_props]
            index_vertice = []
            mains = []
            elongation = []
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
            for i, p in enumerate(props_set):
                print(i)
                # print("Calculating closest vertex...")
                distances_to_my_centr = [
                    np.linalg.norm(np.array(np.array(v)) - np.array(p.centroid))
                    for v in vertices_location
                ]
                index = np.argsort(distances_to_my_centr)[0]
                index_vertice.append(index)
                mains.append(get_mainaxis(readMESHES[i]))
                elongation.append(calculate_elongation(readMESHES[i]))
            DF_ANGLES = pd.DataFrame()
            DF_ANGLES["cell_in_props"] = indices_props
            DF_ANGLES["closest_vert"] = index_vertice
            angles = []
            for i, normal in enumerate(v_normals[index_vertice]):
                angles.append(abs(dot(normal, mains[i])))
            DF_ANGLES["angles"] = angles
            DF_ANGLES["Elongation2"] = elongation
            DF_ANGLES.to_csv(OUTFILE, index=False, header=True)
            print(OUTFILE)
        else:
            print("DIFF TAMAÑO")
        print("--------------------------")
            