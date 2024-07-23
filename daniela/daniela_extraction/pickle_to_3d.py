from lib2to3.pgen2.pgen import DFAState
import pickle
import numpy as np
import pandas as pd
import trimesh
import json

# CONDA ACTIVATE PROESPY


def feature_colorRGB(df, variable):
    variable255 = df[variable] - df[variable].min()
    variable255 = 255 * (variable255 / variable255.max())
    variableRGB = rgbA(variable255)
    return [list(i) for i in variableRGB]


def rgbA(value, only_one_color="rgb", minimum=0, maximum=255):
    # if minimum != -1 or maximum != -1:
    # minimum, maximum = float(min(value)), float(max(value))
    zeros = np.zeros(shape=value.shape, dtype=value.dtype)
    if only_one_color == "rgb":
        ratio = 2 * (value - minimum) / (maximum - minimum)

        b = np.maximum(zeros, 255 * (1 - ratio))
        r = np.maximum(zeros, 255 * (ratio - 1))

        b = np.asarray(b, dtype="uint8")
        r = np.asarray(r, dtype="uint8")

        g = 255 - b - r

        rgb0 = np.array([r, g, b, zeros + 255]).transpose()

        return rgb0.astype("uint8")


def merge(parts, concatenate_colors=True):
    # Concateno los vertices
    vertices = np.concatenate(([p.vertices for p in parts]), axis=0)

    # En cada componente de parts, el indice de las faces empieza desde 0
    # Con offset, se calcula el indice que se debe sumar a las caras de cada parte
    # cuando ests se unan en el mismo objeto trimesh
    offset = [0]
    for i, p in enumerate(parts):
        offset.append(int(p.faces.max() + 1))
    offset = np.cumsum(offset)

    # Concateno las caras añadiendo el offset
    faces = np.concatenate(([p.faces + offset[i] for i, p in enumerate(parts)]), axis=0)

    # Creo nueva mesh
    mesh2send = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Concateno los colores
    if concatenate_colors:
        colors = np.concatenate(
            ([p.visual.vertex_colors for i, p in enumerate(parts)]), axis=0
        )
        mesh2send.visual.vertex_colors = colors

    return mesh2send


if __name__ == "__main__":

    # f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
    # data = json.load(f)
    # flatten_list = [
    #     element for sublist in [data[i] for i in ["stage6"]] for element in sublist
    # ]
    flatten_list = ["0806_E3", "0401_E3", "0401_E1"]

    variables = [
        # "lines",
        # "improved_lines",
        # "sphericities",
        # "eccentricity3d",
        # "volumes",
        # "solidity",
        # "Elongation",
        # "Sphericity",
        # "MeshVolume",
        # "Flatness",
        # "daniela_division"
        "columnarity",
        "Elongation2",
        "angles",
    ]
    for e in flatten_list:
        ESPECIMEN = f"2019{e}"
        print(ESPECIMEN)
        spl_list = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_MYO_lines_corr_filter.pkl"
        # DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/orientation/{ESPECIMEN}_angles_myo_filter.csv"
        with open(spl_list, "rb") as f:
            readMESHES = pickle.load(f)
        list_bads = []
        print(f"List meshes --> {len(readMESHES)}")
        df = pd.read_csv(DFFILE)
        # df = df[df.spl == 1]  # Si la entrada es radiomics
        print(df.shape)
        # print(df.columns)
        print(f"Features SPLACHN {df.shape}")
        if len(readMESHES) == df.shape[0]:
            for variable in variables:
                print(variable)
                # OUT_myo = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/orientation/SPLcells_{ESPECIMEN}_{variable}.ply"
                OUT_spl = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/meshes/MYOcells_{ESPECIMEN}_{variable}_filter.ply"
                print(OUT_spl)
                # df_angles = pd.read_csv(OUTFILE)
                featRGB = feature_colorRGB(df, variable)
                # dict_label_rgb = dict(zip(df.original_labels, featRGB))
                # labels = list(df.original_labels)
                for j, color in enumerate(featRGB):
                    readMESHES[j].visual.vertex_colors = color
                all_readMESHES = merge(readMESHES)
                print(all_readMESHES)
                all_readMESHES.export(OUT_spl)
        else:
            print("No same size --> correct based on json")
        print("----------------------")
