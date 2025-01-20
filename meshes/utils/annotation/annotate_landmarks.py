import numpy as np
import pandas as pd
import json

import open3d as o3d
import trimesh
import shapely
import rtree

from scipy.spatial import cKDTree

from util import values as v
from util.data.dataset_ht import find_group, HtDataset
from util.misc.colors import bcolors as bc


folder = v.data_path + 'ATLAS/myocardium/'
folder_out = v.data_path + 'Landmarks/ATLAS/'

# Global variable to store selected points in order
selected_points = []
selected_vertices = []


def picking_callback(vis):
    # This function is called when the user picks points
    picked_points_indices = vis.get_picked_points()
    # Since we want to record the points in order, we need to keep track
    # of which points have been added since the last callback
    global selected_points, previous_count, selected_vertices
    picked_points = [p.index for p in picked_points_indices[::-1]]

    if len(picked_points) != len(selected_points):
        picked_point = [p for p in picked_points if p not in np.array(selected_vertices)]
        print(picked_points)
        current_count = len(picked_points)
        new_indices = picked_point

        # Get the coordinates of the newly selected vertices
        selected_vertices.extend(new_indices)
        new_points = np.asarray(mesh.vertices)[new_indices]
        selected_points.extend(new_points)

        if current_count > previous_count:
            print(f"Selected points so far ({len(selected_points)}):")
            for idx, pt in enumerate(selected_points):
                print(f"Point {idx + 1}: {pt}")

        previous_count = current_count


if __name__ == '__main__':
    gr = 10
    mesh = o3d.io.read_triangle_mesh(folder + f'ATLAS_Gr{gr}.ply')
    mesh.compute_vertex_normals()

    # Create the VisualizerWithEditing instance
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(mesh)

    # Initialize previous count
    previous_count = 0

    # Register the picking callback
    vis.register_animation_callback(picking_callback)

    print("Instructions:")
    print(" - Use [shift + left click] to pick points on the mesh.")
    print(" - Press 'Q' or close the window when done.")

    # Run the visualizer (user picks points here)
    vis.run()
    vis.destroy_window()

    # After the visualizer is closed, selected_points contains the points in order
    selected_points = np.array(selected_points)
    selected_dict = {idx + 1: v.tolist() for idx, v in enumerate(selected_points)}
    selected_dict_pinned = {idx + 1: v for idx, v in enumerate(selected_vertices)}

    print("Final selected landmark points in clicking order:")
    for idx, pt in enumerate(selected_points):
        print(f"Point {idx + 1}: {pt}")


    reference_list = v.myo_myo_landmark_names
    selected_dict_aux = {}

    # print(f'Landmark names:')
    # [print(f'\t{name}') for name in reference_list]

    print(len(reference_list), len(selected_dict))
    assert len(selected_dict) == len(reference_list), 'Number of landmarks does not match.'

    for name in selected_dict.keys():
        # Change the key to the landmark name
        selected_dict_aux[reference_list[int(name) - 1]] = selected_dict[name]

    print(selected_dict_aux)

    with open(folder_out + f'ATLAS_Gr{gr}_key_points.json', 'w') as f:
        json.dump(selected_dict_aux, f)

    with open(folder_out + f'ATLAS_Gr{gr}_landmarks.pinned', 'w') as f:
        f.write('\n'.join([str(v) for v in selected_dict_pinned.values()]))