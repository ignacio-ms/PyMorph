import numpy as np
import pandas as pd

import trimesh

import os
import pandas as pd
import trimesh
from trimesh.collision import CollisionManager
from collections import defaultdict
from plyfile import PlyData, PlyElement

from auxiliary.utils.colors import bcolors as c


def load_csv(csv_path):
    """Load the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    if 'cell_id' not in df.columns:
        raise ValueError("The CSV file must contain a 'cell_id' column.")
    return df


def load_mesh(mesh_path):
    """Load a mesh from a file and repair if not watertight."""
    mesh = trimesh.load(mesh_path)
    if not mesh.is_watertight:
        print(f"Mesh at {mesh_path} is not watertight. Attempting to repair.")
        mesh = mesh.fill_holes()
    return mesh


def extract_cell_faces(merged_mesh):
    """Extract cell_id for each face."""
    try:
        face_cell_ids = merged_mesh.metadata['_ply_raw']['face']['data']['cell_id']
        return face_cell_ids
    except KeyError:
        raise KeyError("Merged mesh does not contain 'cell_id' in face metadata.")


def group_faces_by_cell_id(face_cell_ids):
    """Group face indices by cell_id."""
    cell_faces_dict = defaultdict(list)
    for face_idx, cell_id in enumerate(face_cell_ids):
        cell_faces_dict[cell_id[0]].append(face_idx)
    return cell_faces_dict


def create_individual_cell_meshes(merged_mesh, cell_faces_dict, verbose=0):
    """Create individual cell meshes using submesh method with validation."""
    individual_cells = []
    for cell_id, face_indices in cell_faces_dict.items():
        mask = np.zeros(len(merged_mesh.faces), dtype=bool)
        mask[face_indices] = True

        cell_mesh = merged_mesh.submesh([mask], append=True, repair=False)

        # **Validation Checks**
        issues = False

        # Check if mesh is empty
        if cell_mesh.is_empty:
            print(f"Warning: Cell ID {cell_id} has an empty submesh. Skipping.")
            issues = True

        # Check if all faces are triangles
        if cell_mesh.faces.shape[1] != 3:
            print(f"Warning: Cell ID {cell_id} does not have triangular faces. Skipping.")
            issues = True

        # Check if mesh has vertices and faces
        if len(cell_mesh.vertices) == 0 or len(cell_mesh.faces) == 0:
            print(f"Warning: Cell ID {cell_id} has no vertices or faces. Skipping.")
            issues = True

        # Check for non-manifold edges or degenerate faces
        if not cell_mesh.is_watertight:
            print(f"Warning: Cell ID {cell_id} mesh is not watertight. Attempting to repair.")
            cell_mesh = cell_mesh.fill_holes()
            if cell_mesh.is_empty or not cell_mesh.is_watertight:
                print(f"Error: Failed to repair Cell ID {cell_id}. Skipping.")
                issues = True

        # **Ensure Correct Data Types**
        cell_mesh.vertices = cell_mesh.vertices.astype(np.float64)
        cell_mesh.faces = cell_mesh.faces.astype(np.int64)

        # **Final Validation**
        if not issues:
            individual_cells.append((cell_id, cell_mesh))

    return individual_cells


def perform_intersection_checks(tissue_mesh, individual_cells):
    """Identify intersecting and non-intersecting cells."""
    collision_manager = CollisionManager()
    collision_manager.add_object('tissue', tissue_mesh)

    intersecting_cell_ids = []
    non_intersecting_cell_ids = []

    for cell_id, cell_mesh in individual_cells:
        try:
            is_collision = collision_manager.in_collision_single(cell_mesh)
            if is_collision:
                intersecting_cell_ids.append(cell_id)
            else:
                non_intersecting_cell_ids.append(cell_id)
        except Exception as e:
            print(f"Error during collision check for Cell ID {cell_id}: {e}")
            non_intersecting_cell_ids.append(cell_id)

    return intersecting_cell_ids, non_intersecting_cell_ids


def update_csv(csv_path, non_intersecting_cell_ids, verbose=0):
    """Remove non-intersecting cells from the CSV and save the updated CSV."""
    df = pd.read_csv(csv_path)

    if 'cell_id' not in df.columns:
        raise ValueError("The CSV file does not contain a 'cell_id' column.")

    initial_count = len(df)
    df_filtered = df[~df['cell_id'].isin(non_intersecting_cell_ids)]
    final_count = len(df_filtered)

    df_filtered.to_csv(csv_path, index=False)
    if verbose:
        print(f"Removed {initial_count - final_count} cells from the CSV.")
        print(f"Filtered CSV saved to {csv_path}")


def merge_individual_cells(individual_cells, intersecting_cell_ids):
    all_vertices = []
    all_faces = []
    all_normals = []
    all_vertex_cell_ids = []
    all_face_cell_ids = []

    intersecting = {cell_id: cell for cell_id, cell in individual_cells if cell_id in intersecting_cell_ids}

    intersecting_cells = intersecting.values()
    ids = intersecting.keys()

    vertex_offset = 0
    for cell_id, cell_mesh in zip(ids, intersecting_cells):
        if cell_mesh is None:
            continue  # Skip invalid meshes
        all_vertices.append(cell_mesh.vertices)
        all_faces.append(cell_mesh.faces + vertex_offset)
        all_normals.append(cell_mesh.vertex_normals)
        all_vertex_cell_ids.append(np.full(len(cell_mesh.vertices), cell_id, dtype=np.int32))
        all_face_cell_ids.append(np.full(len(cell_mesh.faces), cell_id, dtype=np.int32))
        vertex_offset += len(cell_mesh.vertices)

    merged_vertices = np.vstack(all_vertices)
    merged_faces = np.vstack(all_faces)
    merged_normals = np.vstack(all_normals)
    merged_vertex_cell_ids = np.concatenate(all_vertex_cell_ids)
    merged_face_cell_ids = np.concatenate(all_face_cell_ids)

    return {
        'vertices': merged_vertices,
        'faces': merged_faces,
        'normals': merged_normals,
        'vertex_cell_ids': merged_vertex_cell_ids,
        'face_cell_ids': merged_face_cell_ids
    }


def create_structured_arrays(merged_mesh_data):
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('cell_id', 'i4')
    ]

    face_dtype = [
        ('vertex_indices', 'i4', (3,)),
        ('cell_id', 'i4')
    ]

    vertex_data = np.empty(len(merged_mesh_data['vertices']), dtype=vertex_dtype)
    vertex_data['x'] = merged_mesh_data['vertices'][:, 0]
    vertex_data['y'] = merged_mesh_data['vertices'][:, 1]
    vertex_data['z'] = merged_mesh_data['vertices'][:, 2]
    vertex_data['nx'] = merged_mesh_data['normals'][:, 0]
    vertex_data['ny'] = merged_mesh_data['normals'][:, 1]
    vertex_data['nz'] = merged_mesh_data['normals'][:, 2]
    vertex_data['cell_id'] = merged_mesh_data['vertex_cell_ids']

    face_data = np.empty(len(merged_mesh_data['faces']), dtype=face_dtype)
    face_data['vertex_indices'] = merged_mesh_data['faces']
    face_data['cell_id'] = merged_mesh_data['face_cell_ids']

    return vertex_data, face_data


def create_ply_elements(vertex_data, face_data):
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')
    return [vertex_element, face_element]


def save_ply(path_out, ply_elements, binary=False):
    PlyData(ply_elements, text=True).write(path_out)
    print(f"PLY file saved to {path_out}")


def verify_data(vertex_data, face_data):
    if len(vertex_data) == 0:
        print("Error: No vertices to export.")
        return False
    if len(face_data) == 0:
        print("Error: No faces to export.")
        return False
    for face in face_data:
        if face['vertex_indices'].shape != (3,):
            print("Error: All faces must have exactly 3 vertex indices.")
            return False
    return True


def check_non_manifold(mesh):
    return not mesh.is_watertight


def merge_and_save_mesh(individual_cells, intersecting_cell_ids, path_out):
    merged_mesh_data = merge_individual_cells(individual_cells, intersecting_cell_ids)
    vertex_data, face_data = create_structured_arrays(merged_mesh_data)

    if not verify_data(vertex_data, face_data):
        print("Data verification failed. Aborting export.")
        return

    # Create Trimesh object to check manifoldness
    merged_mesh = trimesh.Trimesh(vertices=merged_mesh_data['vertices'],
                                  faces=merged_mesh_data['faces'],
                                  process=False)
    if check_non_manifold(merged_mesh):
        print("Warning: Merged mesh has non-manifold edges. Attempting to repair.")
        merged_mesh = merged_mesh.fill_holes()
        if check_non_manifold(merged_mesh):
            print("Error: Unable to repair non-manifold edges.")
            return
        else:
            # Update merged_mesh_data after repair
            merged_mesh_data['vertices'] = merged_mesh.vertices
            merged_mesh_data['faces'] = merged_mesh.faces
            merged_mesh_data['normals'] = merged_mesh.vertex_normals
            # Note: cell_ids remain unchanged as per previous steps
            vertex_data, face_data = create_structured_arrays(merged_mesh_data)

    ply_elements = create_ply_elements(vertex_data, face_data)
    save_ply(path_out, ply_elements, binary=False)

    return merged_mesh


def visualize_results(tissue_mesh, merged_intersecting_mesh):
    """Visualize the tissue mesh and merged intersecting cell meshes."""
    if merged_intersecting_mesh:
        # Assign distinct colors
        tissue_mesh.visual.face_colors = [0, 255, 0, 100]  # Semi-transparent green
        merged_intersecting_mesh.visual.face_colors = [255, 0, 0, 100]  # Semi-transparent red

        # Create a Trimesh scene
        scene = trimesh.Scene()
        scene.add_geometry(tissue_mesh, node_name='Tissue')
        scene.add_geometry(merged_intersecting_mesh, node_name='Intersecting_Cells')

        # Display the scene
        scene.show()
    else:
        print("No intersecting cells to visualize.")


def run(mesh_path, tissue_path, features_path, verbose=0):
    features = load_csv(features_path)
    tissue_mesh = load_mesh(tissue_path)
    merged_mesh = load_mesh(mesh_path)

    face_cell_ids = extract_cell_faces(merged_mesh)
    cell_faces_dict = group_faces_by_cell_id(face_cell_ids)
    individual_cells = create_individual_cell_meshes(merged_mesh, cell_faces_dict)

    if verbose:
        print(f"Found {c.BOLD}{len(individual_cells)}{c.ENDC} individual cell meshes.")
        print(f"{c.OKBLUE}Performing intersection checks{c.ENDC}...")

    intersecting_cell_ids, non_intersecting_cell_ids = perform_intersection_checks(tissue_mesh, individual_cells)

    if verbose:
        print(f"{c.OKBLUE}Updating CSV{c.ENDC}...")

    update_csv(features_path, non_intersecting_cell_ids, verbose)

    if verbose:
        print(f"{c.OKBLUE}Merging intersecting cells{c.ENDC}...")

    merged_intersecting_mesh = merge_and_save_mesh(individual_cells, intersecting_cell_ids, mesh_path)

    if verbose:
        total_cells = len(individual_cells)
        intersecting_cells = len(intersecting_cell_ids)
        non_intersecting_cells = len(non_intersecting_cell_ids)
        percentage = (intersecting_cells / total_cells) * 100 if total_cells > 0 else 0
        
        print(f'{c.OKGREEN}Summary{c.ENDC}')
        print(f'\tTotal Cells: {total_cells}')
        print(f'\tIntersecting Cells: {intersecting_cells} ({percentage:.2f}%)')
        print(f'\tNon-intersecting Cells: {non_intersecting_cells}')



