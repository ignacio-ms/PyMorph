import numpy as np
import pandas as pd

import trimesh

import os
import pandas as pd
import trimesh
from trimesh.collision import CollisionManager
from collections import defaultdict
from plyfile import PlyData, PlyElement

from scipy.spatial import KDTree

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
    # if not mesh.is_watertight:
    #     print(f"Mesh at {mesh_path} is not watertight. Attempting to repair.")
    #     mesh = mesh.fill_holes()
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
            cell_mesh.fill_holes()
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


def calculate_centroid(mesh):
    """Calculate the centroid of a mesh."""
    return mesh.vertices.mean(axis=0)


def perform_intersection_checks_with_distance(tissue_mesh, individual_cells, distance_threshold):
    """Identify cells that intersect or are within a certain distance of the tissue mesh."""
    collision_manager = CollisionManager()
    collision_manager.add_object('tissue', tissue_mesh)

    intersecting_cell_ids = []
    close_cell_ids = []
    non_intersecting_cell_ids = []

    # Create a KDTree for the tissue mesh
    tissue_tree = KDTree(tissue_mesh.vertices)

    for cell_id, cell_mesh in individual_cells:
        try:
            # Check for collision
            is_collision = collision_manager.in_collision_single(cell_mesh)
            if is_collision:
                intersecting_cell_ids.append(cell_id)
                continue

            # Check for centroid distance
            cell_centroid = calculate_centroid(cell_mesh)
            distance, _ = tissue_tree.query(cell_centroid)
            if distance <= distance_threshold:
                close_cell_ids.append(cell_id)
            else:
                non_intersecting_cell_ids.append(cell_id)
        except Exception as e:
            print(f"Error during collision or distance check for Cell ID {cell_id}: {e}")
            non_intersecting_cell_ids.append(cell_id)

    # Combine intersecting and close cells
    valid_cell_ids = set(intersecting_cell_ids + close_cell_ids)
    return valid_cell_ids, non_intersecting_cell_ids


def update_csv(csv_path, non_intersecting_cell_ids, verbose=0):
    """Remove non-intersecting cells from the CSV and save the updated CSV."""
    df = pd.read_csv(csv_path)

    if 'cell_id' not in df.columns:
        if 'original_labels' in df.columns:
            df.rename(columns={'original_labels': 'cell_id'}, inplace=True)
        else:
            raise ValueError("The CSV file does not contain a 'cell_id' column.")

    initial_count = len(df)
    df_filtered = df[~df['cell_id'].isin(non_intersecting_cell_ids)]
    final_count = len(df_filtered)

    aux_path = csv_path.split('/')
    aux_path[-1] = 'Filtered/' + aux_path[-1].replace('.csv', '_filtered.csv')
    csv_path_filtered = '/'.join(aux_path)

    to_check = '/'.join(csv_path_filtered.split('/')[:-1])
    if not os.path.exists(to_check):
        os.makedirs(to_check)
        print(f"Created directory {to_check}")

    df_filtered.to_csv(csv_path_filtered, index=False)
    if verbose:
        print(f"Removed {initial_count - final_count} cells from the CSV.")
        print(f"Filtered CSV saved to {csv_path_filtered}")


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

    merged_mesh = trimesh.Trimesh(
        vertices=merged_mesh_data['vertices'],
        faces=merged_mesh_data['faces'],
        process=False
    )

    if check_non_manifold(merged_mesh):
        print("Warning: Merged mesh has non-manifold edges. Attempting to repair.")
        merged_mesh = merged_mesh.fill_holes()
        if check_non_manifold(merged_mesh):
            print("Error: Unable to repair non-manifold edges.")
            return
        else:
            merged_mesh_data['vertices'] = merged_mesh.vertices
            merged_mesh_data['faces'] = merged_mesh.faces
            merged_mesh_data['normals'] = merged_mesh.vertex_normals

            vertex_data, face_data = create_structured_arrays(merged_mesh_data)

    ply_elements = create_ply_elements(vertex_data, face_data)
    save_ply(path_out, ply_elements, binary=False)

    return merged_mesh


def visualize_results(tissue_mesh, individual_cells, valid_cell_ids, distance_threshold):
    """
    Visualize the tissue mesh, intersecting cells, and cells within the distance threshold.
    Parameters:
        tissue_mesh: The tissue mesh geometry.
        individual_cells: List of individual cell meshes with their IDs.
        valid_cell_ids: Set of cell IDs that are either intersecting or within the distance threshold.
        distance_threshold: The distance threshold for selecting nearby cells.
    """
    scene = trimesh.Scene()

    # Add the tissue mesh with a distinct color
    tissue_mesh.visual.face_colors = [0, 255, 0, 100]  # Semi-transparent green
    scene.add_geometry(tissue_mesh, node_name='Tissue')

    for cell_id, cell_mesh in individual_cells:
        # Calculate the centroid distance for visualization
        cell_centroid = calculate_centroid(cell_mesh)

        # Reshape the centroid to (1, 3) for `on_surface`
        cell_centroid_reshaped = cell_centroid.reshape(1, 3)
        centroid_distance = tissue_mesh.nearest.on_surface(cell_centroid_reshaped)[1][0]  # Extract scalar distance

        if cell_id in valid_cell_ids:
            # Intersecting or within distance threshold
            if centroid_distance <= distance_threshold:
                cell_mesh.visual.face_colors = [0, 0, 255, 150]  # Semi-transparent blue for near cells
            else:
                cell_mesh.visual.face_colors = [255, 0, 0, 150]  # Semi-transparent red for intersecting cells
            scene.add_geometry(cell_mesh, node_name=f'Cell_{cell_id}_valid')
        else:
            # Non-intersecting and outside the threshold
            cell_mesh.visual.face_colors = [200, 200, 200, 50]  # Grey and less opaque
            scene.add_geometry(cell_mesh, node_name=f'Cell_{cell_id}_invalid')

    print("Displaying the visualization of tissue and cells with distance threshold...")
    scene.show()


def run(mesh_path, tissue_path, features_path, distance_threshold=30.0, verbose=0):
    tissue_mesh = load_mesh(tissue_path)
    merged_mesh = load_mesh(mesh_path)

    face_cell_ids = extract_cell_faces(merged_mesh)
    cell_faces_dict = group_faces_by_cell_id(face_cell_ids)
    individual_cells = create_individual_cell_meshes(merged_mesh, cell_faces_dict)

    if verbose:
        print(f"Found {c.BOLD}{len(individual_cells)}{c.ENDC} individual cell meshes.")
        print(f"{c.OKBLUE}Performing intersection checks{c.ENDC}...")

    intersecting_cell_ids, non_intersecting_cell_ids = perform_intersection_checks_with_distance(
        tissue_mesh, individual_cells, distance_threshold=distance_threshold
    )

    if verbose:
        print(f"{c.OKBLUE}Updating CSV{c.ENDC}...")

    update_csv(features_path, non_intersecting_cell_ids, verbose)

    if verbose:
        print(f"{c.OKBLUE}Merging intersecting cells{c.ENDC}...")

    _ = merge_and_save_mesh(individual_cells, intersecting_cell_ids, mesh_path.replace('.ply', '_filtered.ply'))

    if verbose:
        total_cells = len(individual_cells)
        intersecting_cells = len(intersecting_cell_ids)
        non_intersecting_cells = len(non_intersecting_cell_ids)
        percentage = (intersecting_cells / total_cells) * 100 if total_cells > 0 else 0

        print(f'{c.OKGREEN}Summary{c.ENDC}')
        print(f'\tTotal Cells: {total_cells}')
        print(f'\tIntersecting Cells: {intersecting_cells} ({percentage:.2f}%)')
        print(f'\tNon-intersecting Cells: {non_intersecting_cells}')

    return tissue_mesh, individual_cells, intersecting_cell_ids

