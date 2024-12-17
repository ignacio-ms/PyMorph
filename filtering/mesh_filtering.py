import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(mesh_path):
    """Load a mesh from a file."""
    return trimesh.load(mesh_path)


def compute_face_centroids(mesh):
    """Compute centroids of all faces in a mesh."""
    return mesh.vertices[mesh.faces].mean(axis=1)


def compute_cell_centroids(cell_mesh, face_cell_ids):
    """
    Compute the centroid of each cell mesh.

    Parameters:
    - cell_mesh (trimesh.Trimesh): The merged cell mesh.
    - face_cell_ids (np.ndarray): Array mapping each face to a cell ID.

    Returns:
    - cell_centroids (dict): Mapping of cell ID to centroid coordinates.
    """
    cell_centroids = {}
    unique_cell_ids = np.unique(face_cell_ids)

    for cell_id in unique_cell_ids:
        # Get face indices for the current cell
        cell_faces = np.where(face_cell_ids == cell_id)[0]

        # Compute centroid of the cell
        face_vertices = cell_mesh.vertices[cell_mesh.faces[cell_faces]]
        cell_centroids[cell_id] = face_vertices.mean(axis=0)

    return cell_centroids


def associate_cells_with_tissue(cell_centroids, tissue_mesh, distance_threshold):
    """
    Identify cells that are within the proximity of the tissue mesh.

    Parameters:
    - cell_centroids (dict): Mapping of cell IDs to centroids.
    - tissue_mesh (trimesh.Trimesh): The tissue mesh.
    - distance_threshold (float): Threshold distance for proximity.

    Returns:
    - intersecting_cell_ids (list): Cell IDs within the threshold distance.
    - non_intersecting_cell_ids (list): Cell IDs outside the threshold distance.
    """
    # Compute face centroids for the tissue mesh
    tissue_face_centroids = compute_face_centroids(tissue_mesh)

    # Build KDTree for the tissue face centroids
    tissue_tree = cKDTree(tissue_face_centroids)

    intersecting_cell_ids = []
    non_intersecting_cell_ids = []

    for cell_id, centroid in cell_centroids.items():
        # Query the KDTree for the distance to the nearest tissue face centroid
        distance, _ = tissue_tree.query(centroid)

        distance = np.min(distance)
        # Check if the cell is within the threshold
        if distance <= distance_threshold:
            intersecting_cell_ids.append(cell_id)
        else:
            non_intersecting_cell_ids.append(cell_id)

    return intersecting_cell_ids, non_intersecting_cell_ids


def filter_cells(cell_mesh, face_cell_ids, intersecting_cell_ids):
    """
    Filter the cell mesh to retain only the cells that intersect with the tissue mesh.

    Parameters:
    - cell_mesh (trimesh.Trimesh): The merged cell mesh.
    - face_cell_ids (np.ndarray): Array mapping each face to a cell ID.
    - intersecting_cell_ids (list): List of intersecting cell IDs.

    Returns:
    - filtered_mesh (trimesh.Trimesh): Filtered mesh containing only intersecting cells.
    """
    # Create a mask for faces belonging to intersecting cells
    valid_faces_mask = np.isin(face_cell_ids, intersecting_cell_ids)

    # Apply the mask to get valid face indices
    valid_faces_indices = np.where(valid_faces_mask)[0]

    # Submesh to extract valid cells
    filtered_mesh = cell_mesh.submesh([valid_faces_indices], append=True)
    return filtered_mesh


def visualize_tissue_and_filtered_cells(tissue_mesh, filtered_cell_mesh):
    """
    Visualize the tissue and filtered cell mesh.

    Parameters:
    - tissue_mesh (trimesh.Trimesh): The tissue mesh.
    - filtered_cell_mesh (trimesh.Trimesh): The filtered cell mesh.
    """
    scene = trimesh.Scene()

    # Add tissue mesh with a distinct color
    tissue_mesh.visual.face_colors = [0, 255, 0, 100]  # Semi-transparent green
    scene.add_geometry(tissue_mesh)

    # Add filtered cell mesh with a different color
    filtered_cell_mesh.visual.face_colors = [0, 0, 255, 150]  # Semi-transparent blue
    scene.add_geometry(filtered_cell_mesh)

    print("Displaying the visualization...")
    scene.show()


def run(mesh_path, tissue_path, distance_threshold=30.0):
    """
    Main function to load meshes, perform filtering, and visualize results.

    Parameters:
    - mesh_path (str): Path to the merged cell mesh file.
    - tissue_path (str): Path to the tissue mesh file.
    - distance_threshold (float): Threshold distance for proximity checks.
    """
    # Load meshes
    print("Loading meshes...")
    tissue_mesh = load_mesh(tissue_path)
    cell_mesh = load_mesh(mesh_path)

    # Extract cell IDs from the face metadata
    try:
        face_cell_ids = cell_mesh.metadata['_ply_raw']['face']['data']['cell_id']
    except KeyError:
        raise ValueError("The cell mesh does not contain 'cell_id' metadata in the faces.")

    # Compute centroids for each cell
    print("Computing cell centroids...")
    cell_centroids = compute_cell_centroids(cell_mesh, face_cell_ids)

    # Identify intersecting cells
    print("Associating cells with tissue...")
    intersecting_cell_ids, non_intersecting_cell_ids = associate_cells_with_tissue(
        cell_centroids, tissue_mesh, distance_threshold
    )

    print(f"Intersecting Cells: {len(intersecting_cell_ids)}")
    print(f"Non-Intersecting Cells: {len(non_intersecting_cell_ids)}")

    # Filter the cell mesh
    print("Filtering cell mesh...")
    filtered_mesh = filter_cells(cell_mesh, face_cell_ids, intersecting_cell_ids)

    # Visualize the results
    visualize_tissue_and_filtered_cells(tissue_mesh, filtered_mesh)

    # Save the filtered cell mesh
    filtered_mesh_path = mesh_path.replace('.ply', '_filtered.ply')
    filtered_mesh.export(filtered_mesh_path)
    print(f"Filtered cell mesh saved to: {filtered_mesh_path}")
    return filtered_mesh, intersecting_cell_ids, non_intersecting_cell_ids