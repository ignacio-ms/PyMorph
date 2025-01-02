import numpy as np
import trimesh
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


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
    valid_faces_mask = np.isin(face_cell_ids, intersecting_cell_ids)
    valid_face_indices = np.where(valid_faces_mask)[0]

    # 2) Identify the original vertices used by these faces
    used_vertices = np.unique(cell_mesh.faces[valid_face_indices].flatten())

    # 3) Reindex these vertices from [original IDs] -> [0..len(used_vertices)-1]
    #    A simple way is to build a "look-up" array that says, for each old vertex index,
    #    what is its new index in the submesh?
    old_to_new = np.full(len(cell_mesh.vertices), fill_value=-1, dtype=np.int64)
    old_to_new[used_vertices] = np.arange(len(used_vertices))

    # 4) Create the new faces, using the new vertex indices
    new_faces = old_to_new[cell_mesh.faces[valid_face_indices]]

    # 5) Gather the new vertices
    new_vertices = cell_mesh.vertices[used_vertices]

    # 6) Build the actual submesh
    filtered_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False  # Avoid reprocessing if you want to keep the structure stable
    )

    # -----------------------
    #  A) Preserve face_cell_ids
    # -----------------------
    new_face_cell_ids = face_cell_ids[valid_face_indices]

    # Make sure to create the same structure in metadata
    filtered_mesh.metadata['_ply_raw'] = {}
    filtered_mesh.metadata['_ply_raw']['face'] = {'data': {}}
    filtered_mesh.metadata['_ply_raw']['face']['data']['cell_id'] = new_face_cell_ids

    # -----------------------
    #  B) Preserve vertex_cell_ids (if they exist)
    # -----------------------
    if (
        '_ply_raw' in cell_mesh.metadata
        and 'vertex' in cell_mesh.metadata['_ply_raw']
        and 'data' in cell_mesh.metadata['_ply_raw']['vertex']
        and 'cell_id' in cell_mesh.metadata['_ply_raw']['vertex']['data']
    ):
        old_vertex_cell_ids = cell_mesh.metadata['_ply_raw']['vertex']['data']['cell_id']
        # Now pick the IDs for the used vertices
        new_vertex_cell_ids = old_vertex_cell_ids[used_vertices]

        # Store them in the new mesh metadata
        filtered_mesh.metadata['_ply_raw']['vertex'] = {'data': {}}
        filtered_mesh.metadata['_ply_raw']['vertex']['data']['cell_id'] = new_vertex_cell_ids

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


def save_filtered_mesh_to_ply(filtered_mesh, output_path):
    """
    Save a filtered mesh as a PLY file, preserving face/vertex cell IDs
    in the same structure used at creation time.
    """

    # ---------------------------------------------------------
    # 1) Gather geometry from the filtered Trimesh
    # ---------------------------------------------------------
    vertices = filtered_mesh.vertices
    faces = filtered_mesh.faces

    # If we have face cell IDs in metadata
    face_cell_ids = None
    if (
        '_ply_raw' in filtered_mesh.metadata
        and 'face' in filtered_mesh.metadata['_ply_raw']
        and 'data' in filtered_mesh.metadata['_ply_raw']['face']
        and 'cell_id' in filtered_mesh.metadata['_ply_raw']['face']['data']
    ):
        face_cell_ids = filtered_mesh.metadata['_ply_raw']['face']['data']['cell_id']
    else:
        # fallback to a default
        print("Warning: No face cell IDs found in metadata. Using zeros.")
        face_cell_ids = np.zeros(len(faces), dtype=np.int32)

    # If we have vertex cell IDs in metadata
    vertex_cell_ids = None
    if (
        '_ply_raw' in filtered_mesh.metadata
        and 'vertex' in filtered_mesh.metadata['_ply_raw']
        and 'data' in filtered_mesh.metadata['_ply_raw']['vertex']
        and 'cell_id' in filtered_mesh.metadata['_ply_raw']['vertex']['data']
    ):
        vertex_cell_ids = filtered_mesh.metadata['_ply_raw']['vertex']['data']['cell_id']
    else:
        print("Warning: No vertex cell IDs found in metadata. Using zeros.")
        vertex_cell_ids = np.zeros(len(vertices), dtype=np.int32)

    if filtered_mesh.vertex_normals is None or len(filtered_mesh.vertex_normals) == 0:
        filtered_mesh.rezero()
    normals = filtered_mesh.vertex_normals
    if normals is None or len(normals) != len(vertices):
        # fallback if normals are not available
        normals = np.zeros_like(vertices)

    # ---------------------------------------------------------
    # 2) Create the structured arrays for vertices and faces
    #    matching your original mesh creation approach
    # ---------------------------------------------------------
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('cell_id', 'i4')
    ]
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]
    vertex_data['cell_id'] = np.squeeze(vertex_cell_ids)

    face_dtype = [('vertex_indices', 'i4', (3,)), ('cell_id', 'i4')]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces
    face_data['cell_id'] = np.squeeze(face_cell_ids)

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # ---------------------------------------------------------
    # 3) Write out the PLY
    # ---------------------------------------------------------
    PlyData([vertex_element, face_element], text=True).write(output_path)
    print(f"Filtered mesh saved with cell_id metadata to: {output_path}")


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
    # visualize_tissue_and_filtered_cells(tissue_mesh, filtered_mesh)

    # Save the filtered cell mesh
    filtered_mesh_path = mesh_path.replace('.ply', '_filtered.ply')

    # filtered_mesh.export(filtered_mesh_path)
    save_filtered_mesh_to_ply(filtered_mesh, filtered_mesh_path)
    print(f"Filtered cell mesh saved to: {filtered_mesh_path}")
    return filtered_mesh, intersecting_cell_ids, non_intersecting_cell_ids