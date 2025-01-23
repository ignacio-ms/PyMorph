import numpy as np
from sklearn.decomposition import PCA

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def get_centroid(mesh, cell_id, face_cell_ids, vertex_cell_ids):
    cell_face_idx = np.where(face_cell_ids == cell_id)[0]
    cell_vertex_idx = np.where(vertex_cell_ids == cell_id)[0]

    if len(cell_face_idx) == 0 or len(cell_vertex_idx) == 0:
        raise ValueError(f'Cell ID {cell_id} not found in the mesh')

    cell_faces = mesh.faces[cell_face_idx]
    face_vertices = mesh.vertices[cell_faces].reshape(-1, 3)

    cell_vertices = mesh.vertices[cell_vertex_idx]

    return np.mean(face_vertices, axis=0), cell_vertices


def find_closest_face(centroid, tissue_face_tree):
    dist, dace_idx = tissue_face_tree.query(centroid)
    return dace_idx


def build_face_adjacency_csr_matrix(mesh):
    """
    Builds a CSR matrix representation of the face adjacency graph.
    """
    # Number of faces
    num_faces = len(mesh.faces)

    # Get face adjacency information
    adjacency = mesh.face_adjacency

    # Initialize data for CSR matrix
    row_indices = []
    col_indices = []
    data = []

    for idx, (face_idx_1, face_idx_2) in enumerate(adjacency):
        # # Edge weight: length of the shared edge
        # shared_edge = mesh.face_adjacency_edges[idx]
        # edge_vertices = mesh.vertices[shared_edge]
        # edge_length = np.linalg.norm(edge_vertices[0] - edge_vertices[1])

        # Egde weight: Euclidean distance between face centroids
        face_centroids = mesh.triangles_center
        edge_length = np.linalg.norm(face_centroids[face_idx_1] - face_centroids[face_idx_2])

        # Add entries for both directions
        row_indices.extend([face_idx_1, face_idx_2])
        col_indices.extend([face_idx_2, face_idx_1])
        data.extend([edge_length, edge_length])

    # Create CSR matrix
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_faces, num_faces))

    return adjacency_matrix


def get_neighborhood_points(mesh, face_idx, graph=None, radius=10.0):
    """
    Uses scipy.sparse.csgraph to find neighboring faces within a specified radius.
    """
    if graph is None:
        graph = build_face_adjacency_csr_matrix(mesh)

    iters = 0
    while True:
        # Compute shortest paths from the starting face
        distances = dijkstra(
            csgraph=graph.copy(), directed=False,
            indices=face_idx, return_predecessors=True,
            min_only=True,
            # limit=radius
        )[0]

        # Get faces within the radius
        faces_within_radius = np.where(distances <= radius)[0]

        # Collect the centroids of these faces
        neighborhood_points = mesh.triangles_center[faces_within_radius]

        # Get the centroid of the starting face
        closest_face_centroid = mesh.triangles_center[face_idx]

        if len(neighborhood_points) >= 3:
            return neighborhood_points, closest_face_centroid
        else:
            iters += 1
            radius += 1.0
            if iters > 25:
                raise ValueError('Could not find enough neighborhood points')


def fit_plane(points):
    assert len(points) >= 3, 'At least 3 points are required to fit a plane'

    pca = PCA(n_components=3).fit(points)
    normal = pca.components_[2]

    return normal


def approximate_ellipsoid(points, method='mvee', scaling_factor=2.0, tol=1e-5):
    # from cvxopt import matrix, solvers

    try:
        if method == 'mvee':
            N, d = points.shape
            Q = np.column_stack((points, np.ones(N))).T  # Shape: (d+1, N)

            err = tol + 1.0
            u = np.ones(N) / N  # Initialize uniform weights

            while err > tol:
                X = Q @ np.diag(u) @ Q.T  # Shape: (d+1, d+1)
                M = np.diag(Q.T @ np.linalg.inv(X) @ Q)  # Shape: (N,)
                j = np.argmax(M)
                maximum = M[j]
                step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
                new_u = (1 - step_size) * u
                new_u[j] += step_size
                err = step_size
                u = new_u

            center = u @ points  # Shape: (3,)
            cov = (points - center).T @ ((points - center) * u[:, np.newaxis])  # Shape: (3,3)
            U, s, rotation = np.linalg.svd(cov)

            lengths = np.sqrt(s)
            axes = rotation
            return center, axes, lengths

        elif method == 'pca':
            # Center the points
            centered_vertices = points - np.mean(points, axis=0)

            # Perform PCA
            pca = PCA(n_components=3).fit(centered_vertices)

            axes = pca.components_
            # Scale the axes lengths by the desired scaling factor
            lengths = scaling_factor * np.sqrt(pca.explained_variance_)

            center = np.mean(points, axis=0)

            return center, axes, lengths
    except Exception as e:
        print(f'Error: {e}')
        return None, None, None
    return None


def get_longest_axis(axes, lengths):
    longest_axis_idx = np.argmax(lengths)
    longest_axis_vector = axes[longest_axis_idx]

    # Unit vector
    longest_axis_vector /= np.linalg.norm(longest_axis_vector)
    return longest_axis_vector


def get_angle(v1, v2):
    # Normalize
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot_prod = np.clip(np.dot(v1, v2), -1.0, 1.0)

    alpha_rad = np.arccos(dot_prod)
    alpha_deg = np.degrees(alpha_rad)

    return alpha_deg


def compute_geodesic_distance(face_idx_start, face_idx_end, graph=None):
    """
    Computes the geodesic distance between two faces and returns the path.
    """

    # Use Dijkstra's algorithm to compute shortest paths from face_idx_start
    distances, predecessors = dijkstra(
        csgraph=graph,
        directed=False,
        indices=face_idx_start,
        return_predecessors=True
    )

    # Get the distance to the target face
    geodesic_distance = distances[face_idx_end]

    if np.isinf(geodesic_distance):
        print(f"No path found between face {face_idx_start} and face {face_idx_end}.")
        return None, None

    # Reconstruct the path from start to end
    path = []
    current_face = face_idx_end
    while current_face != face_idx_start and current_face != -9999:
        path.append(current_face)
        current_face = predecessors[current_face]

    if current_face == -9999:
        print(f"No path found between face {face_idx_start} and face {face_idx_end}.")
        return None, None

    path.append(face_idx_start)
    path.reverse()  # Reverse to get path from start to end

    return geodesic_distance, path
