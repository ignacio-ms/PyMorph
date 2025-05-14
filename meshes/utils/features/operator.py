import numpy as np
from sklearn.decomposition import PCA

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# -----------------------------------------------------------------------------
# Helper guards
# -----------------------------------------------------------------------------
_MIN_PTS_ELLIPSOID = 4  # at least 4 non‑coplanar points needed for MVEE


def get_centroid(mesh, cell_id, face_cell_ids, vertex_cell_ids):
    cell_face_idx = np.where(face_cell_ids == cell_id)[0]
    cell_vertex_idx = np.where(vertex_cell_ids == cell_id)[0]

    if len(cell_face_idx) == 0 or len(cell_vertex_idx) == 0:
        raise ValueError(f"Cell ID {cell_id} not found in the mesh")

    face_vertices = mesh.vertices[mesh.faces[cell_face_idx]].reshape(-1, 3)
    cell_vertices = mesh.vertices[cell_vertex_idx]

    return face_vertices.mean(axis=0), cell_vertices


def find_closest_face(centroid, tissue_face_tree):
    dist, face_idx = tissue_face_tree.query(centroid)
    return face_idx


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------

def build_face_adjacency_csr_matrix(mesh):
    num_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency

    row, col, data = [], [], []
    centroids = mesh.triangles_center

    for f1, f2 in adjacency:
        edge_len = np.linalg.norm(centroids[f1] - centroids[f2])
        row.extend([f1, f2])
        col.extend([f2, f1])
        data.extend([edge_len, edge_len])

    return csr_matrix((data, (row, col)), shape=(num_faces, num_faces))


def get_neighborhood_points(mesh, face_idx, graph=None, radius=10.0):
    if graph is None:
        graph = build_face_adjacency_csr_matrix(mesh)

    for _ in range(26):  # safeguard loop
        dist = dijkstra(
            csgraph=graph.copy(), directed=False,
            indices=face_idx, return_predecessors=False, min_only=True
        )
        faces = np.where(dist <= radius)[0]
        pts = mesh.triangles_center[faces]
        if len(pts) >= 3:
            return pts, mesh.triangles_center[face_idx]
        radius += 1.0
    raise ValueError("Could not find enough neighborhood points")


# -----------------------------------------------------------------------------
# Geometry primitives
# -----------------------------------------------------------------------------

def fit_plane(points):
    """Return the normal vector of a best‑fit plane.

    Accepts points of shape (N, 3) or a flat length‑3 array.  Automatically
    reshapes and guards against degenerate inputs.
    """
    points = np.asarray(points)

    # Promote 1‑D [x, y, z] → (1, 3)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.shape[0] < 3:
        # Degenerate: cannot fit a unique plane. Return z‑axis as sentinel.
        return np.array([0.0, 0.0, 1.0])

    pca = PCA(n_components=3).fit(points)
    return pca.components_[2]  # normal vector of the plane


def approximate_ellipsoid(points, method="mvee", scaling_factor=2.0, tol=1e-5):
    """Approximate a bounding ellipsoid.

    Returns (center, axes, lengths) or (None, None, None) on failure.
    If *points* has fewer than 4 unique samples we skip the MVEE step and return
    a degenerate sphere so that downstream code can continue gracefully.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < _MIN_PTS_ELLIPSOID:
        # Fallback: tiny sphere
        ctr = points.mean(axis=0) if points.ndim == 2 else points
        return ctr, np.eye(3), np.zeros(3)

    try:
        if method == "mvee":
            N, d = points.shape
            Q = np.column_stack((points, np.ones(N))).T
            u = np.ones(N) / N
            err = tol + 1.0
            while err > tol:
                X = Q @ np.diag(u) @ Q.T
                M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
                j = np.argmax(M)
                step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
                new_u = (1 - step) * u
                new_u[j] += step
                err = np.linalg.norm(new_u - u)
                u = new_u
            center = u @ points
            cov = (points - center).T @ ((points - center) * u[:, None])
            _, s, rot = np.linalg.svd(cov)
            return center, rot, np.sqrt(s)

        elif method == "pca":
            centered = points - points.mean(axis=0)
            pca = PCA(n_components=3).fit(centered)
            axes = pca.components_
            lengths = scaling_factor * np.sqrt(pca.explained_variance_)
            return points.mean(axis=0), axes, lengths

    except Exception as e:
        print(f"approximate_ellipsoid error: {e}")
    return None, None, None


def get_longest_axis(axes, lengths):
    idx = int(np.argmax(lengths))
    v = axes[idx]
    return v / np.linalg.norm(v)


def get_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def compute_geodesic_distance(face_idx_start, face_idx_end, graph=None):
    dist, pred = dijkstra(csgraph=graph, directed=False, indices=face_idx_start, return_predecessors=True)
    d = dist[face_idx_end]
    if np.isinf(d):
        return None, None
    path = []
    cur = face_idx_end
    while cur != face_idx_start and cur != -9999:
        path.append(cur)
        cur = pred[cur]
    path.append(face_idx_start)
    path.reverse()
    return d, path
