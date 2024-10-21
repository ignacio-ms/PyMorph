from auxiliary import values as v

import trimesh
import numpy as np
from auxiliary.data.dataset_ht import find_group

import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io
from IPython.display import Image as IPyImage

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

import keyboard
import time
import scipy

import gdist
import rtree
import shapely
import pyglet
from pyglet.gl import *
import pyrender


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


def get_neighborhood_points(tissue_mesh, face_idx, radius=10.0):
    closest_face_centroid = tissue_mesh.triangles_center[face_idx]

    tissue_vertices_tree = cKDTree(tissue_mesh.vertices)
    distance, source_vertex = tissue_vertices_tree.query(closest_face_centroid)

    iters = 0
    while True:
        try:
            distances_matrix = gdist.local_gdist_matrix(
                vertices=tissue_mesh.vertices,
                triangles=tissue_mesh.faces.astype(np.int32),
                max_distance=radius
            )
        except Exception as e:
            print(f"Error computing geodesic distances with gdist: {e}")
            raise e

        row = distances_matrix.getrow(source_vertex).tocoo()
        neighborhood_vertex_indices = row.col[row.data <= radius]

        neighborhood_points = tissue_mesh.vertices[neighborhood_vertex_indices]

        if len(neighborhood_points) >= 3:
            return neighborhood_points, closest_face_centroid
        else:
            iters += 1
            radius += 1.0
            if iters > 10:
                raise ValueError('Could not find enough neighborhood points')


def fit_plane(points):
    assert len(points) >= 3, 'At least 3 points are required to fit a plane'

    pca = PCA(n_components=3).fit(points)
    normal = pca.components_[2]

    return normal


def approximate_ellipsoid(points, method='mvee', scaling_factor=2.0, tol=1e-5):
    from cvxopt import matrix, solvers

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