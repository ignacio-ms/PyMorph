import numpy as np
import pandas as pd
import trimesh
from joblib import Parallel, delayed
from auxiliary.utils.timer import LoadingBar
from meshes.utils.visualizer import CellVisualization
from scipy.spatial import cKDTree
from meshes.utils.operator import (
    get_centroid,
    find_closest_face,
    get_neighborhood_points,
    fit_plane,
    approximate_ellipsoid,
    get_longest_axis,
    get_angle
)

class MeshFeatureExtractor:
    def __init__(self, cell_mesh, tissue_mesh, features):
        self.cell_mesh = cell_mesh
        self.tissue_mesh = tissue_mesh
        self.features = features

        self.face_cell_ids = self.cell_mesh.metadata['_ply_raw']['face']['data']['cell_id']
        self.vertex_cell_ids = self.cell_mesh.metadata['_ply_raw']['vertex']['data']['cell_id']
        self.cell_ids = np.unique(self.face_cell_ids)

        assert self.face_cell_ids is not None and self.vertex_cell_ids is not None, 'Cell IDs not found in the mesh metadata'

        self.tissue_face_centroids = self.tissue_mesh.triangles_center
        self.tissue_face_tree = cKDTree(self.tissue_face_centroids)
        self.tissue_vertices_tree = cKDTree(self.tissue_mesh.vertices)

    def cell_perpendicularity(self, cell_id, display=False, dynamic_display=False):
        def deg2perpen(angle):
            return np.sin(np.deg2rad(angle))

        centroid, cell_vertices = get_centroid(
            self.cell_mesh, cell_id,
            self.face_cell_ids, self.vertex_cell_ids
        )

        cell_mesh = self.cell_mesh.submesh([
            np.where(self.face_cell_ids == cell_id)[0]
        ], append=True)

        if isinstance(cell_mesh, trimesh.Scene):
            cell_mesh = cell_mesh.dump(concatenate=True)

        closest_face = find_closest_face(centroid, self.tissue_face_tree)
        neigh_points = get_neighborhood_points(self.tissue_mesh, closest_face, radius=8.0)
        plane = fit_plane(neigh_points[0])
        ellipse_c, ellipse_axes, ellipse_lengths = approximate_ellipsoid(
            cell_vertices,
            method='mvee', tol=1e-4
        )
        longest_axis = get_longest_axis(ellipse_axes, ellipse_lengths)
        angle = get_angle(longest_axis, plane)

        if display:
            visualization = CellVisualization(
                tissue_mesh=self.tissue_mesh,
                cell_mesh=cell_mesh,
                centroid=centroid,
                dynamic_camera=False
            )

            visualization.add_closest_face_centroid(self.tissue_face_centroids[closest_face])
            visualization.add_neighborhood_points(neigh_points[0])
            visualization.add_fitted_plane(plane, neigh_points[1], size=40.0)
            visualization.add_longest_axis(centroid, longest_axis, length=20.0)
            visualization.add_plane_normal(
                neigh_points[1], plane, length=15.0,
                color=[1.0, 1.0, 0.0, 1.0]
            )

            visualization.render_scene(live=dynamic_display)

        return deg2perpen(angle)

    def cell_sphericity(self, cell_id, method='eigenvalues'):
        cell_mesh = self.cell_mesh.submesh([
            np.where(self.face_cell_ids == cell_id)[0]
        ], append=True)

        if isinstance(cell_mesh, trimesh.Scene):
            cell_mesh = cell_mesh.dump(concatenate=True)

        if method == 'eigenvalues':
            cov = np.cov(cell_mesh.vertices.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
            sphericity_value = (eigvals[2] / eigvals[0]) ** (1/2)
            sphericity_value = min(max(sphericity_value, 0.0), 1.0)

        elif method == 'volume':
            volume = cell_mesh.volume
            surface_area = cell_mesh.area

            if volume == 0 or surface_area == 0:
                return 0.0

            sphericity_value = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area
            sphericity_value = min(max(sphericity_value, 0.0), 1.0)

        else:
            raise ValueError(f'Invalid method: {method}')

        return sphericity_value

    def cell_columnarity(self, sphericity, perpendicularity):
        return (1 - sphericity) * perpendicularity

    def extract(self, n_jobs=-1):
        def compute_features(cell_id):
            perpendicularity = self.cell_perpendicularity(cell_id)
            sphericity = self.cell_sphericity(cell_id)
            columnarity = self.cell_columnarity(sphericity, perpendicularity)

            return {
                'cell_id': cell_id,
                'angle': perpendicularity,
                'perpendicularity': perpendicularity,
                'sphericity': sphericity,
                'columnarity': columnarity
            }

        # Use Parallel to compute features in parallel
        features_rows = Parallel(n_jobs=n_jobs)(
            delayed(compute_features)(cell_id) for cell_id in self.cell_ids
        )

        return pd.DataFrame(
            features_rows,
            columns=['cell_id', 'angle', 'perpendicularity', 'sphericity', 'columnarity']
        )
