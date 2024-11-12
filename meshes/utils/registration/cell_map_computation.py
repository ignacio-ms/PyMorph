import json
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trimesh
import pymeshlab
import open3d as o3d

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import find_specimen, HtDataset, find_group
from auxiliary import values as v

from meshes.utils.features.operator import (
    build_face_adjacency_csr_matrix,
    get_centroid,
    get_neighborhood_points
)

import os


class CellTissueMap:
    def __init__(self, spec, tissue='myocardium', verbose=1):
        self.specimen = spec
        self.group = find_group(spec)
        self.tissue = tissue
        self.verbose = verbose

        ds = HtDataset()
        # self.nuclei_mesh_path = ds.get_mesh_cell(spec, 'Nuclei', tissue, verbose=verbose, filtered=True)
        self.mem_mesh_path = ds.get_mesh_cell(spec, 'Membrane', tissue, verbose=verbose, filtered=True)
        self.tissue_path = ds.get_mesh_tissue(spec, tissue, verbose=verbose)
        self.atlas_path = v.data_path + f'ATLAS/{tissue}/ATLAS_{self.group}.ply'
#         self.nuclei_features_path = ds.get_features(spec, 'Nuclei', tissue, verbose=verbose, only_path=True)
        self.mem_features_path = ds.get_features(spec, 'Membrane', tissue, verbose=verbose, only_path=True)

#         self.nuclei_mesh = trimesh.load(self.nuclei_mesh_path)
        self.mem_mesh = trimesh.load(self.mem_mesh_path)
        self.tissue_mesh = trimesh.load(self.tissue_path)
        self.atlas_mesh = trimesh.load(self.atlas_path)
#         self.nuclei_features = pd.read_csv(self.nuclei_features_path)
        self.mem_features = pd.read_csv(self.mem_features_path)

        self.mapping_path = v.data_path + f'{self.group}/3DShape/map/{self.specimen}_cell_map.csv'

        aux = self.mapping_path.split('/')[:-1]
        if not os.path.exists('/'.join(aux)):
            os.makedirs('/'.join(aux), exist_ok=True)

    def init_vars(self, type='Membrane'):
        if type == 'Membrane':
            self.cell_mesh = self.mem_mesh
            self.cell_features = self.mem_features
        elif type == 'Nuclei':
            self.cell_mesh = self.nuclei_mesh
            self.cell_features = self.nuclei_features

        self.face_cell_ids = self.cell_mesh.metadata['_ply_raw']['face']['data']['cell_id']
        self.vertex_cell_ids = self.cell_mesh.metadata['_ply_raw']['vertex']['data']['cell_id']
        self.cell_ids = np.unique(self.face_cell_ids)

        assert self.face_cell_ids is not None and self.vertex_cell_ids is not None, 'Cell IDs not found in the mesh metadata'

        self.tissue_face_centroids = self.tissue_mesh.triangles_center
        self.tissue_face_tree = cKDTree(self.tissue_face_centroids)
        self.tissue_vertices_tree = cKDTree(self.tissue_mesh.vertices)

        self.tissue_graph = build_face_adjacency_csr_matrix(self.tissue_mesh)

        cell_centroids = {}

        for cell_id in self.cell_ids:
            cell_centroids[cell_id], _ = get_centroid(
                self.cell_mesh, cell_id,
                self.face_cell_ids, self.vertex_cell_ids
            )
        self.cell_centroids = cell_centroids

    def map_cells(self, type='Membrane'):
        assert type in ['Membrane', 'Nuclei'], f'Invalid cell type: {type}'
        self.init_vars(type)

        # Remove cell_ids without features
        self.cell_ids = np.intersect1d(self.cell_ids, self.cell_features['cell_id'].values)
        self.cell_centroids = {k: i for k, i in self.cell_centroids.items() if k in self.cell_ids}

        # Build a KDTree for the cell centroids
        centroid_array = np.array(list(self.cell_centroids.values()))
        cell_tree = cKDTree(centroid_array)

        # Get the closest cell for each tissue face
        _, closest_cell_ids = cell_tree.query(self.tissue_face_centroids, k=1)
        tissue_face_cell_ids = self.cell_ids[closest_cell_ids]

        assert len(tissue_face_cell_ids) == len(self.tissue_mesh.faces), 'Invalid cell map'
        mapping = pd.DataFrame({
            'tissue_face_id': np.arange(len(self.tissue_mesh.faces)),
            f'cell_id_{type}': tissue_face_cell_ids
        })

        mapping.to_csv(self.mapping_path, index=False)
        print(f'{c.OKGREEN}Cell map saved to:{c.ENDC} {self.mapping_path}')
        self.mapping = mapping

    def get_neighborhood(self, radius=34.0):
        assert self.mapping is not None, 'Cell map not found'
        face_neighbors = {}

        for face_idx in np.arange(len(self.tissue_mesh.faces)):
            dist = dijkstra(
                csgraph=self.tissue_graph, directed=False,
                indices=face_idx, return_predecessors=True,
                min_only=True
            )[0]
            face_neighbors[face_idx] = np.where(dist <= radius)[0]

        assert len(face_neighbors) == len(self.tissue_mesh.faces), 'Invalid neighborhood'
        new_cols = pd.DataFrame({
            'tissue_face_id': np.array(list(face_neighbors.keys()), dtype=int),
            'tissue_neighbors': face_neighbors.values()
        })

        # Ensure tissue_face_id has same type in both dataframes
        self.mapping['tissue_face_id'] = self.mapping['tissue_face_id'].astype(int)

        self.mapping = pd.merge(self.mapping, new_cols, on='tissue_face_id', how='left')
        self.mapping.to_csv(self.mapping_path, index=False)
        print(f'{c.OKGREEN}Neighborhood saved to:{c.ENDC} {self.mapping_path}')

    def color_mesh(self, feature_name, type='Membrane', cmap='seismic', normalize=False, smooth=False):
        if not hasattr(self, 'mapping'):
            if os.path.exists(self.mapping_path):
                self.mapping = pd.read_csv(self.mapping_path)

        cmap = plt.get_cmap(cmap)

        assert self.mapping is not None, 'Cell map not found'
        assert self.mapping.columns.isin(['tissue_face_id', f'cell_id_{type}', 'tissue_neighbors']).all(), 'Invalid mapping file'
        assert feature_name in self.cell_features.columns, f'Feature not found: {feature_name}'
        self.init_vars(type)

        feature_map = self.cell_features.set_index('cell_id')[feature_name].to_dict()
        face_values = self.mapping[f'cell_id_{type}'].map(feature_map)

        # Neighbor averaging
        aux_face_values = face_values.copy()
        for i, row in self.mapping.iterrows():
            if row['tissue_neighbors'] is not None:
                aux_face_values[i] = np.mean(face_values[row['tissue_neighbors']])

        face_values = aux_face_values

        if smooth:
            from scipy.ndimage import gaussian_filter1d
            face_values = gaussian_filter1d(face_values, sigma=1)

        if normalize:
            f_min = face_values.min()
            f_max = face_values.max()
            face_values = (face_values - f_min) / (f_max - f_min + 1e-9)

        face_colors = cmap(face_values)[:, :3]  # Remove alpha channel
        # face_colors = (face_colors * 255).astype(np.uint8)

        self.tissue_mesh.visual.face_colors = face_colors
        return self.tissue_mesh
