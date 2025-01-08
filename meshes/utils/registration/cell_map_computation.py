import json
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trimesh
import pymeshlab
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

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

        self.ds = HtDataset()
        self.tissue_path = self.ds.get_mesh_tissue(spec, tissue, verbose=verbose)
        self.tissue_mesh = trimesh.load(self.tissue_path)

        self.mapping_path = v.data_path + f'{self.group}/3DShape/Tissue/{tissue}/cell_map/{self.specimen}_cell_map.csv'
        try:
            with open(self.mapping_path, 'r') as f:
                self.mapping = pd.read_csv(f)
        except FileNotFoundError:
            self.mapping = None

        aux = self.mapping_path.split('/')[:-1]
        if not os.path.exists('/'.join(aux)):
            os.makedirs('/'.join(aux), exist_ok=True)

    def init_vars(self, type='Membrane'):
        try:
            cell_mesh_path = self.ds.get_mesh_cell(self.specimen, type, self.tissue, verbose=self.verbose, filtered=True)
            self.cell_mesh = trimesh.load(cell_mesh_path)

            self.cell_features = self.ds.get_features(self.specimen, type, self.tissue, verbose=self.verbose, filtered=True)
        except FileNotFoundError:
            print(f'{c.FAIL}Error - {self.specimen}:{c.ENDC} Cells or features not found')

        try:
            self.face_cell_ids = self.cell_mesh.metadata['_ply_raw']['face']['data']['cell_id']
            self.vertex_cell_ids = self.cell_mesh.metadata['_ply_raw']['vertex']['data']['cell_id']
            self.cell_ids = np.unique(self.face_cell_ids)
        except KeyError:
            print(f'{c.FAIL}Error - {self.specimen}:{c.ENDC} Cell IDs not found in the mesh metadata')

        assert self.face_cell_ids is not None and self.vertex_cell_ids is not None, 'Cell IDs not found in the mesh metadata'

        self.tissue_face_centroids = self.tissue_mesh.triangles_center
        self.tissue_face_tree = cKDTree(self.tissue_face_centroids)
        self.tissue_vertices_tree = cKDTree(self.tissue_mesh.vertices)

        self.tissue_graph = build_face_adjacency_csr_matrix(self.tissue_mesh)

        cell_centroids = {}

        for cell_id in self.cell_ids:
            try:
                cell_centroids[cell_id], _ = get_centroid(
                    self.cell_mesh, cell_id,
                    self.face_cell_ids, self.vertex_cell_ids
                )
            except Exception as e:
                print(f'{c.WARNING}Error in cell{c.ENDC}: {cell_id} - {e}')
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

        if self.mapping is not None:
            if f'cell_id_{type}' in self.mapping.columns:
                # Remove previous cell_id column
                self.mapping.drop(columns=[f'cell_id_{type}'], inplace=True)

            mapping = pd.merge(
                self.mapping, mapping,
                on='tissue_face_id', how='left'
            )

        mapping.to_csv(self.mapping_path, index=False)
        print(f'{c.OKGREEN}Cell map saved to:{c.ENDC} {self.mapping_path}')
        self.mapping = mapping

    def get_neighborhood(self, radius=34.0, type='Membrane'):
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
            f'tissue_neighbors_{type}': face_neighbors.values()
        })

        # Ensure tissue_face_id has same type in both dataframes
        self.mapping['tissue_face_id'] = self.mapping['tissue_face_id'].astype(int)

        if f'tissue_neighbors_{type}' in self.mapping.columns:
            self.mapping.drop(columns=[f'tissue_neighbors_{type}'], inplace=True)

        self.mapping = pd.merge(self.mapping, new_cols, on='tissue_face_id', how='left')
        self.mapping.to_csv(self.mapping_path, index=False)
        print(f'{c.OKGREEN}Neighborhood saved to:{c.ENDC} {self.mapping_path}')

    def color_mesh(self, feature_name, type='Membrane', cmap=None):
        if not hasattr(self, 'mapping'):
            if os.path.exists(self.mapping_path):
                self.mapping = pd.read_csv(self.mapping_path)

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        self.init_vars(type)
        assert self.mapping is not None, 'Cell map not found'
        assert feature_name in self.cell_features.columns, f'Feature not found: {feature_name}'

        feature_map = self.cell_features.set_index('cell_id')[feature_name].to_dict()
        face_values = self.mapping[f'cell_id_{type}'].map(feature_map)

        if feature_name == 'cell_division':
            face_values = [0 if v > 1 else 1 for v in face_values]

        # Neighbor averaging
        aux_face_values = face_values.copy()
        for i, row in self.mapping.iterrows():
            if row[f'tissue_neighbors_{type}'] is not None:
                try:
                    neigh = np.array(row[f'tissue_neighbors_{type}'])
                    neighbors = np.array(neigh)
                    neighbor_values = [face_values[int(n)] for n in neighbors if not np.isnan(face_values[int(n)])]
                    if neighbor_values:
                        aux_face_values[i] = np.mean(neighbor_values)
                except Exception as e:
                    neigh = row[f'tissue_neighbors_{type}'].replace('[', '').replace(']', '').split()
                    neighbors = np.array(neigh)
                    neighbor_values = [face_values[int(n)] for n in neighbors if not np.isnan(face_values[int(n)])]
                    if neighbor_values:
                        aux_face_values[i] = np.mean(neighbor_values)

        face_values = aux_face_values

        if cmap is None:
            colors = [
                (0, 0, 1),  # Pure blue
                (0, 0.5, 1),  # Cyan-like
                (0, 1, 0),  # Green
                (1, 1, 0),  # Yellow
                (1, 0, 0),  # Red
            ]

            cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=1024 if feature_name != 'cell_division' else 10)

        # Set min/max from percentile
        vmin = np.percentile(face_values, 1)
        vmax = np.percentile(face_values, 99)

        if feature_name == 'cell_division':
            vmin = np.min(face_values)
            vmax = np.max(face_values)
            print(f'{c.OKGREEN}Min{c.ENDC}: {vmin}')
            print(f'{c.OKGREEN}Max{c.ENDC}: {vmax}')

        norm = BoundaryNorm(
            boundaries=np.linspace(
                vmin, vmax,
                cmap.N
            ), ncolors=cmap.N
        )
        face_colors = cmap(norm(face_values))
        # face_colors = (face_colors * 255).astype(np.uint8)

        self.tissue_mesh.visual.face_colors = face_colors

        self.tissue_mesh.export(
            f'{"/".join(self.tissue_path.split("/")[:-1])}/map/{self.specimen}/{type}_{feature_name}.ply'
        )
        # self.tissue_mesh.export(
        #     f'{"/".join(self.tissue_path.split("/")[:-1])}/map/{self.specimen}/{type}_{feature_name}.obj',
        #     file_type='obj'
        # )

        face_values = pd.DataFrame({
            'tissue_face_id': np.arange(len(face_values)),
            'value': face_values
        })
        face_values.to_csv('/'.join(self.tissue_path.split('/')[:-1]) + f'/map/{self.specimen}/{type}_{feature_name}.csv', index=False)

        return self.tissue_mesh

    def normalize_features(self, values, feature_name, type='Membrane'):
        ds = HtDataset()

        all_features = []
        print(f'{c.OKGREEN}Normalizing feature values across embryos{c.ENDC}')
        count, missing = 0, 0
        for g in v.specimens:
            for s in v.specimens[g]:
                try:
                    features = ds.get_features(s, type, self.tissue, verbose=0)
                    all_features.extend(features[feature_name].values)
                    count += 1
                except Exception as e:
                    print(f'\t{c.WARNING}Warning{c.ENDC}: Embryo {s} - {e}')
                    missing += 1
                    continue

        print(f'{c.OKBLUE}Embryos averaged{c.ENDC}: {count} / {count + missing} ({missing} missing)')
        all_features = np.array(all_features, dtype=np.float64)
        all_features = all_features[~np.isnan(all_features)]
        np.sort(all_features)

        f_min = all_features.min()
        f_max = all_features.max()

        print(f'{c.OKGREEN}Min{c.ENDC}: {f_min}')
        print(f'{c.OKGREEN}Max{c.ENDC}: {f_max}')

        if f_min == f_max or np.isnan(f_min) or np.isnan(f_max):
            return values

        return values.map(lambda x: (x - f_min) / (f_max - f_min))
