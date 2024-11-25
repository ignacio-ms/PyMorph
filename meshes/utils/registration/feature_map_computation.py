import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trimesh

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import find_specimen, HtDataset, find_group
from auxiliary import values as v
from auxiliary.utils.timer import LoadingBar

from meshes.utils.features.operator import (
    build_face_adjacency_csr_matrix,
    get_centroid,
    get_neighborhood_points
)

import os


class FeatureMap:
    def __init__(
            self, group, specimens, feature_maps, feature,
            tissue='myocardium', level='Membrane',
            verbose=1
    ):
        self.group = group
        self.specimens = specimens
        self.feature_maps = feature_maps
        self.feature = feature
        self.tissue = tissue
        self.level = level
        self.verbose = verbose

        self.atlas_path = v.data_path + f'ATLAS/{tissue}/ATLAS_{group}.ply'
        self.atlas = trimesh.load(self.atlas_path)

        for s, feature_map in zip(specimens, feature_maps):
            assert len(self.atlas.vertices) == len(feature_map.vertices), f'{c.FAIL}Feature map does not match atlas {s}{c.ENDC}'

        self.vertex_maps = [
            self.load_map_file(
                v.data_path + f'{group}/3DShape/Tissue/{tissue}/map/{specimen}/map/latest.map'
            )
            for specimen in specimens
        ]

    @staticmethod
    def load_map_file(map_file_path):
        def hex2float(hex_str):
            hex_str = hex_str.lstrip()
            return float.fromhex(hex_str)

        vertex_map = {}
        with open(map_file_path, 'r') as file:
            for line in file:
                if line.startswith('p'):
                    parts = line.strip().split()

                    source_vertex = int(parts[1])
                    target_vertex_c = int(parts[2])
                    target_vertex_a = int(parts[3])
                    alpha = hex2float(parts[4])
                    beta = hex2float(parts[5])

                    # Calculate gamma as the remaining barycentric coordinate
                    gamma = 1.0 - alpha - beta
                    barycentric_coords = [alpha, beta, gamma]

                    vertex_map[source_vertex] = ([target_vertex_c, target_vertex_a], barycentric_coords)
        return vertex_map

    @staticmethod
    def find_third_vertex(target_mesh, vertex_c, vertex_a):
        for face in target_mesh.faces:
            if vertex_c in face and vertex_a in face:
                return [v for v in face if v != vertex_c and v != vertex_a][0]
        return None

    @staticmethod
    def assign_colors_to_atlas(atlas, feature_map, vertex_map):
        atlas_colors = np.zeros((len(atlas.vertices), 4))

        feature_map_vertex_colors = trimesh.visual.color.face_to_vertex_color(
            feature_map,
            feature_map.visual.face_colors
        )
        feature_map.visual.vertex_colors = feature_map_vertex_colors

        for atlas_vertex, (triangle_vertices_fm, bary_coords) in vertex_map.items():
            v_c, v_a = triangle_vertices_fm
            # v_b = find_third_vertex(target_mesh, v_c, v_a)
            v_b = triangle_vertices_fm[1]
            if v_b is None:
                continue

            triangle_vertices_fm = [v_c, v_a, v_b]

            if any(v >= len(feature_map.vertices) for v in triangle_vertices_fm):
                continue

            if atlas_vertex >= len(atlas.vertices):
                continue

            colors_B = feature_map.visual.vertex_colors[triangle_vertices_fm]
            color_A = np.dot(bary_coords, colors_B)
            atlas_colors[atlas_vertex] = color_A

        uncolored_vertex = np.where(np.all(atlas_colors == 0, axis=1))[0]
        for vertex in uncolored_vertex:
            neighbours = atlas.vertex_neighbors[vertex]
            neighbour_colors = np.array([
                atlas_colors[n] for n in neighbours if n < len(atlas_colors) and not np.all(atlas_colors[n] == 0)
            ])

            if len(neighbour_colors) > 0:
                atlas_colors[vertex] = np.mean(neighbour_colors, axis=0)

        return atlas_colors

    def avergage_feature_maps(self, out_path):
        if out_path is None:
            out_path = v.data_path + f'{self.group}/3DShape/Tissue/{self.tissue}/map/latest.map'

        bar = LoadingBar(len(self.specimens))

        face_colors = []
        for s, vertex_map, feature_map in zip(
                self.specimens, self.vertex_maps, self.feature_maps
        ):
            try:
                print(f'{c.OKGREEN}Specimen{c.ENDC}: {s} ({self.group})')

                face_colors.append(self.assign_colors_to_atlas(
                    self.atlas, feature_map, vertex_map
                ))
            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC}: {e}')
                import traceback
                traceback.print_exc()

            bar.update()

        print(face_colors)
        self.atlas.visual.vertex_colors = np.mean(face_colors, axis=0).astype(np.uint8)
        print(np.mean(face_colors, axis=0).astype(np.uint8))
        self.atlas.export(out_path)
        self.atlas.export(out_path.replace('.ply', '.obj'), file_type='obj')
        print(f'{c.OKGREEN}Atlas saved{c.ENDC}: {out_path}')

