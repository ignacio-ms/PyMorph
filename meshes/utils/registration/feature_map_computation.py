import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trimesh
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

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

        self.face_graph = build_face_adjacency_csr_matrix(self.atlas)

        for s, feature_map in zip(specimens, feature_maps):
            assert len(self.atlas.vertices) == len(feature_map.vertices), f'{c.FAIL}Feature map does not match atlas {s}{c.ENDC}'

        self.vertex_maps = [
            self.load_map_file(
                v.data_path + f'{group}/3DShape/Tissue/{tissue}/map/{specimen}/map/latest.map'
            )
            for specimen in specimens
        ]

        self.cell_mappings = [
            pd.read_csv(
                v.data_path + f'{group}/3DShape/Tissue/{tissue}/cell_map/{specimen}_cell_map.csv'
            )
            for specimen in specimens
        ]

        ds = HtDataset()
        self.features = [
            ds.get_features(specimen, level, tissue, filtered=True)
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

    @staticmethod
    def find_face(target_mesh, vertex_c, vertex_a):
        for idx, face in enumerate(target_mesh.faces):
            if vertex_c in face and vertex_a in face:
                return idx
        return None

    @staticmethod
    def precompute_edge_to_face_mapping(mesh):
        """Precompute edge-to-face mapping for efficient lookups."""
        edge_to_face = defaultdict(list)
        for face_idx, face in enumerate(mesh.faces):
            edges = [
                tuple(sorted((face[i], face[(i + 1) % 3])))
                for i in range(3)
            ]
            for edge in edges:
                edge_to_face[edge].append(face_idx)
        return edge_to_face

    @staticmethod
    def find_face_with_edge(edge_to_face, vertex_c, vertex_a):
        """Efficiently find a face using the edge-to-face mapping."""
        edge = tuple(sorted((vertex_c, vertex_a)))
        faces = edge_to_face.get(edge, [])
        return faces[0] if faces else None

    def assign_features_to_atlas(self, atlas, source, face_values, vertex_map, edge_to_face):
        """Map scalar values to atlas vertices in parallel."""
        atlas_values = np.zeros(len(atlas.vertices))

        def process_vertex(args):
            atlas_vertex, (triangle_vertices_src, _) = args
            if atlas_vertex >= len(atlas.vertices):
                return atlas_vertex, 0

            # Locate the face index from the source mesh
            face_idx = self.find_face_with_edge(edge_to_face, *triangle_vertices_src)

            if face_idx is None or face_idx >= len(face_values):
                return atlas_vertex, 0

            # Assign face value directly (no interpolation needed)
            face_value = face_values[face_idx]
            return atlas_vertex, face_value

        # Parallel processing of vertex mappings
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                process_vertex,
                vertex_map.items()
            )

        # Assign results to the atlas_values array
        for atlas_vertex, value in results:
            if atlas_vertex >= len(atlas_values):
                continue
            atlas_values[atlas_vertex] = value

        # Handle unmapped atlas vertices by averaging values from neighbors
        unassigned_vertices = np.where(atlas_values == 0)[0]
        for vertex in unassigned_vertices:
            neighbours = atlas.vertex_neighbors[vertex]
            neighbour_values = [
                atlas_values[n] for n in neighbours
                if n < len(atlas_values) and atlas_values[n] != 0
            ]

            if len(neighbour_values) > 0:
                atlas_values[vertex] = np.mean(neighbour_values)

        return atlas_values

    def color_atlas(self, out_path, cmap=None, type='Membrane', out_path_aux=None):
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        bar = LoadingBar(len(self.specimens))

        face_values = []
        for feat, cell_map, vertex_map, feature_map in zip(
                self.features, self.cell_mappings, self.vertex_maps, self.feature_maps
        ):
            try:
                # Precompute edge-to-face mapping
                edge_to_face = self.precompute_edge_to_face_mapping(feature_map)

                features = feat.set_index('cell_id')[self.feature].to_dict()
                face_val_s = cell_map[f'cell_id_{self.level}'].map(features)

                aux_face_val = face_val_s.copy()
                for i, row in cell_map.iterrows():
                    if row[f'tissue_neighbors_{type}'] is not None:
                        try:
                            neigh = np.array(row[f'tissue_neighbors_{type}'])

                            neighbour_vals = [face_val_s[int(n)] for n in neigh if not np.isnan(face_val_s[int(n)])]
                            if neighbour_vals:
                                aux_face_val[i] = np.mean(neighbour_vals)
                        except Exception as e:
                            neigh = row[f'tissue_neighbors_{type}'].replace('[', '').replace(']', '').split()
                            neigh = np.array(neigh)
                            neighbour_vals = [face_val_s[int(n)] for n in neigh if not np.isnan(face_val_s[int(n)])]
                            if neighbour_vals:
                                aux_face_val[i] = np.mean(neighbour_vals)

                face_values.append(
                    self.assign_features_to_atlas(
                        self.atlas, feature_map, aux_face_val, vertex_map, edge_to_face
                    )
                )
            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC}: {e}')
                import traceback
                traceback.print_exc()

            bar.update()

        # Aggregate the face values across all feature maps
        atlas_values = np.mean(face_values, axis=0)
        values_df = pd.DataFrame({
            'tissue_face_id': np.arange(len(atlas_values)),
            'value': atlas_values
        })

        # Check if .csv file exists
        aux_path = out_path.replace('.ply', '.csv')
        if not os.path.isfile(aux_path):
            if not os.path.exists('/'.join(aux_path.split('/')[:-1])):
                os.makedirs('/'.join(aux_path.split('/')[:-1]))
            values_df.to_csv(aux_path, index=False)

        if cmap is None:
            colors = [
                (0, 0, 1),  # Pure blue
                (0, 0.5, 1),  # Cyan-like
                (0, 1, 0),  # Green
                (1, 1, 0),  # Yellow
                (1, 0, 0),  # Red
            ]
            cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=1024)

        norm = BoundaryNorm(
            boundaries=np.linspace(
                atlas_values.min(), atlas_values.max(),
                cmap.N
            ), ncolors=cmap.N
        )
        atlas_colors = cmap(norm(atlas_values))

        # Save auxiliary files for quality assessment
        for s, face_val in zip(self.specimens, face_values):
            norm = BoundaryNorm(
                boundaries=np.linspace(
                    face_val.min(), face_val.max(),
                    cmap.N
                ), ncolors=cmap.N
            )

            self.atlas.visual.vertex_colors = cmap(norm(face_val))
            self.atlas.export(out_path.replace('.ply', f'_{s}.ply'))

        bar.update()
        bar.end()

        self.atlas.visual.vertex_colors = atlas_colors
        self.atlas.export(out_path)

        if out_path_aux is not None:
            split_path = out_path_aux.split('/')

            aux_folder_level = '/'.join(split_path[:-2])
            if not os.path.exists(aux_folder_level):
                os.makedirs(aux_folder_level)

            aux_folder_feature = '/'.join(split_path[:-1])
            if not os.path.exists(aux_folder_feature):
                os.makedirs(aux_folder_feature)

            self.atlas.export(out_path_aux)
            values_df.to_csv(out_path_aux.replace('.ply', '.csv'), index=False)

        self.export_obj(out_path)

        atlas_values_df = pd.DataFrame({
            'tissue_face_id': np.arange(len(atlas_values)),
            'value': atlas_values
        })
        atlas_values_df.to_csv(out_path.replace('.ply', '.csv'), index=False)

        print(f'{c.OKGREEN}Atlas saved{c.ENDC}: {out_path}')

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

        self.atlas.visual.vertex_colors = np.mean(face_colors, axis=0).astype(np.uint8)
        self.atlas.export(out_path)

        self.export_obj(out_path)
        print(f'{c.OKGREEN}Atlas saved{c.ENDC}: {out_path}')

    def export_obj(self, out_path):
        min_coords = np.min(self.atlas.vertices, axis=0)
        max_coords = np.max(self.atlas.vertices, axis=0)
        scale = max_coords - min_coords

        scale[scale == 0] = 1.0
        uv = (self.atlas.vertices[:, :2] - min_coords[:2]) / scale[:2]
        self.atlas.visual.uv = uv

        from PIL import Image
        import matplotlib.tri as mtri
        from skimage.draw import polygon

        texture_size = 1024
        texture_image = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
        uv = self.atlas.visual.uv.copy()
        uv *= (texture_size - 1)
        uv = uv.astype(np.int32)
        tri = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=self.atlas.faces)

        for i, triangle in enumerate(tri.triangles):
            idx0, idx1, idx2 = triangle
            uvs = uv[[idx0, idx1, idx2]]
            colors = self.atlas.visual.vertex_colors[[idx0, idx1, idx2], :]
            rr, cc = polygon(uvs[:, 1], uvs[:, 0], shape=(texture_size, texture_size))
            average_color = colors.mean(axis=0)
            texture_image[rr, cc] = average_color.astype(np.uint8)

        # Save the texture image
        texture_filename = out_path.replace('.ply', '.png')
        image = Image.fromarray(texture_image)  # .save(texture_filename, format='png')

        material = trimesh.visual.material.SimpleMaterial(
            image=image  # Image.open(texture_filename)
        )
        self.atlas.visual.material = material

        obj, texture = trimesh.exchange.obj.export_obj(
            self.atlas,
            return_texture=True,
            include_texture=True,
            write_texture=True,
            mtl_name=out_path.replace('.ply', '.mtl').split('/')[-1],
            resolver=trimesh.visual.resolvers.FilePathResolver(
                out_path.replace('.ply', '.mtl')
            )
        )

        with open(out_path.replace('.ply', '.obj'), 'w') as file:
            file.write(obj)
