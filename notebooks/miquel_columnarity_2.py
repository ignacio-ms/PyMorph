import os
import sys
import numpy as np
import pandas as pd
import trimesh

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import torch

#is_gpu = torch.cuda.is_available()
is_gpu = False
print(f"GPU enabled: {is_gpu}")
if is_gpu:
    if not hasattr(np, "bool"):
        np.bool = np.bool_

    from pytorch3d.ops import knn_points
    import cudf, cugraph, cupy as cp
else:
    torch = None  # type: ignore


# Project‑local utils -----------------------------------------------------------
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

# ──────────────────────────────────────────────────────────────────────────────
# Optional GPU back‑end ---------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────────

from util.gpu.gpu_torch import current_device
from meshes.utils.features.operator import build_face_adjacency_csr_matrix, get_centroid
from meshes.utils.features.extractor import MeshFeatureExtractor
from util.misc.colors import bcolors as c


# -----------------------------------------------------------------------------
# I/O paths (unchanged)
# -----------------------------------------------------------------------------
base_dir = (
    "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/"
    "U_Bioinformatica/Morena/MeisdKO_WT_mTmG_Columnarity/results/"
) # MeisdKO_WT_mTmG_Columnarity
mesh_dir = os.path.join(base_dir, "mesh", "cells")
tissue_dir = os.path.join(base_dir, "mesh", "tissue")
skeleton_dir = os.path.join(base_dir, "mesh", "skeleton")
out_dir = os.path.join(base_dir, "columnarity")
features_dir = os.path.join(base_dir, "features")

PARALLELIZE = False  # joblib workers for CPU path only

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run_features(mesh_path, tissue_path, path_out, verbose=0, parallelize=False):
    cell_mesh = trimesh.load(mesh_path, file_type="ply")
    tissue_mesh = trimesh.load(tissue_path, file_type="ply")

    if verbose:
        print(f"{c.OKBLUE}Extracting features{c.ENDC} for {c.BOLD}{mesh_path}{c.ENDC}")

    extractor = MeshFeatureExtractor(cell_mesh, tissue_mesh)
    new_features = extractor.extract(n_jobs=3 if parallelize else 1)

    if verbose:
        print(f"{c.OKGREEN}Features extracted{c.ENDC} [{len(new_features)}]")

    new_features["cell_id"] = new_features["cell_id"].astype(int)
    new_features.to_csv(path_out, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# Mapping class (GPU‑aware)
# ──────────────────────────────────────────────────────────────────────────────

class CellTissueMap:
    def __init__(self, cells_path, tissue_path, features_path, mapping_path):
        self.tissue_mesh = trimesh.load(tissue_path, file_type="ply")
        self.cell_mesh = trimesh.load(cells_path, file_type="ply")
        self.features = pd.read_csv(features_path)
        self.mapping_path = mapping_path
        self.mapping = None

        aux = os.path.dirname(self.mapping_path)
        os.makedirs(aux, exist_ok=True)

        # Precompute CPU adjacency graph once
        self._tissue_graph_cpu = build_face_adjacency_csr_matrix(self.tissue_mesh)
        if is_gpu:
            self._build_gpu_structures()

    # ---------------------------------------------------------------------
    # GPU helpers
    # ---------------------------------------------------------------------
    def _build_gpu_structures(self):
        # tensors on the default device
        self._dev = current_device()
        self._face_centroids_t = torch.as_tensor(
            self.tissue_mesh.triangles_center, dtype=torch.float32, device=self._dev
        )
        # cuGraph adjacency
        coo = self._tissue_graph_cpu.tocoo()
        df = cudf.DataFrame(
            {
                "src": cp.asarray(coo.row),
                "dst": cp.asarray(coo.col),
                "weight": cp.ones(len(coo.data), dtype=cp.float32),
            }
        )
        self._G_gpu = cugraph.Graph()
        self._G_gpu.from_cudf_edgelist(
            df, source="src", destination="dst", edge_attr="weight", renumber=False
        )

    def _radius_faces_gpu(self, face_idx: int, radius: float):
        dist_df = cugraph.sssp(self._G_gpu, source=face_idx)
        mask = dist_df["distance"] <= radius
        return dist_df[mask]["vertex"].to_numpy()

    def _closest_face_gpu(self, points: np.ndarray):
        pts = torch.as_tensor(points[None, ...], dtype=torch.float32, device=self._dev)
        # KNN against tissue face centroids
        _, idx, _ = knn_points(pts, self._face_centroids_t[None, ...], K=1, return_sorted=False)
        return idx[0, :, 0].cpu().numpy()

    # ---------------------------------------------------------------------
    # Init common vars
    # ---------------------------------------------------------------------
    def init_vars(self):
        self.face_cell_ids = self.cell_mesh.metadata["_ply_raw"]["face"]["data"][
            "cell_id"
        ]
        self.vertex_cell_ids = self.cell_mesh.metadata["_ply_raw"]["vertex"]["data"][
            "cell_id"
        ]
        self.cell_ids = np.unique(self.face_cell_ids)

        if self.face_cell_ids is None or self.vertex_cell_ids is None:
            raise RuntimeError("Cell IDs missing in mesh metadata")

        self.tissue_face_centroids = self.tissue_mesh.triangles_center
        self.tissue_face_tree = cKDTree(self.tissue_face_centroids)
        self.tissue_vertices_tree = cKDTree(self.tissue_mesh.vertices)

        # cell centroids ---------------------------------------------------
        self.cell_centroids = {}
        for cid in self.cell_ids:
            try:
                self.cell_centroids[cid], _ = get_centroid(
                    self.cell_mesh, cid, self.face_cell_ids, self.vertex_cell_ids
                )
            except Exception as e:
                print(f"{c.WARNING}Error in cell{c.ENDC}: {cid} - {e}")

    # ---------------------------------------------------------------------
    # Public routines
    # ---------------------------------------------------------------------
    def map_cells(self):
        """Assign each tissue face to the closest cell (KD‑tree or GPU kNN)."""
        self.init_vars()

        # only keep cells present in features
        self.cell_ids = np.intersect1d(self.cell_ids, self.features["cell_id"].values)
        self.cell_centroids = {k: v for k, v in self.cell_centroids.items() if k in self.cell_ids}

        centroid_array = np.stack(list(self.cell_centroids.values()))  # (N,3)

        if is_gpu:
            # GPU kNN
            face2cell_idx = self._closest_face_gpu(centroid_array[self.cell_ids.argsort()])
            tissue_face_cell_ids = self.cell_ids[face2cell_idx]
        else:
            cell_tree = cKDTree(centroid_array)
            _, idx = cell_tree.query(self.tissue_face_centroids, k=1)
            tissue_face_cell_ids = self.cell_ids[idx]

        mapping = pd.DataFrame(
            {
                "tissue_face_id": np.arange(len(self.tissue_mesh.faces)),
                "cell_id": tissue_face_cell_ids,
            }
        )

        mapping.to_csv(self.mapping_path, index=False)
        print(f"{c.OKGREEN}Cell map saved to:{c.ENDC} {self.mapping_path}")
        self.mapping = mapping

    def get_neighborhood(self, radius: float = 34.0):
        """For every tissue face, store ids of neighbours within geodesic radius."""
        if self.mapping is None:
            raise RuntimeError("Call map_cells() first")

        face_neighbors = {}
        num_faces = len(self.tissue_mesh.faces)

        for fidx in range(num_faces):
            if is_gpu:
                neigh = self._radius_faces_gpu(fidx, radius)
            else:
                dist = dijkstra(
                    csgraph=self._tissue_graph_cpu,
                    directed=False,
                    indices=fidx,
                    return_predecessors=False,
                    min_only=True,
                )
                neigh = np.where(dist <= radius)[0]
            face_neighbors[fidx] = neigh

        new_cols = pd.DataFrame(
            {
                "tissue_face_id": np.array(list(face_neighbors.keys()), dtype=int),
                "tissue_neighbors": face_neighbors.values(),
            }
        )

        self.mapping = self.mapping.merge(new_cols, on="tissue_face_id", how="left")
        self.mapping.to_csv(self.mapping_path, index=False)
        print(f"{c.OKGREEN}Neighborhood saved to:{c.ENDC} {self.mapping_path}")

    # ------------------------------------------------------------------
    def color_mesh(self, feature_name: str, out_path: str, cmap=None):
        """Paint tissue mesh faces according to selected feature."""
        if self.mapping is None and os.path.exists(self.mapping_path):
            self.mapping = pd.read_csv(self.mapping_path)

        self.init_vars()

        if feature_name not in self.features.columns:
            raise ValueError(f"Feature not found: {feature_name}")

        feature_map = self.features.set_index("cell_id")[feature_name].to_dict()
        face_values = self.mapping["cell_id"].map(feature_map).to_numpy()

        # Optional neighbour averaging -----------------------------------
        aux_face_values = face_values.copy()
        for i, row in self.mapping.iterrows():
            neigh = row.get("tissue_neighbors")
            if neigh is None or (isinstance(neigh, float) and np.isnan(neigh)):
                continue
            neigh = np.asarray(neigh)
            vals = [face_values[int(n)] for n in neigh if not np.isnan(face_values[int(n)])]
            if vals:
                aux_face_values[i] = np.mean(vals)
        face_values = aux_face_values

        # ----------------------------------------------------------------
        if cmap is None:
            colors = [
                (0, 0, 1),
                (0, 0.5, 1),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ]
            cmap = LinearSegmentedColormap.from_list("custom_jet", colors, N=1024)

        vmin, vmax = np.percentile(face_values, [1, 99])
        norm = BoundaryNorm(np.linspace(vmin, vmax, cmap.N), ncolors=cmap.N)
        face_colors = cmap(norm(face_values))

        self.tissue_mesh.visual.face_colors = face_colors
        self.tissue_mesh.export(out_path, file_type="ply")
        print(f"{c.OKGREEN}Mesh colored{c.ENDC}: {out_path}")

        pd.DataFrame(
            {"tissue_face_id": np.arange(len(face_values)), "value": face_values}
        ).to_csv(out_path.replace(".ply", ".csv"), index=False)

        return self.tissue_mesh

# -----------------------------------------------------------------------------
# Orchestrator helpers
# -----------------------------------------------------------------------------

def run_heatmap(
    mesh_path, tissue_path, features_path, mapping_path, feature_name, out_path, cmap=None
):
    mapper = CellTissueMap(mesh_path, tissue_path, features_path, mapping_path)
    mapper.map_cells()
    mapper.get_neighborhood(radius=20)
    mapper.color_mesh(feature_name, out_path, cmap)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    skeleton = False
    skip_existing_features = False

    for fname in os.listdir(mesh_dir):
        try:
            if not fname.endswith(".ply"):
                continue
            print(f"{c.OKGREEN}Processing{c.ENDC}: {fname}")

            mesh_path = os.path.join(mesh_dir, fname)

            tissue_path = os.path.join(tissue_dir, fname.replace(".ply", "_SHF_segmentation_tissue.ply"))
            skeleton_path = os.path.join(skeleton_dir, fname.replace(".ply", "_SHF_segmentation_tissue_skeleton.ply"))
            features_path = os.path.join(features_dir, fname.replace(".ply", "_features.csv"))

            if not (os.path.exists(features_path) and skip_existing_features):
                run_features(
                    mesh_path,
                    skeleton_path if skeleton else tissue_path,
                    features_path,
                    verbose=1, parallelize=PARALLELIZE
                )
            else:
                print(f"{c.OKGREEN}Features already extracted{c.ENDC}: {features_path}")

            mapping_path = os.path.join(features_dir, fname.replace(".ply", "_mapping.csv"))
            out_path = os.path.join(out_dir, fname.replace(".ply", "_columnarity.ply"))
            run_heatmap(
                mesh_path, tissue_path,
                features_path, mapping_path,
                "columnarity", out_path
            )
        except Exception as e:
            print(f"{c.FAIL}Error processing{c.ENDC}: {fname}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
