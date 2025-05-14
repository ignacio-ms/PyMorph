import os
import numpy as np
import pandas as pd
import trimesh
from joblib import Parallel, delayed

# ──────────────────────────────────────────────────────────────────────────────
# Optional GPU back‑end (set USE_GPU=True via env‑var or gpu_utils)
# ──────────────────────────────────────────────────────────────────────────────
from util.gpu.gpu_torch import gpu_enabled, current_device, torch_device  # local helper

if gpu_enabled():
    import torch
    from pytorch3d.ops import knn_points
    import cudf, cugraph
    import cupy as cp
else:
    torch = None  # type: ignore

# -----------------------------------------------------------------------------
# Project‑specific utilities
# -----------------------------------------------------------------------------
from meshes.utils.features.visualizer import CellVisualization
from scipy.spatial import cKDTree
from meshes.utils.features.operator import (
    get_centroid,
    find_closest_face,
    get_neighborhood_points,
    fit_plane,
    approximate_ellipsoid,
    get_longest_axis,
    get_angle,
    build_face_adjacency_csr_matrix,
)

__all__ = [
    "MeshFeatureExtractor",
]


class MeshFeatureExtractor:
    """Extract per‑cell geometric features from two registered meshes.

    If the environment variable ``USE_GPU`` evaluates to *true* **and** CUDA is
    available, the heavy nearest‑neighbour and graph‑radius queries are executed
    on the GPU (multi‑GPU round‑robin).  Otherwise the original CPU path is
    used.  Results are identical up to ±1 face / ±1 voxel numerical tolerance.
    """

    # ────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, cell_mesh: trimesh.Trimesh, tissue_mesh: trimesh.Trimesh):
        self.cell_mesh = cell_mesh
        self.tissue_mesh = tissue_mesh

        self.face_cell_ids = self.cell_mesh.metadata["_ply_raw"]["face"]["data"][
            "cell_id"
        ]
        self.vertex_cell_ids = self.cell_mesh.metadata["_ply_raw"]["vertex"]["data"][
            "cell_id"
        ]
        self.cell_ids = np.unique(self.face_cell_ids)

        if self.face_cell_ids is None or self.vertex_cell_ids is None:
            raise ValueError("Cell IDs not found in the mesh metadata")

        # Pre‑compute tissue data ------------------------------------------------
        self.tissue_face_centroids = self.tissue_mesh.triangles_center
        self.tissue_face_tree = cKDTree(self.tissue_face_centroids)
        self.tissue_vertices_tree = cKDTree(self.tissue_mesh.vertices)
        self.tissue_graph_cpu = build_face_adjacency_csr_matrix(self.tissue_mesh)

        # GPU tensors -----------------------------------------------------------
        self._torch_device = (
            current_device() if gpu_enabled() else None
        )  # type: ignore

        if gpu_enabled():
            self._tissue_face_centroids_t = torch.as_tensor(
                self.tissue_face_centroids.copy(),
                dtype=torch.float32,
                device=self._torch_device
            )
            # build cugraph once
            self._build_tissue_cugraph()

    # ---------------------------------------------------------------------
    # GPU helpers
    # ---------------------------------------------------------------------
    def _closest_face_gpu(self, point: np.ndarray) -> int:
        """Return index of closest tissue face centre using GPU k‑NN."""
        pts = torch.as_tensor(
            point[None, None, :], dtype=torch.float32, device=self._torch_device
        )
        _, idx, _ = knn_points(
            pts, self._tissue_face_centroids_t[None, ...], K=1, return_sorted=False
        )
        return int(idx.item())

    def _build_tissue_cugraph(self):
        """Create cuGraph CSR representation of the tissue face adjacency."""
        coo = self.tissue_graph_cpu.tocoo()
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

    def _radius_faces_gpu(self, face_idx: int, radius: float) -> np.ndarray:
        """Return faces within a geodesic radius using cuGraph SSSP."""
        dist_df = cugraph.sssp(self._G_gpu, source=face_idx)
        mask = dist_df["distance"] <= radius
        return dist_df[mask]["vertex"].to_numpy()

    # ────────────────────────────────────────────────────────────────────────
    # Feature primitives
    # ────────────────────────────────────────────────────────────────────────
    def cell_perpendicularity(
        self,
        cell_id: int,
        sigma: float = 30.0,
        display: bool = False,
        dynamic_display: bool = False,
        radius: float = 10.0,
    ) -> float:
        """Perpendicularity metric in [0, 1]."""

        def deg2perpen(angle):
            return np.clip(1.0 - np.exp(-((angle - 90.0) ** 2) / (2.0 * sigma**2)), 0.0, 1.0)

        centroid, cell_vertices = get_centroid(
            self.cell_mesh, cell_id, self.face_cell_ids, self.vertex_cell_ids
        )

        cell_mesh = self.cell_mesh.submesh([
            np.where(self.face_cell_ids == cell_id)[0]
        ], append=True)
        if isinstance(cell_mesh, trimesh.Scene):
            cell_mesh = cell_mesh.dump(concatenate=True)

        # ----- nearest face ---------------------------------------------------
        if gpu_enabled():
            closest_face = self._closest_face_gpu(centroid)
            neigh_face_ids = self._radius_faces_gpu(closest_face, radius=radius)
            neigh_points = self.tissue_mesh.triangles_center[neigh_face_ids]
        else:
            closest_face = find_closest_face(centroid, self.tissue_face_tree)
            neigh_points, neigh_face_ids = get_neighborhood_points(
                self.tissue_mesh,
                closest_face,
                radius=radius,
                graph=self.tissue_graph_cpu,
            )

        plane = fit_plane(neigh_points)
        plane_point = neigh_points.mean(axis=0)

        # ----- ellipsoid axes (torch or numpy) --------------------------------
        if gpu_enabled():
            verts_t = torch.as_tensor(cell_vertices, dtype=torch.float32, device=self._torch_device)
            # Minimum volume enclosing ellipsoid uses numpy; fall back to CPU copy
            ellipse_c, ellipse_axes, ellipse_lengths = approximate_ellipsoid(
                verts_t.cpu().numpy(), method="mvee", tol=1e-4
            )
        else:
            ellipse_c, ellipse_axes, ellipse_lengths = approximate_ellipsoid(
                cell_vertices, method="mvee", tol=1e-4
            )

        longest_axis = get_longest_axis(ellipse_axes, ellipse_lengths)
        angle = get_angle(longest_axis, plane)

        # ----- optional visualisation ----------------------------------------
        if display:
            visualization = CellVisualization(
                tissue_mesh=self.tissue_mesh,
                cell_mesh=cell_mesh,
                centroid=centroid,
                dynamic_camera=False,
            )
            visualization.add_closest_face_centroid(self.tissue_face_centroids[closest_face])
            visualization.add_neighborhood_points(neigh_points)
            visualization.add_fitted_plane(plane, plane_point, size=40.0)
            visualization.add_longest_axis(centroid, longest_axis, length=20.0)
            visualization.add_plane_normal(
                plane_point, plane, length=15.0, color=[1.0, 1.0, 0.0, 1.0]
            )
            visualization.render_scene(live=dynamic_display)

        return float(deg2perpen(angle))

    # ---------------------------------------------------------------------
    def cell_sphericity(
        self,
        cell_id: int,
        method: str = "standard",
        sigma: float = 35.0,
        center: float = 0.6,
    ) -> float:
        """Return [0, 1] sphericity metric."""

        def activation(x):
            return 1.0 / (1.0 + np.exp(-sigma * (x - center)))

        cell_mesh = self.cell_mesh.submesh([
            np.where(self.face_cell_ids == cell_id)[0]
        ], append=True)
        if isinstance(cell_mesh, trimesh.Scene):
            cell_mesh = cell_mesh.dump(concatenate=True)

        if method == "standard":
            if gpu_enabled():
                verts_t = torch.as_tensor(
                    cell_mesh.vertices, dtype=torch.float32, device=self._torch_device
                )
                cov = torch.cov(verts_t.T)
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals, _ = torch.sort(eigvals, descending=True)
                eigvals = eigvals.cpu().numpy()
            else:
                cov = np.cov(cell_mesh.vertices.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(eigvals)[::-1]
            sphericity = ((eigvals[-1] / eigvals[0]) * (eigvals[-2] / eigvals[0])) ** (1.0 / 3.0)
            sphericity = np.clip(sphericity, 0.0, 1.0)

        elif method == "volume":
            volume = cell_mesh.volume
            area = cell_mesh.area
            if volume == 0 or area == 0:
                return 0.0
            roundness_value = (36.0 * np.pi) * (volume**2) / (area**3)
            sphericity = min(max(roundness_value, 0.0), 1.0)
        else:
            raise ValueError(f"Invalid method: {method}")

        return float(sphericity)

    # ---------------------------------------------------------------------
    @staticmethod
    def cell_columnarity(sphericity: float, perpendicularity: float) -> float:
        return (1.0 - sphericity) * perpendicularity

    @staticmethod
    def cell_columnarity_parallel(sphericity: float, perpendicularity: float) -> float:
        return (1.0 - sphericity) * (1.0 - perpendicularity)

    def _compute_features_single(self, cell_id: int):
        try:
            perp = self.cell_perpendicularity(cell_id)
            spher = self.cell_sphericity(cell_id, method="standard")
            col = self.cell_columnarity(spher, perp)
            col_par = self.cell_columnarity_parallel(spher, perp)
        except Exception as e:
            print(f"Error in cell {cell_id}: {e}")
            perp = spher = col = col_par = np.nan
        return {
            "cell_id": cell_id,
            "perpendicularity": perp,
            "sphericity": spher,
            "columnarity": col,
            "columnarity_parallel": col_par,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────
    def extract(self, n_jobs: int = 3) -> pd.DataFrame:
        """Compute features for every cell ID in *parallel*.

        * CPU path → joblib multiprocessing
        * GPU path → thread pool, each worker bound to its own CUDA device
        """

        def compute_features(cell_id: int):
            try:
                perp = self.cell_perpendicularity(cell_id)
                spher = self.cell_sphericity(cell_id, method="standard")
                col = self.cell_columnarity(spher, perp)
                col_par = self.cell_columnarity_parallel(spher, perp)
            except Exception as e:
                print(f"Error in cell {cell_id}: {e}")
                perp = spher = col = col_par = np.nan
            return {
                "cell_id": cell_id,
                "perpendicularity": perp,
                "sphericity": spher,
                "columnarity": col,
                "columnarity_parallel": col_par,
            }

        # ------------------- CPU path --------------------------------------
        if not gpu_enabled():
            rows = Parallel(n_jobs=n_jobs)(delayed(compute_features)(cid) for cid in self.cell_ids)
            return pd.DataFrame(rows)

        # ------------------- GPU path --------------------------------------
        num_gpus = torch.cuda.device_count()
        splits = np.array_split(self.cell_ids, num_gpus)
        results = []

        import concurrent.futures as cf

        def _gpu_worker(rank: int, subset: np.ndarray):
            with torch_device(rank):
                # build a brand‑new extractor on this GPU
                local_extractor = MeshFeatureExtractor(self.cell_mesh, self.tissue_mesh)
                # force its tensors to this device
                local_extractor._torch_device = current_device(rank)
                local_extractor._tissue_face_centroids_t = \
                    local_extractor._tissue_face_centroids_t.to(local_extractor._torch_device)
                # (rebuild cuGraph here if you use it)
                local = []
                for cid in subset:
                    local.append(local_extractor._compute_features_single(cid))
                return local

        with cf.ThreadPoolExecutor(max_workers=num_gpus) as exe:
            futs = [exe.submit(_gpu_worker, r, s) for r, s in enumerate(splits) if len(s)]
            for fut in cf.as_completed(futs):
                results.extend(fut.result())

        return pd.DataFrame(results)
