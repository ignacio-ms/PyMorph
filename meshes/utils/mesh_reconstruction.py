import numpy as np
from scipy import ndimage

import mcubes
import trimesh
import porespy as ps
from scipy.ndimage import zoom
from skimage import morphology

from joblib import Parallel, delayed
import multiprocessing

from plyfile import PlyData, PlyElement  # Importing plyfile for PLY export

from auxiliary.utils.colors import bcolors as c
from auxiliary.data import imaging

from filtering.cardiac_region import filter_by_tissue
from auxiliary.data.dataset_ht import find_specimen

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Remove the get_color_from_id function if not needed


def median_3d_array(img, disk_size=3):
    if len(img.shape) == 4:
        img = img[:, :, :, 0]
    return ndimage.median_filter(img, size=disk_size)


def process_cell(cell_data, metadata):
    coords = cell_data['mask']
    centroid = cell_data['centroid']
    cell_id = cell_data['cell_id']

    add = 10
    aux = np.zeros(np.array(coords.shape) + add, dtype=np.uint8)
    aux[
        add // 2: -add // 2,
        add // 2: -add // 2,
        add // 2: -add // 2
    ] = coords
    coords = aux
    coords = median_3d_array(coords)
    vert, trian = mcubes.marching_cubes(mcubes.smooth(coords), 0)
    if len(vert) == 0 or len(trian) == 0:
        return None

    vert -= vert.mean(axis=0)
    vert += centroid
    vert *= np.array([metadata['x_res'], metadata['y_res'], metadata['z_res']])

    mesh = trimesh.Trimesh(vertices=vert, faces=trian, process=False)

    # Apply Laplacian smoothing here
    trimesh.smoothing.filter_laplacian(
        mesh, lamb=0.8, iterations=40,
        volume_constraint=False
    )
    mesh = mesh.simplify_quadratic_decimation(250)

    # Fixes (Watertigh, normals, etc.)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if not mesh.is_watertight:
        return None

    # Access vertex normals (computes them if not already computed)
    normals = mesh.vertex_normals

    # Prepare cell IDs for vertices and faces
    vertex_cell_ids = np.full(len(mesh.vertices), cell_id, dtype=np.int32)
    face_cell_ids = np.full(len(mesh.faces), cell_id, dtype=np.int32)

    return {
        'vertices': mesh.vertices,
        'faces': mesh.faces,
        'normals': normals,
        'vertex_cell_ids': vertex_cell_ids,
        'face_cell_ids': face_cell_ids
    }


def marching_cubes(img, metadata):
    props = ps.metrics.regionprops_3D(morphology.label(img))
    # Use all available CPUs minus two
    num_jobs = max(1, multiprocessing.cpu_count() - 2)

    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [img[ce[0], ce[1], ce[2]] for ce in centroids]

    # Extract necessary data into picklable objects
    cell_data_list = []
    for i, p in enumerate(props):
        cell_data = {
            'mask': p.mask.astype(np.uint8),
            'centroid': p.centroid,
            'cell_id': centroids_labels[i]
        }
        cell_data_list.append(cell_data)

    # Process cells in parallel
    cell_meshes = Parallel(n_jobs=num_jobs)(
        delayed(process_cell)(cell_data, metadata) for cell_data in cell_data_list
    )

    # Initialize lists to collect data
    all_vertices = []
    all_faces = []
    all_normals = []
    all_vertex_cell_ids = []
    all_face_cell_ids = []

    vertex_offset = 0

    for cell_mesh in cell_meshes:
        if cell_mesh is None:
            continue
        vertices = cell_mesh['vertices']
        faces = cell_mesh['faces'] + vertex_offset  # Adjust face indices
        normals = cell_mesh['normals']
        vertex_cell_ids = cell_mesh['vertex_cell_ids']
        face_cell_ids = cell_mesh['face_cell_ids']

        all_vertices.append(vertices)
        all_faces.append(faces)
        all_normals.append(normals)
        all_vertex_cell_ids.append(vertex_cell_ids)
        all_face_cell_ids.append(face_cell_ids)

        vertex_offset += len(vertices)

    # Combine all data
    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_normals = np.vstack(all_normals)
    all_vertex_cell_ids = np.concatenate(all_vertex_cell_ids)
    all_face_cell_ids = np.concatenate(all_face_cell_ids)

    return {
        'vertices': all_vertices,
        'faces': all_faces,
        'normals': all_normals,
        'vertex_cell_ids': all_vertex_cell_ids,
        'face_cell_ids': all_face_cell_ids
    }


def run(img_path, path_out, img_path_raw, lines_path, tissue, level, verbose=0):
    img = imaging.read_image(img_path, axes='XYZ', verbose=1)

    if img_path_raw is None:
        img_path_raw = img_path

    metadata, _ = imaging.load_metadata(img_path_raw)

    if lines_path is not None and tissue is not None:
        lines = imaging.read_image(lines_path, axes='XYZ', verbose=1)
        img = filter_by_tissue(
            img, lines, tissue,
            dilate=2 if level == 'Membrane' else 3, dilate_size=3,
            verbose=1
        )

    mesh_data = marching_cubes(img, metadata)

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    normals = mesh_data['normals']
    vertex_cell_ids = mesh_data['vertex_cell_ids']
    face_cell_ids = mesh_data['face_cell_ids']

    # Prepare structured arrays for vertices and faces
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('cell_id', 'i4')
    ]
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]
    vertex_data['cell_id'] = vertex_cell_ids

    face_dtype = [('vertex_indices', 'i4', (3,)), ('cell_id', 'i4')]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces
    face_data['cell_id'] = face_cell_ids

    # Create PlyElement objects
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # Write the PLY file using plyfile
    PlyData([vertex_element, face_element], text=True).write(path_out)

    if verbose:
        print(f'{c.OKGREEN}Saved{c.ENDC}: {path_out}')
