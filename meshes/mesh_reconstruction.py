import numpy as np
from scipy import ndimage

import mcubes
import trimesh
import porespy as ps
from skimage import morphology

from joblib import Parallel, delayed


from auxiliary.utils.colors import bcolors as c
from auxiliary.data import imaging

from filtering.cardiac_region import filter_by_tissue

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def median_3d_array(img, disk_size=3):
    if len(img.shape) == 4:
        img = img[:, :, :, 0]
    return ndimage.median_filter(img, size=disk_size)


def process_cell(cell_data, metadata):
    coords = cell_data['mask']
    centroid = cell_data['centroid']
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
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.8, iterations=40, volume_constraint=False)
    mesh = mesh.simplify_quadratic_decimation(100)
    return mesh


def marching_cubes(img, metadata):
    props = ps.metrics.regionprops_3D(morphology.label(img))
    num_jobs = -1  # Use all available CPUs

    # Extract necessary data into picklable objects
    cell_data_list = []
    for p in props:
        cell_data = {
            'mask': p.mask.astype(np.uint8),
            'centroid': p.centroid,
        }
        cell_data_list.append(cell_data)

    # Process cells in parallel
    meshes = Parallel(n_jobs=num_jobs)(
        delayed(process_cell)(cell_data, metadata) for cell_data in cell_data_list
    )

    # Filter out None results and combine meshes
    meshes = [mesh for mesh in meshes if mesh is not None]
    combined_mesh = trimesh.util.concatenate(meshes)
    return combined_mesh


def run(img_path, path_out, img_path_raw, lines_path, tissue, level, verbose=0):
    img = imaging.read_image(img_path, axes='XYZ', verbose=1)
    lines = imaging.read_image(lines_path, axes='XYZ', verbose=1)
    metadata, _ = imaging.load_metadata(img_path_raw)

    img = filter_by_tissue(
        img, lines, tissue,
        dilate=2 if level == 'Membrane' else 3, dilate_size=3,
        verbose=1
    )

    mesh = marching_cubes(img, metadata)
    mesh.export(path_out)

    if verbose:
        print(f'{c.OKGREEN}Saved{c.ENDC}: {path_out}')

