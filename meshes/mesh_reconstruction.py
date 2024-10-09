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


def process_cell(p, metadata):
    coords = p.mask.astype(np.uint8)
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
    vert -= vert.mean(axis=0)
    vert += p.centroid
    vert *= np.array([metadata['x_res'], metadata['y_res'], metadata['z_res']])

    if len(vert) > 0 and len(trian) > 0:
        mesh = trimesh.Trimesh(vertices=vert, faces=trian, process=False)
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.5, iterations=20)
        mesh = mesh.simplify_quadratic_decimation(int(100))

        # laplacian smoothing
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=20, volume_constraint=False)
        mesh = mesh.simplify_quadratic_decimation(int(100))
        return mesh
    else:
        return None


def marching_cubes(img, metadata):
    props = ps.metrics.regionprops_3D(morphology.label(img))
    num_jobs = -1  # Use all available CPUs

    meshes = Parallel(n_jobs=num_jobs)(
        delayed(process_cell)(p, metadata) for p in props
    )

    # Filter out any None results
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

    mesh = marching_cubes(img)
    mesh.export(path_out)

    if verbose:
        print(f'{c.OKGREEN}Saved{c.ENDC}: {path_out}')

