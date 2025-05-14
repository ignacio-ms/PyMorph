import numpy as np
from pymeshlab import PercentageValue

import mcubes
import trimesh
import porespy as ps
from skimage import morphology

from joblib import Parallel, delayed
import warnings
import os
import sys

from plyfile import PlyData, PlyElement  # Importing plyfile for PLY export

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c
from util.data import imaging

warnings.filterwarnings("ignore", category=DeprecationWarning)

base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/MeisdKO_WT_mTmG_Columnarity/'  # MeisdKO_WT_mTmG_Columnarity
seg_dir = os.path.join(base_dir, 'results')
out_dir = os.path.join(base_dir, 'results', 'mesh', 'cells')


def median_3d_array(img, disk_size=3, thr=.4):
    from skimage import morphology, filters

    if len(img.shape) == 4:
        img = img[:, :, :, 0]

    # img = (img > 0).astype(np.uint8) * 255

    img_closed = morphology.binary_closing(img, morphology.ball(disk_size))

    # img_denoised = filters.gaussian(img_closed, sigma=3)
    # img_denoised = (img_denoised > thr).astype(np.uint8) * 255
    # return img_denoised

    return img_closed


def process_cell(cell_data, metadata):
    try:
        coords = cell_data['mask']
        centroid = cell_data['centroid']
        cell_id = cell_data['cell_id']

        img = median_3d_array(coords)
        aux = np.zeros(np.array(img.shape), dtype=np.uint8)
        # Border (1px) to 0
        aux[
        1:-1,
        1:-1,
        1:-1
        ] = img[1:-1, 1:-1, 1:-1]

        vert, trian = mcubes.marching_cubes(aux, 0)
        if len(vert) == 0 or len(trian) == 0:
            return None

        # vert -= vert.mean(axis=0)
        vert += centroid
        vert *= np.array([metadata['x_res'], metadata['y_res'], metadata['z_res']])

        mesh = trimesh.Trimesh(vertices=vert, faces=trian, process=False)
        # print(f'{c.OKGREEN}Mesh created{c.ENDC}: #vertices={len(vert)}, #faces={len(trian)}')

        # print(f'{c.OKBLUE}Cell ID{c.ENDC}: {cell_id}')
        # print(f'{c.OKGREEN}Mesh created{c.ENDC}: #vertices={len(mesh.vertices)}, #faces={len(mesh.faces)}')

        # Apply Laplacian smoothing here
        trimesh.smoothing.filter_laplacian(
            mesh, lamb=0.8, iterations=30,
            volume_constraint=False
        )
        # print(f'{c.OKGREEN}Laplacian smoothing applied{c.ENDC}: #vertices={len(mesh.vertices)}, #faces={len(mesh.faces)}')

        # Fixes (Watertigh, normals, etc.)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

        if not mesh.is_watertight:
            return None

        # print(f'{c.OKGREEN}Mesh smoothed{c.ENDC}: #vertices={len(mesh.vertices)}, #faces={len(mesh.faces)}')

        # ---------------------------------------------------
        # Mesh Decimation
        # ---------------------------------------------------
        import pymeshlab

        #     print(f'{c.OKGREEN}Decimating mesh{c.ENDC}')
        vs, fs = np.array(mesh.vertices), np.array(mesh.faces)
        pmesh = pymeshlab.Mesh(vertex_matrix=vs, face_matrix=fs)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pmesh, 'cell_mesh')
        #
        # for i in range(1):
        #    ms.apply_filter('meshing_repair_non_manifold_edges')
        #    # ms.apply_filter('meshing_close_holes', maxholesize=5000)
        ms.apply_filter(
            'meshing_isotropic_explicit_remeshing', iterations=1,
            checksurfdist=False, targetlen=PercentageValue(5)
        )
        ms.apply_filter('apply_coord_taubin_smoothing')
        #
        vs, fs = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
        # print(len(vs), len(fs))
        mesh = trimesh.Trimesh(vertices=vs, faces=fs, process=False)

        #     print(f'{c.OKGREEN}Mesh decimated{c.ENDC}: #vertices={len(mesh.vertices)}, #faces={len(mesh.faces)}')

        # print(f'{c.OKGREEN}Mesh remeshed{c.ENDC}: #vertices={len(mesh.vertices)}, #faces={len(mesh.faces)}')

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
    except Exception as e:
        print(f'{c.WARNING}Error processing cell{c.ENDC}: {e} - Skipping cell')
        import traceback
        traceback.print_exc()
        return None


def filter_cells_by_volume(cell_data_list, perc_b=15, perc_t=92):
    """Remove cells by volume based on the top and bottom percentiles."""
    volumes = np.array([cell_data['volume'] for cell_data in cell_data_list])
    lower_bound = np.percentile(volumes, perc_b)
    upper_bound = np.percentile(volumes, perc_t)

    filtered_cells = [
        cell_data for cell_data in cell_data_list
        if lower_bound <= cell_data['volume'] <= upper_bound
    ]

    return filtered_cells


def marching_cubes(img, metadata):
    props = ps.metrics.regionprops_3D(morphology.label(img))
    # Use all available CPUs minus two
    num_jobs = 1

    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [img[ce[0], ce[1], ce[2]] for ce in centroids]

    # Extract necessary data into picklable objects
    cell_data_list = []
    for i, p in enumerate(props):
        cell_data = {
            'mask': p.mask.astype(np.uint8),
            'centroid': p.centroid,
            'cell_id': centroids_labels[i],
            'volume': p.volume,
        }
        cell_data_list.append(cell_data)

    # Filter cells by volume
    past_cells = len(cell_data_list)
    cell_data_list = filter_cells_by_volume(cell_data_list)

    print(
        f'{c.OKGREEN}Filtered cells{c.ENDC}: {len(cell_data_list)} / {past_cells} ({100 * len(cell_data_list) / past_cells:.2f}%)')

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


def run(img_path, path_out, metadata, verbose=0):
    img = imaging.read_image(img_path, axes='XYZ', verbose=1)

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


def main():
    for r_name in os.listdir(seg_dir):
        try:
            if not r_name.endswith('.tif'):
                continue

            r_path = os.path.join(seg_dir, r_name)
            out_path = os.path.join(out_dir, r_name.replace('_mask.tif', '.ply'))

            print(f'{c.BOLD}Processing{c.ENDC}: {r_path}')
            run(
                r_path, out_path,
                metadata={'x_res': 0.7575758, 'y_res': 0.7575758, 'z_res': 0.9999286},
                verbose=1
            )

            print(f'{c.OKGREEN}Saved{c.ENDC}: {out_path}')
        except Exception as e:
            print(f'{c.FAIL}Error processing image{c.ENDC}: {r_name}')
            print(f'{c.FAIL}Error{c.ENDC}: {e}')
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
