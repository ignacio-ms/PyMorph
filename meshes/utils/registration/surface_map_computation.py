import json

import numpy
import subprocess

import numpy as np
import trimesh
import pymeshlab
from scipy.spatial import cKDTree
import open3d as o3d

from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import find_specimen, HtDataset, find_group
from auxiliary import values as v

import os


def run(
        source_mesh, target_mesh,
        source_landmarks, target_landmarks,
        output_path, specimen, data_path=None,
        init_method=2,
        n_pre_iters=200,
        n_main_iters=600,
        gpu_type='INTEL',
        verbose=0
):
    """
    Compute the surface map between two meshes using the landmarks.
    :param source_mesh: Path to the source mesh.
    :param target_mesh: Path to the target mesh.
    :param source_landmarks: Path to the source landmarks.
    :param target_landmarks: Path to the target landmarks.
    :param output_path: Path to save the surface map.
    :param data_path: Path to the data directory.
    :param init_method: Initialization method.
    :param n_pre_iters: Number of pre-iterations.
    :param n_main_iters: Number of main iterations. 0 (automatic), 1 (Schreiner), 2 (Born A -> B), 3 (Born B -> A)
    :param gpu_type: GPU type. 'INTEL' or 'AMD'.
    :param verbose: Verbosity level.
    """
    print(f'{c.OKBLUE}Running surface map computation{c.ENDC}...{specimen}')
    # if data_path is None or data_path == v.data_path:
    data_path = data_path + f'{specimen}/surface_map/'
    data_path_out = data_path + f'{specimen}/surface_map/map/'
    #
    # else:
    #     data_path_out = data_path + 'map/'

    if verbose:
        print(f'{c.OKBLUE}Running surface map computation{c.ENDC}...')
        print(f'\t{c.BOLD}Creating auxiliary directories{c.ENDC}...')
    # If the output path does not exist, create it
    subprocess.run(
        f'mkdir -p {output_path} {data_path} {data_path_out} {output_path}/map/',
        shell=True, check=True
    )

    if verbose:
        print(f'\t{c.BOLD}Moving meshes and landmarks{c.ENDC}...')

    # Move meshes and landmarks to data directory
    for file in [source_mesh, target_mesh, source_landmarks, target_landmarks]:
        subprocess.run(
            f'cp {file} {data_path}',
            shell=True, check=True
        )

    # Get just the file names
    source_mesh = source_mesh.split('/')[-1]
    target_mesh = target_mesh.split('/')[-1]
    source_landmarks = source_landmarks.split('/')[-1]
    target_landmarks = target_landmarks.split('/')[-1]

    try:
        current_dir = os.path.dirname(__file__)
    except NameError:
        current_dir = os.getcwd()

    current_dir = current_dir.split('/')[:-2]
    current_dir = '/'.join(current_dir)

    if verbose:
        print(f'\t{c.BOLD}Running SurfaceMapComputation{c.ENDC}...')

    # Define the LD_LIBRARY_PATH
    subprocess.run(
        f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{current_dir}/surface_map/{gpu_type}/libs/',
        shell=True, check=True
    )

    subprocess.run(
        f'{current_dir}/surface_map/{gpu_type}/SurfaceMapComputation --path {data_path} '
        f'--init_method {init_method} --n_pre_iters {n_pre_iters} '
        f'--n_main_iters {n_main_iters} --shapeA {source_mesh} '
        f'--shapeB {target_mesh} --landmarksA {source_landmarks} '
        f'--landmarksB {target_landmarks}',
        shell=True, check=True
    )

    if verbose:
        print(f'\t{c.BOLD}Moving output{c.ENDC}...')

    # Move the output to the output directory
    subprocess.run(
        f'cp {data_path}{source_mesh.split(".")[0]}_on_{target_mesh.split(".")[0]}.ply {output_path}',
        shell=True, check=True
    )

    subprocess.run(
        f'cp {data_path_out}* {output_path}map/',
        shell=True, check=True
    )

    # Remove auxiliary directories
    subprocess.run(
        f'rm -r {data_path}',
        shell=True, check=True
    )

    if verbose:
        print(f'{c.OKGREEN}Output moved{c.ENDC}')
        print(f'{c.OKGREEN}Surface map computation finished{c.ENDC}')


def check_mesh_consistency(source_mesh, target_mesh):
    """
    Check if the source and target meshes have the same number of vertices and faces.
    """
    source = trimesh.load(source_mesh, file_type='ply')
    target = trimesh.load(target_mesh, file_type='ply')

    print(f'\t{c.OKGREEN}Source mesh number of faces: {c.ENDC}{len(source.faces)}')
    print(f'\t{c.OKGREEN}Target mesh number of faces: {c.ENDC}{len(target.faces)}')

    if len(source.faces) > len(target.faces):
        print(f'{c.WARNING}Source mesh has more faces than target mesh{c.ENDC}')
        print(f'{c.WARNING}Decimating source mesh{c.ENDC}')
        target = subdivide_mesh(target, len(source.faces))
        target.export(target_mesh)

        s = find_specimen(target_mesh)
        update_land_pinned(s)

    elif len(source.faces) < len(target.faces):
        print(f'{c.WARNING}Target mesh has more faces than source mesh{c.ENDC}')
        print(f'{c.WARNING}Decimating target mesh{c.ENDC}')
        source = subdivide_mesh(source, len(target.faces))
        source.export(source_mesh)

        for g in v.specimens.keys():
            if g in source_mesh:
                update_land_pinned('', gr=g)
                break


def subdivide_mesh(mesh, target_face_count):
    mesh.export('temp_mesh.ply')

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh('temp_mesh.ply')
    # ms.show_polyscope()

    current_face_count = len(mesh.faces)

    while current_face_count < target_face_count:
        ms.apply_filter('meshing_surface_subdivision_loop', iterations=1)
        current_face_count = ms.current_mesh().face_number()

    ms.save_current_mesh('subdivided_mesh.ply')
    subdivided_mesh = trimesh.load('subdivided_mesh.ply')

    if len(subdivided_mesh.faces) > target_face_count:
        subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_face_count)

    subprocess.run(['rm', 'temp_mesh.ply'])
    subprocess.run(['rm', 'subdivided_mesh.ply'])

    return subdivided_mesh


def update_land_pinned(specimen, gr=None, path=None, tissue='myocardium'):
    landmarks_guide_names = v.myo_myo_landmark_names

    if path is None:
        path = v.data_path + f'Landmarks/'

    if gr is not None:
        in_path = path + 'ATLAS/ATLAS_' + gr + '_key_points.json'
        out_path = path + 'ATLAS/ATLAS_' + gr + '_landmarks.pinned'
        mesh_path = v.data_path + f'ATLAS/{tissue}/ATLAS_{gr}.ply'
    else:
        in_path = path + f'2019{specimen}_key_points.json'
        out_path = path + f'2019{specimen}_landmarks.pinned'
        gr = find_group(specimen)
        mesh_path = v.data_path + f'{gr}/3DShape/Tissue/{tissue}/2019{specimen}Shape.ply'

    with open(in_path, 'r') as f:
        key_points = json.load(f)

    names = list(key_points.keys())
    coords = np.array(list(key_points.values()))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    kdtree = cKDTree(np.asarray(mesh.vertices))

    landmarks = {}

    for name, coord in zip(names, coords):
        if name in landmarks_guide_names:
            _, idx = kdtree.query(coord)
            landmarks[name] = idx

    with open(out_path, 'w') as f:
        f.write('\n'.join([str(v) for v in landmarks.values()]))

    print(f'Updated landmarks for {gr} - {specimen if specimen != "" else "ATLAS"}')
