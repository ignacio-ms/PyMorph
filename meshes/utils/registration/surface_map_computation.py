import numpy
import subprocess

import trimesh

from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import find_specimen, HtDataset
from auxiliary import values as v

import os


def run(
        source_mesh, target_mesh,
        source_landmarks, target_landmarks,
        output_path, data_path=None,
        init_method=0,
        n_pre_iters=100,
        n_main_iters=500,
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
    if data_path is None or data_path == v.data_path:
        data_path = v.data_path + 'surface_map/'
        data_path_out = v.data_path + 'surface_map_output/'

    else:
        data_path_out = data_path.replace(
            data_path.split('/')[-1],
            data_path.split('/')[-1] + '_output'
        )

    if verbose:
        print(f'{c.OKBLUE}Running surface map computation{c.ENDC}...')
        print(f'\t{c.BOLD}Creating auxiliary directories{c.ENDC}...')
    # If the output path does not exist, create it
    subprocess.run(
        f'mkdir -p {output_path} {data_path} {data_path_out}',
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

    # Run the surface map computation
    # current_dir = subprocess.run(
    #     'pwd',
    #     shell=True, check=True,
    #     stdout=subprocess.PIPE
    # ).stdout.decode('utf-8').strip()

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
        f'bash {current_dir}/surface_map/{gpu_type}/SurfaceMapComputation --path {data_path} '
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
        f'mv {data_path_out}/* {output_path}',
        shell=True, check=True
    )

    # Remove auxiliary directories
    subprocess.run(
        f'rm -r {data_path} {data_path_out}',
        shell=True, check=True
    )

    if verbose:
        print(f'{c.OKGREEN}Output moved{c.ENDC}')
        print(f'{c.OKGREEN}Surface map computation finished{c.ENDC}')


def check_mesh_consistency(source_mesh, target_mesh):
    """
    Check if the meshes are consistent. If not, transform the mesh with the least number of vertices to have the same,
    the same number of vertices, faces, and edges.
    :param source_mesh:
    :param target_mesh:
    :return:
    """
    source = trimesh.load(source_mesh, file_type='ply')
    target = trimesh.load(target_mesh, file_type='ply')

    if source.vertex.

