import numpy as np

import os
import sys
import tempfile  # We add this to handle ephemeral files

import pymeshlab
from scipy import ndimage
import mcubes
import trimesh
from scipy.ndimage import zoom
from plyfile import PlyData, PlyElement

import warnings


try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging
from util.misc.colors import bcolors as c

warnings.filterwarnings("ignore", category=DeprecationWarning)

base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/2025/'
tissue_seg_dir = os.path.join(base_dir, 'binary_masks')
out_dir = os.path.join(base_dir, 'results', 'mesh', 'tissue')
out_skeleton_dir = os.path.join(base_dir, 'results', 'mesh', 'skeleton')


def median_3d_array(img, disk_size=3, thr=.3):
    """
    Apply a median filter to a 3D image. (Stack by stack)
    :param img: Input 3D image
    :param disk_size: Size of the disk structuring element
    :return: Image with median filter applied
    """
    from skimage import morphology, filters

    if len(img.shape) == 4:
        img = img[:, :, :, 0]

    img = (img > 0).astype(np.uint8) * 255
    img_closed = morphology.binary_closing(img, morphology.ball(disk_size))
    img_denoised = filters.gaussian(img_closed, sigma=10)

    img_denoised = (img_denoised > thr).astype(np.uint8) * 255
    return img_denoised

def post_process(mesh, n_iters=1):
    """
    Post-process a mesh to ensure it is watertight.
    """
    # if not isinstance(mesh, trimesh.Trimesh):
    #     mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])

    for _ in range(n_iters):
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        max_iters = 10
        is_watertight = mesh.is_watertight
        while not is_watertight and max_iters > 0:
            mesh.fill_holes()

            is_watertight = mesh.is_watertight
            max_iters -= 1
    return mesh


def fix_mesh_with_pymeshlab(verts, faces, perc=1.0, n_iters=30):
    from pymeshlab import PercentageValue

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces), 'raw_mesh')

    #for _ in range(n_iters):
    ms.apply_filter('meshing_repair_non_manifold_edges')
    #ms.apply_filter('meshing_close_holes', maxholesize=5000)
    ms.apply_filter(
        'meshing_isotropic_explicit_remeshing', iterations=n_iters,
        checksurfdist=False, targetlen=PercentageValue(perc)
    )
    ms.apply_filter('apply_coord_taubin_smoothing')

    repaired = ms.current_mesh()
    return repaired.vertex_matrix(), repaired.face_matrix()


def marching_cubes(img, metadata, n_faces=7500, zoom_factor=1.0):
    """
    Generate a mesh from a binary image using the marching cubes algorithm.
    :param img: Input binary 3D image
    :param metadata: Metadata containing the resolution of the image
    :param n_faces: Number of faces of the mesh
    :return: Dictionary containing the vertices, faces and normals of the mesh
    """
    if zoom_factor != 1.0:
        print(f"Downsampling mask by factor {zoom_factor} in X, Y, Z...")
        img = zoom(img, zoom=zoom_factor, order=0)
    else:
        img = img

    img = median_3d_array(img, disk_size=3)
    aux = np.zeros(np.array(img.shape), dtype=np.uint8)
    aux[
        1:-1,
        1:-1,
        1:-1
    ] = img[1:-1, 1:-1, 1:-1]

    #vert, trian = mcubes.marching_cubes(mcubes.smooth(aux), 0)
    vert, trian = mcubes.marching_cubes(aux, 0)
    assert len(vert) > 0 and len(trian) > 0, 'No mesh was generated'


    # vert -= vert.mean(axis=0)
    scale_x = metadata['x_res'] / zoom_factor
    scale_y = metadata['y_res'] / zoom_factor
    scale_z = metadata['z_res'] / zoom_factor
    vert *= np.array([scale_x, scale_y, scale_z])

    mesh = trimesh.Trimesh(vert, trian, process=False)
    print('Number of faces:', len(mesh.faces))
    print('Number of vertices:', len(mesh.vertices))
    #print('Smoothing mesh...')
    #trimesh.smoothing.filter_laplacian(
    #    mesh, lamb=0.6, iterations=10,
    #    volume_constraint=False
    #)

    # mesh = post_process(mesh, n_iters=15)
    faces, vertices = mesh.faces, mesh.vertices
    new_vertices, new_faces = fix_mesh_with_pymeshlab(vertices, faces)
    new_vertices, new_faces = fix_mesh_with_pymeshlab(new_vertices, new_faces)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    #trimesh.smoothing.filter_laplacian(
    #    mesh, lamb=0.6, iterations=5,
    #    volume_constraint=False
    #)

    print('Number of faces:', len(mesh.faces))
    print('Number of vertices:', len(mesh.vertices))
    print(mesh.faces.shape)
    # e.g., (20040, 3) means all triangular.

    all_triangles = np.all([len(set(f)) == 3 for f in mesh.faces])
    print("All triangular faces?", all_triangles)

    normals = mesh.vertex_normals
    return {
        'vertices': mesh.vertices,
        'faces': mesh.faces,
        'normals': normals
    }


def run(img_path, path_out, metadata, n_faces=7500):
    img = imaging.read_image(img_path, axes='XYZ', verbose=1)

    # Convert binary
    img = img > 0

    print('Generating mesh...')
    mesh_data = marching_cubes(img, metadata, n_faces)
    print('Mesh generated')
    assert mesh_data is not None, 'No mesh was generated'

    print('Saving mesh...')
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    normals = mesh_data['normals']

    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ]
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]

    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')
    mesh_ply = PlyData([vertex_element, face_element], text=False, byte_order='<')
    mesh_ply.write(path_out)

    print('Number of faces:', len(faces))
    print('Number of vertices:', len(vertices))
    print('Mesh saved to:', path_out)


import struct

def convert_binary_ply_to_ply2(input_file, output_file):
    with open(input_file, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        face_count = 0
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])

        vertices = []
        for i in range(vertex_count):
            vertex = struct.unpack('<fff', f.read(12))
            vertices.append(vertex)

        faces = []
        for i in range(face_count):
            #face = struct.unpack('<3I', f.read(12))
            #faces.append(face)

            face = struct.unpack('<BIII', f.read(13))
            if face[0] == 3:
                faces.append(face[1:])

        print(f"Read {len(vertices)} vertices and {len(faces)} faces from {input_file}")

    with open(output_file, 'w') as f:
        f.write(f"{len(vertices)}\n")
        f.write(f"{len(faces)}\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        print(f"Converted {input_file} to {output_file} in ply2 format")


def convert_ply2_to_binary_ply(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    vertex_count = int(lines[0].strip())
    face_count = int(lines[1].strip())

    vertices = [list(map(float, line.strip().split())) for line in lines[2:2 + vertex_count]]
    faces = [list(map(int, line.strip().split())) for line in lines[2 + vertex_count:]]

    with open(output_file, 'wb') as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {face_count}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode('utf-8'))

        for vertex in vertices:
            f.write(struct.pack('<fff', *vertex))

        for face in faces:
            if face[0] != 3:
                raise ValueError(f"Non-triangle face detected: {face}")
            f.write(struct.pack('<BIII', face[0], face[1], face[2], face[3]))

    print(f"Converted {input_file} to {output_file} in binary format")


def skeleton(input_file, output_file, n_iters=1, skeleton_path=None):
    import subprocess

    assert input_file.endswith('.ply') and output_file.endswith('.ply'), "Input and output files must be in .ply format"

    if skeleton_path is None:
        skeleton_path = '/home/imarcoss/ht_morphogenesis/meshes/skeleton_approximation/Skeleton'

    skeleton_dir = os.path.dirname(os.path.abspath(skeleton_path))
    qhull_path = os.path.join(skeleton_dir, 'qhull')

    if not os.path.exists(qhull_path) or not os.access(qhull_path, os.X_OK):
        raise FileNotFoundError(f"qhull executable not found or not executable in {skeleton_dir}")

    env = os.environ.copy()
    env["PATH"] = skeleton_dir + os.pathsep + env.get("PATH", "")

    convert_binary_ply_to_ply2(input_file, input_file.replace('.ply', '.ply2'))
    input_file_aux = input_file.replace('.ply', '.ply2')
    output_file_aux = output_file.replace('.ply', '.ply2')

    for i in range(n_iters):
        subprocess.run(
            f'{skeleton_path} {input_file_aux} {output_file_aux}',
            shell=True, check=True, env=env
        )

        convert_ply2_to_binary_ply(output_file_aux, output_file)
        mesh = trimesh.load(output_file, file_type='ply')

        #smoothed = trimesh.smoothing.filter_taubin(
        #    mesh, lamb=0.5, nu=-0.53,
        #    iterations=1
        #)
        smoothed = trimesh.smoothing.filter_laplacian(
            mesh, lamb=0.6, iterations=5,
            volume_constraint=False
        )
        smoothed.export(output_file)

        convert_binary_ply_to_ply2(output_file, output_file_aux)
        input_file_aux = output_file_aux

    print(f"Generated skeleton for {output_file_aux} - {n_iters} iterations")
    convert_ply2_to_binary_ply(output_file_aux, output_file)
    print(f"Converted {output_file} to ply format")


def run_skeleton(input_file, output_file, n_iters=1):
    skeleton(input_file, output_file, n_iters=n_iters)


def main():
    for raw_name in os.listdir(tissue_seg_dir):
        try:
            print(raw_name)
            if not raw_name.endswith('.tif'):
                continue
            #if raw_name != 'CE6_SHF_segmentation.tif':
            #    continue
            if raw_name == "GSU55_9_KO_decon_c1_mask.tif":
                continue
            raw_path = os.path.join(tissue_seg_dir, raw_name)
            out_path = os.path.join(out_dir, raw_name.replace('.tif', '_tissue.ply'))

            print(f'{c.BOLD}Reconstructing{c.ENDC}: {raw_path}')
            run(raw_path, out_path, metadata={
                'x_res': 0.7575758, 'y_res': 0.7575758, 'z_res': 0.9999286
            }, n_faces=7500)

            print(f'{c.OKGREEN}Mesh saved{c.ENDC}: {out_path}')

            # Load final mesh to split into 2 components in memory
            # Then skeletonize each in memory, merge them, and save one final skeleton
            loaded_mesh = trimesh.load(out_path, file_type='ply')
            comps = loaded_mesh.split(only_watertight=False)
            for com in comps:
                print(f"Component: {len(com.faces)} faces")

            submeshes = sorted(comps, key=lambda x: len(x.faces), reverse=True)
            print(f"{c.OKGREEN}Split into components{c.ENDC}: {len(comps)}")

            if len(submeshes) > 2:
                submeshes = submeshes[:2]
                print(f"{c.WARNING}WARNING: More than 2 components found. Keeping the 2 largest by face count.{c.ENDC}")
            if len(submeshes) < 2:
                print(f"{c.WARNING}WARNING: Fewer than 2 components found. Skeletonizing what we have.{c.ENDC}")

            # Skeletonize each submesh to produce two skeleton submeshes (in memory)
            import tempfile
            skeleton_submeshes = []
            for i, subm in enumerate(submeshes):
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_in:
                    sub_in = tmp_in.name
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_out:
                    sub_out = tmp_out.name

                # Save submesh to sub_in
                #subm = post_process(subm, n_iters=10)

                v = subm.vertices
                f = subm.faces
                vertex_dtype = [
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ]
                vert_data = np.empty(len(v), dtype=vertex_dtype)
                vert_data['x'] = v[:, 0]
                vert_data['y'] = v[:, 1]
                vert_data['z'] = v[:, 2]

                face_dtype = [('vertex_indices', 'i4', (3,))]
                face_data = np.empty(len(f), dtype=face_dtype)
                face_data['vertex_indices'] = f

                ve = PlyElement.describe(vert_data, 'vertex')
                fe = PlyElement.describe(face_data, 'face')
                PlyData([ve, fe], text=False, byte_order='<').write(sub_in)

                # Skeletonize
                run_skeleton(sub_in, sub_out, n_iters=1)

                # Load submesh skeleton
                skeleton_mesh = trimesh.load(sub_out, file_type='ply')
                skeleton_submeshes.append(skeleton_mesh)

                # Clean up temp files
                os.remove(sub_in)
                os.remove(sub_out)

            # Merge the 2 skeleton submeshes
            if skeleton_submeshes:
                final_skel = trimesh.util.concatenate(skeleton_submeshes)
                out_skeleton_path = os.path.join(
                    out_skeleton_dir,
                    raw_name.replace('.tif', '_tissue_skeleton.ply')
                )
                final_skel.export(out_skeleton_path)
                print(f"{c.OKGREEN}Skeleton saved{c.ENDC}: {out_skeleton_path}")
            else:
                print(f"{c.WARNING}No submeshes to skeletonize, skipping final merge.{c.ENDC}")
        except Exception as e:
            print(f"{c.FAIL}Error processing {raw_name}: {e}{c.ENDC}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
