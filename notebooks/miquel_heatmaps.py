#!/usr/bin/env python3

import os
import sys
import numpy as np
import trimesh
import pyrender
from PIL import Image
import matplotlib.pyplot as plt

############################################
# Camera & Mesh Utility Functions (from snippet)
############################################

def compute_camera_pose(mesh, direction):
    """
    Compute the camera pose based on the mesh's bounding box and a direction vector.
    Positions the camera at 'center + direction * size * 0.6'
    and returns a 4x4 view matrix (world->camera).
    """
    bounds = mesh.bounds           # min/max corners
    center = mesh.bounding_box.centroid
    size = np.linalg.norm(bounds[1] - bounds[0])  # diagonal length

    camera_position = center + direction * size * 0.6

    # 'forward' vector points from camera to center
    forward = (center - camera_position)
    forward /= np.linalg.norm(forward)

    # default 'up_vector'
    up_vector = np.array([0, 1, 0], dtype=float)
    # watch out for edge cases where forward is parallel to up
    if np.allclose(forward, up_vector) or np.allclose(forward, -up_vector):
        up_vector = np.array([0, 0, 1], dtype=float)

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # camera looks along -Z
    pose[:3, 3] = camera_position
    return pose


def rotate_mesh(mesh, angles):
    """
    Rotate the mesh by given angles (radians) around X, Y, Z axes.
    angles: (angle_x, angle_y, angle_z)
    """
    import trimesh.transformations as tf
    rx, ry, rz = angles
    if abs(rx) > 1e-8:
        m = tf.rotation_matrix(rx, [1, 0, 0])
        mesh.apply_transform(m)
    if abs(ry) > 1e-8:
        m = tf.rotation_matrix(ry, [0, 1, 0])
        mesh.apply_transform(m)
    if abs(rz) > 1e-8:
        m = tf.rotation_matrix(rz, [0, 0, 1])
        mesh.apply_transform(m)
    return mesh


############################################
# Rendering Logic
############################################

def render_single_view(
    mesh: trimesh.Trimesh,
    view_name: str,
    direction: np.ndarray,
    rotation_angles=(0,0,0),
    img_res=(1024, 768)
):
    """
    Renders 'mesh' from a given 'direction' with optional Euler rotations,
    returning a PIL Image. The mesh is assumed to be pre-colored (face_colors).
    """
    # Make a copy so we can rotate without affecting the original
    mesh_copy = mesh.copy()
    # Apply the specified rotation angles
    mesh_copy = rotate_mesh(mesh_copy, rotation_angles)

    # Convert to pyrender mesh
    try:
        # smooth=False keeps face colors distinct if you used per-face coloring
        mesh_pyr = pyrender.Mesh.from_trimesh(mesh_copy, smooth=False)
    except ValueError as e:
        print(f"Error creating a pyrender mesh: {e}")
        return None

    # Create a scene
    scene = pyrender.Scene()
    scene.add(mesh_pyr)

    # Create and add a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    pose = compute_camera_pose(mesh_copy, direction)
    scene.add(camera, pose=pose)

    # Basic directional light from the same pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=pose)

    # Also an ambient light at origin
    ambient = pyrender.PointLight(color=np.ones(3), intensity=0.8)
    scene.add(ambient, pose=np.eye(4))

    # Render off-screen
    r = pyrender.OffscreenRenderer(viewport_width=img_res[0], viewport_height=img_res[1])
    color, depth = r.render(scene)
    r.delete()

    # If the rendered image is all black or empty, warn
    if not np.any(color):
        print(f"Warning: {view_name} view is empty. Possibly need to adjust camera.")
    return Image.fromarray(color)


def render_mesh_three_views(mesh_path, out_folder, specimen_name):
    """
    Loads a pre-colored mesh from 'mesh_path',
    renders it from 3 views (cardinal, dorsal, ventral),
    and saves each as a PNG in 'out_folder/<specimen_name>/'.
    """
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        return

    # Load the (already colored) mesh
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # If it loads as a Scene, combine geometries
        if hasattr(mesh, 'geometry'):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        else:
            print(f"Cannot interpret {mesh_path} as a single mesh.")
            return

    # Folder for this specimen
    specimen_folder = os.path.join(out_folder, specimen_name)
    os.makedirs(specimen_folder, exist_ok=True)

    # Define your 3 viewpoints
    # You can tweak direction vectors & rotation angles to match your definitions
    views = {
        "cardinal": {
            "direction": np.array([0, 0, -1], dtype=float),
            "rotation": (0, 3 * np.pi / 2, np.pi)
        },
        "dorsal": {
            "direction": np.array([0, -1, -1], dtype=float),
            "rotation": (3*np.pi / 2, np.pi / 2, 3*np.pi / 2)
        },
        "ventral": {
            "direction": np.array([0, -1, 0], dtype=float),
            "rotation": (0, np.pi / 2, np.pi)
        },
    }

    # For each view, render & save
    for view_name, params in views.items():
        direction = params["direction"]
        angles = params.get("rotation", (0,0,0))

        img = render_single_view(
            mesh=mesh,
            view_name=view_name,
            direction=direction,
            rotation_angles=angles,
            img_res=(1024, 768),
        )
        if img is None:
            continue

        png_name = f"{specimen_name}_{view_name}.png"
        png_path = os.path.join(specimen_folder, png_name)
        img.save(png_path)
        print(f"Saved: {png_path}")


############################################
# (Optional) Mosaic: combine all specimens
############################################

def create_mosaic_of_specimens(
    out_folder,      # same folder containing subfolders => one folder per specimen
    specimen_names,  # list of specimen folder names
    views=["cardinal", "dorsal", "ventral"],
    mosaic_name="all_specimens_views.png"
):
    """
    Creates a single PNG with rows = the 3 views, columns = the specimens.
    Looks for <specimen>/<specimen>_<view>.png in out_folder.
    Saves mosaic in out_folder as 'all_specimens_views.png'.
    """
    # Collect images
    n_rows = len(views)
    n_cols = len(specimen_names)

    # We'll store each image in a 2D list
    images_grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for col, spec_name in enumerate(specimen_names):
        spec_folder = os.path.join(out_folder, spec_name)
        for row, vw in enumerate(views):
            png_file = os.path.join(spec_folder, f"{spec_name}_{vw}.png")
            if os.path.exists(png_file):
                images_grid[row][col] = Image.open(png_file)
            else:
                images_grid[row][col] = None

    # Figure out how wide/tall each cell is (you can unify or set fixed)
    cell_w, cell_h = 0, 0
    for row in range(n_rows):
        for col in range(n_cols):
            img = images_grid[row][col]
            if img is not None:
                w, h = img.size
                cell_w = max(cell_w, w)
                cell_h = max(cell_h, h)

    # Build the final mosaic
    mosaic_w = cell_w * n_cols
    mosaic_h = cell_h * n_rows
    mosaic_img = Image.new("RGB", (mosaic_w, mosaic_h), color=(255,255,255))

    # Paste images into mosaic
    for row in range(n_rows):
        for col in range(n_cols):
            img = images_grid[row][col]
            if img is not None:
                x_off = col * cell_w
                y_off = row * cell_h
                mosaic_img.paste(img, (x_off, y_off))

    # Save mosaic
    mosaic_path = os.path.join(out_folder, mosaic_name)
    mosaic_img.save(mosaic_path)
    print(f"Mosaic saved: {mosaic_path}")


############################################
# Main script logic
############################################

def main():
    """
    1) We look for .ply meshes in 'colored_mesh_dir'.
    2) For each .ply => interpret the filename (or part before .ply) as specimen_name.
    3) Render cardinal, dorsal, ventral => save to 'png_out_dir/<specimen_name>/'.
    4) Optionally build a single mosaic with all specimens in columns, views in rows.
    """

    # Where the colored .ply files live
    colored_mesh_dir = (
        "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,"
        "share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/2025/results/columnarity"
    )
    # Where we store the final PNGs
    png_out_dir = (
        "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,"
        "share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/2025/results/heatmaps"
    )

    # 1) Gather all .ply in 'colored_mesh_dir'
    all_ply_files = [f for f in os.listdir(colored_mesh_dir) if f.endswith(".ply")]

    specimen_names = []
    for ply_name in all_ply_files:
        # Specimen name = "whatever is before .ply"
        specimen_name = os.path.splitext(ply_name)[0]
        # Render
        mesh_path = os.path.join(colored_mesh_dir, ply_name)
        render_mesh_three_views(mesh_path, png_out_dir, specimen_name)
        specimen_names.append(specimen_name)

    # 2) Build a single mosaic with columns=specimens, rows=(cardinal,dorsal,ventral)
    # If you do not want a single mosaic, just comment out the following lines
    # Sort specimen_names for a consistent order
    specimen_names.sort()
    create_mosaic_of_specimens(
        out_folder=png_out_dir,
        specimen_names=specimen_names,
        views=["cardinal", "dorsal", "ventral"],
        mosaic_name="all_specimens_views.png"
    )


if __name__ == "__main__":
    main()
