import getopt
import os
import re
import sys

import numpy as np
import pandas as pd
import pyrender
import trimesh
import trimesh.viewer
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LogNorm

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

# Subtract one folder in the current dir
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from utils.misc.bash import arg_check
from utils import values as v


def compute_camera_pose(mesh, direction):
    """
    Compute the camera pose based on the mesh's bounding box and a direction vector.
    """
    bounds = mesh.bounds
    center = mesh.bounding_box.centroid
    size = np.linalg.norm(bounds[1] - bounds[0])

    # Position the camera along the given direction
    camera_position = center + direction * size * .6

    # Compute a look-at transform
    forward = (center - camera_position)
    forward /= np.linalg.norm(forward)

    # Use a fallback vector to avoid alignment issues
    up_vector = np.array([0, 1, 0])
    if np.allclose(forward, up_vector) or np.allclose(forward, -up_vector):
        up_vector = np.array([0, 0, 1])

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def rotate_mesh(mesh, angles):
    """
    Rotate the mesh by given angles (in radians) around X, Y, and Z axes.
    :param mesh: The mesh to rotate.
    :param angles: A tuple (angle_x, angle_y, angle_z) with rotation angles in radians.
    """
    rotation_axes = [
        (angles[0], [1, 0, 0]),  # Rotation around X-axis
        (angles[1], [0, 1, 0]),  # Rotation around Y-axis
        (angles[2], [0, 0, 1]),  # Rotation around Z-axis
    ]
    for angle, axis in rotation_axes:
        if angle != 0:
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)
    return mesh


def save_mesh_views(input_folder):
    """
    Reads all .ply files from the input folder and generates ventral, dorsal, and caudal views
    for each mesh. Saves the images in a folder structure where each group has its own subfolder,
    and images for different views are saved within it.
    """
    output_folder = os.path.join(input_folder, "output_images")
    os.makedirs(output_folder, exist_ok=True)

    # Define the view directions and rotations (angle_x, angle_y, angle_z)
    views = {
        "ventral": {
            "direction": np.array([0, 0, -1]),
            "rotation": (0, 3 * np.pi / 2, np.pi)
        },
        "dorsal": {
            "direction": np.array([0, -1, -1]),
            "rotation": (3*np.pi / 2, np.pi / 2, 3*np.pi / 2)
        },
        "caudal": {
            "direction": np.array([0, -1, 0]),
            "rotation": (0, np.pi / 2, np.pi)
        },
    }

    for filename in os.listdir(input_folder):
        if filename.endswith(".ply"):
            filepath = os.path.join(input_folder, filename)
            mesh = trimesh.load(filepath)

            # Create a subfolder for the group based on the filename
            group_name = os.path.splitext(filename)[0]  # Use the filename without extension as group name
            group_folder = os.path.join(output_folder, group_name)
            os.makedirs(group_folder, exist_ok=True)

            for view_name, params in views.items():
                # Rotate the mesh for the specific view
                rotated_mesh = mesh.copy()
                angles = params.get("rotation", (0, 0, 0))  # Default to no rotation
                rotated_mesh = rotate_mesh(rotated_mesh, angles)

                # Convert the rotated mesh to pyrender format
                try:
                    mesh_pyrender = pyrender.Mesh.from_trimesh(rotated_mesh)
                except ValueError as e:
                    print(f"Error converting {filename} to pyrender mesh: {e}")
                    continue

                # Create a scene
                scene = pyrender.Scene()
                scene.add(mesh_pyrender)

                # Add a camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                camera_pose = compute_camera_pose(rotated_mesh, params["direction"])
                scene.add(camera, pose=camera_pose)

                # Add lights
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(light, pose=camera_pose)

                ambient_light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
                scene.add(ambient_light, pose=np.eye(4))  # At the origin

                # Render the scene
                renderer = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=768)
                color, _ = renderer.render(scene)

                # Check if the rendered image is empty
                if not np.any(color):
                    print(f"Warning: {view_name} view of {filename} is empty. Adjusting camera.")
                    continue

                img = Image.fromarray(color)
                if view_name == "dorsal":
                    img = img.rotate(180, expand=True)

                # Save the rotated image in the group folder
                output_path = os.path.join(group_folder, f"{group_name}_{view_name}.png")
                img.save(output_path)

                # Clean up
                renderer.delete()

    print(f"Images saved to {output_folder}")


def create_feature_grid(input_folder, output_folder, feature_name):
    """
    Creates a grid plot for the feature with group-specific colorbars for non-normalized features.
    Dynamically assesses cmap, norm, f_min, and f_max for each group.
    :param input_folder: Folder containing the images for the feature.
    :param output_folder: Folder to save the grid plot.
    :param feature_name: Name of the feature (used as the title of the grid).
    """
    # Determine if features are normalized based on the input folder path
    is_normalized = "FeaturesNormalized" in input_folder

    # Define colormap
    colors = [
        (0, 0, 1),  # Pure blue
        (0, 0.5, 1),  # Cyan-like
        (0, 1, 0),  # Green
        (1, 1, 0),  # Yellow
        (1, 0, 0),  # Red
    ]
    cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=1024)

    # Prepare groups and view names
    groups = sorted(
        [name for name in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, name))]
    )
    views = ["ventral", "dorsal", "caudal"]

    # Number of views (rows) and groups (columns)
    n_rows = len(views)
    n_cols = len(groups)

    # Prepare the grid plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(feature_name, fontsize=20, y=1.02)

    # Track colorbars for each group
    group_colorbars = {}

    for col, group in enumerate(groups):
        # Determine f_min and f_max for the current group from its CSV
        csv_path_split = input_folder.split('/')
        csv_path_split = csv_path_split[:-1]
        csv_path = '/'.join(csv_path_split) + f'/{group}.csv'

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df['value'].notna()]
            f_min, f_max = np.percentile(df['value'], 0.1), np.percentile(df['value'], 99.9)

            if f_max > 10:
                f_max = np.percentile(df['value'], 95)
                f_min = np.percentile(df['value'], 5)
                print(f"Feature values (clipped) for {group}: {f_min} - {f_max}")

            # Define normalization for the current group
            norm = BoundaryNorm(
                boundaries=np.linspace(f_min, f_max, cmap.N),
                ncolors=cmap.N
            )

            group_colorbars[group] = (norm, f_min, f_max)
        else:
            print(f"Warning: CSV file not found for {group} - Skipping")
            continue

        for row, view in enumerate(views):
            img_path = os.path.join(input_folder, group, f"{group}_{view}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[row, col].imshow(img)
            else:
                axes[row, col].text(
                    0.5,
                    0.5,
                    "Missing Image",
                    color="red",
                    fontsize=12,
                    ha="center",
                    va="center",
                )
            axes[row, col].axis("off")

            # Add column title (group name)
            if row == 0:
                axes[row, col].set_title(group.split('_')[0], fontsize=18)

            # Add row label (view name)
            if col == 0:
                axes[row, col].text(
                    -0.5,
                    0.5,
                    view.capitalize() + " View",
                    fontsize=20,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=axes[row, col].transAxes,
                )

    # Add a colorbar for each group below its column
    for col, group in enumerate(groups):
        if group in group_colorbars:
            norm, f_min, f_max = group_colorbars[group]
            cbar_ax = fig.add_axes([
                axes[-1, col].get_position().x0,  # Align with the column's x-position
                axes[-1, col].get_position().y0 - 0.1,  # Slightly below the last row
                axes[-1, col].get_position().width,  # Match the width of the column
                0.02,  # Height of the colorbar
            ])
            cb = plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation="horizontal"
            )
            cb.set_label(f"{f_min:.2f} - {f_max:.2f}", fontsize=12)
            cb.ax.tick_params(labelsize=8)

    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)  # Increased bottom space for colorbars

    # Save the grid plot
    output_path = os.path.join(output_folder, f"{feature_name}_grid.png")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grid saved at {output_path}")


def save_mesh_views_for_group(group_folder, feature_name):
    """
    For each embryo subfolder in 'group_folder', render ventral, dorsal, and caudal views,
    saving them in a subfolder named 'rendered_views' inside each embryo folder.
    """
    # Define the view directions and rotations (angle_x, angle_y, angle_z)
    views = {
        "ventral": {
            "direction": np.array([0, 0, -1]),
            "rotation": (0, 3 * np.pi / 2, np.pi)
        },
        "dorsal": {
            "direction": np.array([0, -1, -1]),
            "rotation": (3*np.pi / 2, np.pi / 2, 3*np.pi / 2)
        },
        "caudal": {
            "direction": np.array([0, -1, 0]),
            "rotation": (0, np.pi / 2, np.pi)
        },
    }

    # List all subfolders in group_folder (each subfolder = an embryo)
    embryo_names = [
        f for f in os.listdir(group_folder)
        if os.path.isdir(os.path.join(group_folder, f))
    ]

    embryo_names = [e for e in embryo_names if e in v.specimens_to_analyze]

    for embryo_name in embryo_names:
        embryo_path = os.path.join(group_folder, embryo_name)

        # Look for a .ply file in the embryo folder:
        ply_files = [f for f in os.listdir(embryo_path) if f.endswith(".ply") and re.search(feature_name, f)]
        if not ply_files:
            print(f"No .ply found in {embryo_path}, skipping.")
            continue

        # For simplicity, assume there's only one .ply or we pick the first:
        ply_file = ply_files[0]
        mesh_path = os.path.join(embryo_path, ply_file)
        mesh = trimesh.load(mesh_path)

        # Create an output subfolder for images inside embryo folder
        output_views_folder = os.path.join(embryo_path, "rendered_views")
        os.makedirs(output_views_folder, exist_ok=True)

        # Render each view
        for view_name, params in views.items():
            # Make a copy of the original mesh
            rotated_mesh = mesh.copy()
            angles = params.get("rotation", (0, 0, 0))
            rotated_mesh = rotate_mesh(rotated_mesh, angles)

            # Convert to a pyrender mesh
            try:
                mesh_pyrender = pyrender.Mesh.from_trimesh(rotated_mesh, smooth=False)
            except ValueError as e:
                print(f"Error converting {mesh_path} to pyrender mesh: {e}")
                continue

            # Create a scene
            scene = pyrender.Scene()
            scene.add(mesh_pyrender)

            # Add a camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = compute_camera_pose(rotated_mesh, params["direction"])
            scene.add(camera, pose=camera_pose)

            # Lights
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)
            ambient_light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
            scene.add(ambient_light, pose=np.eye(4))

            # Render offscreen
            renderer = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=768)
            color, _ = renderer.render(scene)

            if not np.any(color):
                print(f"Warning: {view_name} view of {mesh_path} is empty. Adjust camera.")
                renderer.delete()
                continue

            img = Image.fromarray(color)
            if view_name == "dorsal":
                img = img.rotate(180, expand=True)

            # Save image: embryoName_view.png
            output_path = os.path.join(output_views_folder, f"{embryo_name}_{view_name}.png")
            img.save(output_path)

            renderer.delete()

    print(f"Finished rendering views for group folder: {group_folder}")


def create_grid_for_group(group_folder, feature_name):
    """
    Creates a grid of ventral/dorsal/caudal views for each embryo in this group.
    Saves the result as a single .png.
    """
    views = ["ventral", "dorsal", "caudal"]
    embryo_names = [
        f for f in os.listdir(group_folder)
        if os.path.isdir(os.path.join(group_folder, f))
    ]
    embryo_names.sort()  # So the columns are in a consistent order

    n_rows = len(views)
    n_cols = len(embryo_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    fig.suptitle(f"{os.path.basename(group_folder)} - {feature_name}", fontsize=16)

    for col, embryo_name in enumerate(embryo_names):
        # The images should be in 'rendered_views' subfolder
        rendered_views_folder = os.path.join(group_folder, embryo_name, "rendered_views")

        for row, view in enumerate(views):
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes
            img_filename = f"{embryo_name}_{view}.png"
            img_path = os.path.join(rendered_views_folder, img_filename)

            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing Image", color="red",
                        fontsize=12, ha="center", va="center")
            ax.axis("off")

            # Row label (view)
            if col == 0:
                ax.set_ylabel(view.capitalize(), fontsize=14)

            # Column label (embryo name)
            if row == 0:
                ax.set_title(embryo_name, fontsize=14)

    plt.tight_layout()

    # Save grid
    output_path = os.path.join(group_folder, f"{os.path.basename(group_folder)}_{feature_name}_grid.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Grid for group {os.path.basename(group_folder)} saved at {output_path}")


def run_all_groups(groups_root_folder, feature_name):
    """
    Suppose 'groups_root_folder' contains subfolders: Gr1, Gr2, etc.
    Each group folder has multiple embryo subfolders.
    """
    group_names = list(v.specimens.keys())
    group_names.sort()

    for group_name in group_names:
        if group_name == 'Gr10' or group_name == 'Gr11':
            continue

        group_folder = os.path.join(groups_root_folder, group_name + '/3DShape/Tissue/myocardium/map/')

        save_mesh_views_for_group(group_folder, feature_name)
        create_grid_for_group(group_folder, feature_name)


def run(input_folder, output_folder, feature_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_mesh_views(input_folder)
    create_feature_grid(input_folder + 'output_images', output_folder, feature_name)


def print_usage():
    print('TODO...')
    sys.exit(2)


# if __name__ == '__main__':
#     feature_name = 'cell_division'
#     input_folder = v.data_path
#
#     run_all_groups(input_folder, feature_name)


if __name__ == "__main__":
    argv = sys.argv[1:]

    data_path = v.data_path
    tissue = 'myocardium'
    level = 'Membrane'
    feature = None
    norm = False
    verbose = 1

    try:
        opts, args = getopt.getopt(argv, 'hp:t:l:f:n:v:', [
            'help', 'path=', 'tissue=', 'level=', 'feature=', 'norm=', 'verbose='
        ])

        if len(opts) > 5:
            print_usage()

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print_usage()
            elif opt in ('-p', '--path'):
                data_path = arg_check(opt, arg, '-p', '--path', str, print_usage)
            elif opt in ('-t', '--tissue'):
                tissue = arg_check(opt, arg, '-t', '--tissue', str, print_usage)
            elif opt in ('-l', '--level'):
                level = arg_check(opt, arg, '-l', '--level', str, print_usage)
            elif opt in ('-f', '--feature'):
                feature = arg_check(opt, arg, '-f', '--feature', str, print_usage)
            elif opt in ('-n', '--norm'):
                norm = arg_check(opt, arg, '-n', '--norm', bool, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print_usage()

        if feature is None:
            print(f'No feature provided')
            sys.exit(2)

        if norm:
            input_folder = data_path + f'ATLAS/{tissue}/FeaturesNormalized/{level}/{feature}/'
        else:
            input_folder = data_path + f'ATLAS/{tissue}/Features/{level}/{feature}/'

        output_folder = input_folder + 'grid/'
        run(input_folder, output_folder, feature)

        if verbose > 0:
            print(f'Feature grid created for {feature} in {tissue} at {level} level.')

    except getopt.GetoptError:
        print_usage()
