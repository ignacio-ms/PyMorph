import numpy as np
import trimesh
import pyrender

import json

from auxiliary import values as v

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import json


def main():
    # Load your data paths and meshes
    gr = 'Gr1'
    specimens = v.specimens[gr]
    spec_idx = 0
    # s = specimens[spec_idx]
    s = '0806_E5'

    # myo_path = v.data_path + f'{gr}/3DShape/Tissue/myocardium/2019{s}Shape.ply'
    myo_path = v.data_path + f'ATLAS/myocardium/ATLAS_{gr}.ply'
    myo_mesh = o3d.io.read_triangle_mesh(myo_path)
    myo_mesh.compute_vertex_normals()
    myo_mesh.paint_uniform_color([1.0, 0.0, 0.0])

    # Load landmarks
    # land_path = v.data_path + f'Landmarks/2019{s}_key_points.json'
    land_path = v.data_path + f'Landmarks/ATLAS/ATLAS_{gr}_key_points.json'
    with open(land_path, 'r') as f:
        key_points = json.load(f)
        myo_landmarks = {
            name: key_points[name]
            for name in v.myo_myo_landmark_names
            if name in key_points
        }

    app = gui.Application.instance
    app.initialize()

    # Create a window
    window = app.create_window("Landmark Viewer", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    # Set background color (optional)
    scene.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background

    # Add meshes to the scene
    material = rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1.0, 0.0, 0.0, 0.5]  # Red color
    scene.scene.add_geometry("myo_mesh", myo_mesh, material)
    # Add spl_mesh if needed

    # Set up the camera
    bbox = myo_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent().max()
    scene.scene.camera.look_at(
        center,
        center + [0, 0, extent],
        [0, 1, 0]
    )

    # Create labels for landmarks
    labels = {}
    material.shader = "defaultLitTransparency"
    material.base_color = [0.0, 0.0, 0.0, 1.0]  # Black color
    for name, point in myo_landmarks.items():
        # Add sphere at the landmark position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.compute_vertex_normals()
        sphere.translate(point)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        scene.scene.add_geometry(name, sphere, material)

        # Create a label
        label = o3d.t.geometry.TriangleMesh.create_text(text=name, depth=0.1)
        # label.paint_uniform_color([0.0, 0.0, 0.0])
        label.translate(point)
        scene.scene.add_geometry(f'{name}_label', label, material)

        # label = gui.Label(name)
        # label.visible = True
        labels[name] = {'label': label, 'point': np.array(point)}
        # window.add_child(label)

    # Function to update label positions
    def update_labels():
        # Get the camera matrices
        cam = scene.scene.camera
        # View matrix
        view_matrix = np.asarray(cam.get_view_matrix()).reshape((4, 4)).T
        # Projection matrix
        proj_matrix = np.asarray(cam.get_projection_matrix()).reshape((4, 4)).T
        # Viewport dimensions
        viewport = window.content_rect
        width = viewport.width
        height = viewport.height

        for info in labels.values():
            point = info['point']
            label = info['label']

            # Transform the point to clip space
            p = np.append(point, 1.0)  # Convert to homogeneous coordinates
            # Transform to camera space
            camera_space = view_matrix @ p
            # Transform to clip space
            clip_space = proj_matrix @ camera_space

            # Check for points behind the camera
            if clip_space[3] == 0:
                continue  # Cannot project

            # Perform perspective division to get NDC
            ndc = clip_space[:3] / clip_space[3]

            # Check if point is within NDC bounds (-1 to 1)
            if np.any(ndc < -1) or np.any(ndc > 1):
                label.visible = False  # Point is outside the view frustum
                continue
            else:
                label.visible = True

            # Map NDC to screen coordinates
            x = int((ndc[0] * 0.5 + 0.5) * width)
            y = int((1 - (ndc[1] * 0.5 + 0.5)) * height)

            # Set label position (adjust as needed)
            label.frame = gui.Rect(x, y, 100, 20)

    # Callback to update labels when the camera changes
    def on_camera_changed():
        update_labels()

    def on_layout(layout_context):
        update_labels()

    # scene.set_on_post_draw(on_post_draw)
    # window.add_child(scene)

    # window.set_on_layout(on_layout)

    window.add_child(scene)
    app.run()

if __name__ == '__main__':
    main()
