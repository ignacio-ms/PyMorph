import trimesh
import numpy as np

from PIL import Image as PILImage
from IPython.display import Image as IPyImage
from scipy.spatial import cKDTree

from pyglet.gl import *
import pyrender


def look_at(eye, target, up):
    """
    Create a look-at view matrix.

    Parameters:
    - eye: (3,) array-like, camera position
    - target: (3,) array-like, point camera is looking at
    - up: (3,) array-like, up direction

    Returns:
    - view_matrix: (4,4) numpy array
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    rotation = np.eye(4)
    rotation[0, :3] = right
    rotation[1, :3] = true_up
    rotation[2, :3] = -forward  # Negative forward for right-handed system

    translation = np.eye(4)
    translation[:3, 3] = -eye

    view_matrix = rotation @ translation
    return view_matrix


class CellVisualization:
    def __init__(self, tissue_mesh, cell_mesh, centroid, dynamic_camera=False):
        """
        Initialize the CellVisualization with tissue and cell meshes and their centroid.

        Parameters:
        - tissue_mesh: trimesh.Trimesh, the tissue mesh
        - cell_mesh: trimesh.Trimesh, the cell mesh
        - centroid: (3,) array-like, the centroid of the cell
        """
        # Initialize the scene
        self.scene = pyrender.Scene()

        # Add tissue mesh
        self.add_mesh(
            mesh=tissue_mesh,
            color=[0.8, 0.8, 0.8, .75],  # Light gray with transparency
            name='Tissue Mesh',
            is_transparent=True
        )

        # Add cell mesh
        self.add_mesh(
            mesh=cell_mesh,
            color=[0.0, 0.5, 1.0, .5],  # Blue
            name='Cell Mesh',
            is_transparent=True
        )

        # Add centroid as a red sphere
        self.add_sphere(
            center=centroid,
            radius=1,
            color=[1.0, 0.0, 0.0, 1.0],  # Red
            name='Centroid'
        )

        # Initialize camera
        self.centroid = centroid
        self.initialize_camera(dynamic_camera)

        # Initialize lighting
        self.add_lights()

    def add_mesh(self, mesh, color, name, is_transparent=False):
        """
        Add a mesh to the scene with specified color and name.

        Parameters:
        - mesh: trimesh.Trimesh, the mesh to add
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the mesh
        """
        alpha_mode = 'BLEND' if is_transparent else 'OPAQUE'

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode=alpha_mode
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(
            mesh, smooth=False,
            material=material
        )
        self.scene.add(pyrender_mesh, name=name)

    def add_sphere(self, center, radius, color, name):
        """
        Add a sphere to the scene at a specified center with given radius and color.

        Parameters:
        - center: (3,) array-like, position of the sphere center
        - radius: float, radius of the sphere
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the sphere
        """
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(center)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.0
        )
        pyrender_sphere = pyrender.Mesh.from_trimesh(sphere, smooth=False, material=material)
        self.scene.add(pyrender_sphere, name=name)

    def initialize_camera(self, dynamic=False):
        """
        Initialize the camera in the scene.
        """
        if dynamic:
            all_meshes = [node.mesh for node in self.scene.mesh_nodes]
            all_vertices = np.vstack([mesh.primitives[0].positions for mesh in all_meshes if mesh.primitives])
            if len(all_vertices) == 0:
                all_vertices = np.array([self.centroid])
            center = all_vertices.mean(axis=0)
            max_extent = np.max(np.linalg.norm(all_vertices - center, axis=1))

            # Define camera parameters
            eye_distance = max_extent  # Position the camera at a distance proportional to the scene size
            eye = center + np.array([eye_distance, eye_distance, eye_distance])
            target = center
            up = np.array([0.0, 0.0, 1.0])

        else:
            # Define camera parameters
            eye = self.centroid + np.array([70.0, 70.0, 70.0])
            target = self.centroid
            up = np.array([0.0, 0.0, 1.0])

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        view_matrix = look_at(eye, target, up)
        camera_pose = np.linalg.inv(view_matrix)  # Invert to get camera pose
        self.scene.add(camera, pose=camera_pose, name='Camera')

    def add_lights(self):
        """
        Add lighting to the scene.
        """
        # Add ambient light
        # ambient_light = pyrender.AmbientLight(color=np.ones(3), intensity=0.5)
        # self.scene.add(ambient_light, name='Ambient Light')

        # Add directional light
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light_pose = look_at(eye=[10.0, -10.0, 10.0], target=self.centroid, up=[0.0, 0.0, 1.0])
        self.scene.add(directional_light, pose=light_pose, name='Directional Light')

    def add_closest_face_centroid(self, face_centroid):
        """
        Add the closest face centroid as a green sphere.

        Parameters:
        - face_centroid: (3,) array-like, position of the closest face centroid
        """
        self.add_sphere(
            center=face_centroid,
            radius=1,
            color=[0.0, 1.0, 0.0, 1.0],  # Green
            name='Closest Face Centroid'
        )

    def add_neighborhood_points(self, neighborhood_points, max_points=100):
        """
        Add neighborhood points as small cyan spheres.

        Parameters:
        - neighborhood_points: (N, 3) array-like, positions of neighborhood points
        - max_points: int, maximum number of points to visualize
        """
        # Limit the number of points to prevent clutter
        if len(neighborhood_points) > max_points:
            indices = np.random.choice(len(neighborhood_points), size=max_points, replace=False)
            sampled_points = neighborhood_points[indices]
        else:
            sampled_points = neighborhood_points

        for i, point in enumerate(sampled_points):
            self.add_sphere(
                center=point,
                radius=0.8,
                color=[0.0, 1.0, 1.0, 1.0],  # Cyan
                name=f'Neighborhood Point {i}'
            )

    def add_fitted_plane(self, normal, point, size=5.0):
        """
        Add a fitted plane to the scene as a yellow semi-transparent box.

        Parameters:
        - normal: (3,) array-like, normal vector of the plane
        - point: (3,) array-like, a point on the plane
        - size: float, size of the plane
        """
        # Create a plane as a thin box
        plane = trimesh.creation.box(extents=[size, size, 0.1])

        # Align the plane with the normal vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # If normal is parallel to z-axis
            if np.dot(z_axis, normal) < 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            else:
                rotation_matrix = np.eye(4)
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
        plane.apply_transform(rotation_matrix)
        plane.apply_translation(point)

        # Create pyrender mesh
        plane_pyrender = pyrender.Mesh.from_trimesh(
            plane,
            smooth=False,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 1.0, 0.0, 0.5],  # Yellow with transparency
                metallicFactor=0.0,
                roughnessFactor=0.5
            )
        )
        self.scene.add(plane_pyrender, name='Fitted Plane')

    def add_ellipsoid(self, axes, lengths, center, longest_axis_vector, color=[1.0, 0.0, 1.0, 0.3], name='Ellipsoid'):
        """
        Add an approximated ellipsoid to the scene based on PCA or MVEE axes and lengths.
        Additionally, align the ellipsoid based on the provided longest axis vector.

        Parameters:
        - axes: (3,3) array-like, principal axes from PCA or MVEE
        - lengths: (3,) array-like, lengths (radii) of the ellipsoid
        - center: (3,) array-like, center of the ellipsoid
        - longest_axis_vector: (3,) array-like, vector defining the longest axis direction
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the ellipsoid
        """
        # Ensure the longest_axis_vector is normalized
        longest_axis_vector = longest_axis_vector / np.linalg.norm(longest_axis_vector)

        # Sort axes and lengths in descending order of lengths to ensure the first axis is the longest
        sorted_indices = np.argsort(lengths)[::-1]
        sorted_axes = axes[sorted_indices]
        sorted_lengths = lengths[sorted_indices]

        # Align the first principal axis with the longest_axis_vector
        default_axis = sorted_axes[0]  # This is the longest axis
        rotation_axis = np.cross(default_axis, longest_axis_vector)
        norm = np.linalg.norm(rotation_axis)
        if norm < 1e-6:
            # Axes are already aligned or opposite; determine the rotation angle
            dot_product = np.dot(default_axis, longest_axis_vector)
            if dot_product > 0:
                # No rotation needed
                rotation_matrix = np.eye(4)
            else:
                # 180-degree rotation around any perpendicular axis
                # Find a vector perpendicular to default_axis
                perp_vector = np.array([1, 0, 0]) if not np.allclose(default_axis, [1, 0, 0]) else np.array([0, 1, 0])
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.pi, perp_vector
                )
        else:
            rotation_axis /= norm
            angle = np.arccos(np.clip(np.dot(default_axis, longest_axis_vector), -1.0, 1.0))
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)

        # Apply rotation to all axes
        rotated_axes = sorted_axes.copy()
        rotated_axes = rotation_matrix[:3, :3] @ rotated_axes

        # Create a unit sphere
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

        # Apply scaling along each principal axis
        scaled_sphere = sphere.copy()
        scaling_matrix = np.diag(sorted_lengths.tolist() + [1.0])
        scaled_sphere.apply_transform(scaling_matrix)

        # Apply rotation to align with rotated_axes
        rotation_full = np.eye(4)
        rotation_full[:3, :3] = rotated_axes
        scaled_sphere.apply_transform(rotation_full)

        # Translate to the center
        scaled_sphere.apply_translation(center)

        # Create pyrender mesh with transparency
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode='BLEND',
            wireframe=True
        )
        pyrender_ellipsoid = pyrender.Mesh.from_trimesh(
            scaled_sphere,
            smooth=False,
            material=material,
            wireframe=True
        )
        self.scene.add(pyrender_ellipsoid, name=name)

    def add_longest_axis(self, center, direction, length=20.0, color=[0.0, 0.0, 0.0, 1.0], name='Longest Axis'):
        """
        Add the longest axis as a line with arrows extending in both directions from the center.

        Parameters:
        - center: (3,) array-like, starting point of the axis
        - direction: (3,) array-like, direction vector of the axis (should be normalized)
        - length: float, half-length of the axis in each direction
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the axis
        """
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Define the two end points
        end_positive = center + direction * length
        end_negative = center - direction * length

        # Create two arrows: one in the positive direction, one in the negative direction
        self._add_arrow(start=center, end=end_positive, color=color, name=f'{name} +')
        self._add_arrow(start=center, end=end_negative, color=color, name=f'{name} -')

    def _add_arrow(self, start, end, color, name):
        """
        Helper function to add an arrow from start to end.

        Parameters:
        - start: (3,) array-like, starting point of the arrow
        - end: (3,) array-like, ending point of the arrow
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the arrow
        """
        # Define arrow parameters
        shaft_radius = 0.05
        shaft_height = np.linalg.norm(end - start) * 0.8
        head_radius = 0.15
        head_height = np.linalg.norm(end - start) * 0.2

        # Create shaft (cylinder)
        shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
        shaft.apply_translation([0, 0, shaft_height / 2])  # Position the cylinder

        # Create head (cone)
        head = trimesh.creation.cone(radius=head_radius, height=head_height)
        head.apply_translation([0, 0, head_height / 2])  # Position the cone

        # Combine shaft and head
        arrow = trimesh.util.concatenate([shaft, head])

        # Align the arrow with the direction
        direction = end - start
        direction /= np.linalg.norm(direction)
        rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        if rotation_matrix is not None:
            arrow.apply_transform(rotation_matrix)

        # Translate to the start position
        arrow.apply_translation(start)

        # Create pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.0,
            alphaMode='OPAQUE'
        )
        pyrender_arrow = pyrender.Mesh.from_trimesh(
            arrow,
            smooth=False,
            material=material,
        )
        self.scene.add(pyrender_arrow, name=name)

    def add_plane_normal(self, point, normal, length=10.0, color=[1.0, 0.0, 1.0, 1.0], name='Plane Normal'):
        """
        Add the normal vector of the plane as a purple arrow.

        Parameters:
        - point: (3,) array-like, starting point of the normal vector
        - normal: (3,) array-like, normal vector of the plane
        - length: float, length of the normal vector arrow
        - color: list of 4 floats, RGBA color
        - name: str, name identifier for the normal vector
        """
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Define the end point
        end = point + normal * length

        # Create a cylinder for the shaft
        shaft_radius = 0.05
        shaft_height = length * 0.8
        shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
        shaft.apply_translation([0, 0, shaft_height / 2])  # Position the cylinder

        # Align the shaft with the normal vector
        shaft_axis = trimesh.geometry.align_vectors([0, 0, 1], normal)
        if shaft_axis is not None:
            shaft.apply_transform(shaft_axis)
        shaft.apply_translation(point)

        # Create a cone for the head
        head_radius = 0.15
        head_height = length * 0.2
        head = trimesh.creation.cone(radius=head_radius, height=head_height)
        head.apply_translation([0, 0, head_height / 2])  # Position the cone

        # Align the cone with the normal vector
        head_axis = trimesh.geometry.align_vectors([0, 0, 1], normal)
        if head_axis is not None:
            head.apply_transform(head_axis)
        head.apply_translation(end)

        # Combine shaft and head
        arrow = trimesh.util.concatenate([shaft, head])

        # Create pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.0,
            alphaMode='OPAQUE'
        )
        pyrender_arrow = pyrender.Mesh.from_trimesh(
            arrow,
            smooth=False,
            material=material,
        )
        self.scene.add(pyrender_arrow, name=name)

    def add_geodesic_path(self, mesh, path, color=[0.0, 0.0, 0.0, 1.0], name='Geodesic Path'):
        """
        Add the geodesic path to the scene by highlighting the faces along the path.

        Parameters:
        - mesh: trimesh.Trimesh, the original mesh
        - path: list of face indices along the geodesic path
        - color: list of 4 floats, RGBA color for the path
        - name: str, name identifier for the path
        """
        path_faces = mesh.faces[path]

        unique_vertex_indices = np.unique(path_faces.flatten())
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
        new_faces = np.array([[index_mapping[idx] for idx in face] for face in path_faces])

        path_vertices = mesh.vertices[unique_vertex_indices]
        path_mesh = trimesh.Trimesh(vertices=path_vertices, faces=new_faces, process=False)

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode='OPAQUE'
        )

        pyrender_mesh = pyrender.Mesh.from_trimesh(
            path_mesh, smooth=False, material=material
        )

        self.scene.add(pyrender_mesh, name=name)

    def render_scene(self, live=False):
        """
        Render the current scene using an offscreen renderer and display the image.
        """
        if live:
            pyrender.Viewer(
                self.scene, use_raymond_lighting=True,
                viewport_size=(800, 600),
                run_in_thread=True
            )
        else:
            try:
                # Initialize the renderer
                renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
                color, depth = renderer.render(self.scene)
                renderer.delete()

                # Convert to PIL Image
                img = PILImage.fromarray(color)

                # Display the image in the notebook
                display(img)
            except Exception as e:
                print(f"Rendering failed: {e}")
                color, depth = renderer.render(self.scene)
                renderer.delete()

                # Convert to PIL Image
                img = PILImage.fromarray(color)

                # Display the image in the notebook
                display(img)
            except Exception as e:
                print(f"Rendering failed: {e}")
