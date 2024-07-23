from mesh3d import merge
import trimesh
import numpy as np


def draw_spheres(list_of_coordinates, colors, r=5.0):
    if list_of_coordinates.shape == (3,):
        list_of_coordinates = list_of_coordinates.reshape((1, 3))
    esferas = []
    for i, coord in enumerate(list_of_coordinates):
        mesh_sph = trimesh.creation.icosphere(subdivisions=2, radius=r, color=None)
        mesh_sph.apply_translation(((coord[0], coord[1], coord[2])))
        esferas.append(mesh_sph)
    for j, color in enumerate(colors):
        esferas[j].visual.vertex_colors = color
    all_ball = merge(esferas)
    return all_ball


def draw_line_normal(origin, direction, colors, extension=20, r=2):
    sphs = []
    for k in range(extension):
        sphs.append(draw_spheres(origin + direction * k, colors, r))
    sphs = merge(sphs)
    return sphs


def get_mainaxis(mesh):
    cov = np.cov(mesh.vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigen_i = np.where(np.asarray(eigenvalues==eigenvalues.max()))[0][0]
    return eigenvectors[:,eigen_i]

# def draw_line_normal(origin, direction, extension=20, r=2):
#     sphs = []
#     for k in range(extension):
#         sphs.append(draw_spheres(origin + direction * k, r))
#     sphs = merge(sphs)
#     return sphs

# def draw_spheres(list_of_coordinates, r=8.0):
#     # Si meto un punto solo, pues con esto me garantizo que se lea bien
#     if list_of_coordinates.shape == (3,):
#         list_of_coordinates = list_of_coordinates.reshape((1, 3))

#     # Quiero asignar colores en un orden no aleatorio. Para eso voy a usar los colores
#     # presentes en -sphere_colors-
#     if len(sphere_colors.keys()) >= list_of_coordinates.shape[0]:
#         # Creo esferas
#         esferas, colores = [], []
#         for i, coord in enumerate(list_of_coordinates):
#             mesh_sph = trimesh.creation.icosphere(subdivisions=2, radius=r, color=None)
#             mesh_sph.apply_translation(((coord[0], coord[1], coord[2])))

#             esferas.append(mesh_sph)

#             colores.append(
#                 np.asarray(
#                     0 * mesh_sph.visual.face_colors
#                     + np.asarray(sphere_colors["p" + str((i + 1) % 70)])
#                 )
#             )

#         # Merge las esferas y ponerles el color
#         all_ball = merge(esferas)
#         all_ball.visual.face_colors = np.vstack(colores)
#     else:
#         print("hay mas vertices que colores... crea mas colores!")
#         # Creo esferas
#         esferas = []
#         for i, coord in enumerate(list_of_coordinates):
#             mesh_sph = trimesh.creation.icosphere(subdivisions=2, radius=r, color=None)
#             mesh_sph.apply_translation(((coord[0], coord[1], coord[2])))

#             esferas.append(mesh_sph)
#         # Merge las esferas y ponerles el color
#         all_ball = merge(esferas)

#     return all_ball
