import sys
import os

import trimesh

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util import values as v
from meshes.utils.registration.surface_map_computation import check_mesh_consistency

v.data_path = '/home/txete/data/cluster/'

for gr in v.specimens.keys():
    if gr != 'Gr10':
        continue
    print(f'Group: {gr}')
    atlas_path = v.data_path + f'ATLAS/myocardium/ATLAS_{gr}.ply'
    atlas = trimesh.load(atlas_path, file_type='ply')

    for s in v.specimens[gr]:
        print(f'\tSpecimen: {s}')
        mesh_path = v.data_path + f'{gr}/3DShape/Tissue/splanchnic/2019{s}Shape.ply'
        mesh = trimesh.load(mesh_path, file_type='ply')

        # trimesh.repair.fix_normals(mesh)
        # trimesh.repair.fix_inversion(mesh)
        # trimesh.repair.fix_winding(mesh)

        check_mesh_consistency(atlas, mesh)

        # mesh.export(mesh_path)
        # update_land_pinned(s)

    # trimesh.repair.fix_normals(atlas)
    # trimesh.repair.fix_inversion(atlas)
    # trimesh.repair.fix_winding(atlas)

    # atlas.export(atlas_path)
    # update_land_pinned('', gr=gr)