# Standard libraries
import os
import sys
import getopt

import numpy as np
import pandas as pd

import trimesh
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LogNorm

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util import values as v
from util.misc.bash import arg_check
from util.misc.colors import bcolors as c
from meshes.utils.visualize_analysis import save_mesh_views, create_feature_grid


def print_usage():
    print(
        'TODO...'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path = v.data_path
    tissue = 'myocardium'
    level = 'Membrane'
    feature = None
    verbose = 1

    try:
        opts, args = getopt.getopt(argv, 'hp:t:l:f:v:', [
            'help', 'path=', 'tissue=', 'level=', 'feature=', 'verbose='
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
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()

        if feature is None:
            print(f'{c.FAIL}No feature provided{c.ENDC}')
            sys.exit(2)

        feature_values = []
        all_values = []
        atlas_meshes = []
        groups = ['Gr1', 'Gr2', 'Gr3', 'Gr4', 'Gr5', 'Gr6', 'Gr7', 'Gr8', 'Gr9']

        print(f'{c.OKGREEN}Feature{c.ENDC}: {feature}')
        print(f'{c.OKBLUE}Processing...{c.ENDC}')
        for group in groups:

            try:
                feature_df = pd.read_csv(
                    data_path + f'ATLAS/{tissue}/Features/{level}/{feature}/{group}_atlas_{feature}.csv'
                )
                vals = feature_df['value'].values

                feature_values.append(feature_df)
                all_values.extend(vals)

                atlas_path = v.data_path + f'ATLAS/{tissue}/ATLAS_{group}.ply'
                atlas_meshes.append(trimesh.load(atlas_path))
            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC} - {group}: {e}')
                import traceback
                traceback.print_exc()

        all_values = np.array(all_values, dtype=np.float64)
        all_values = all_values[~np.isnan(all_values)]

        print(f'{c.OKGREEN}Feature values{c.ENDC}: {all_values.min()} - {all_values.max()}')

        f_max = np.percentile(all_values, 99.9)
        f_min = np.percentile(all_values, 0.1)

        if f_max > 10:
            f_max = np.percentile(all_values, 95)
            f_min = np.percentile(all_values, 5)
            print(f'{c.OKGREEN}Feature values{c.ENDC} (clipped): {f_min} - {f_max}')

        colors = [
            (0, 0, 1),  # Pure blue
            (0, 0.5, 1),  # Cyan-like
            (0, 1, 0),  # Green
            (1, 1, 0),  # Yellow
            (1, 0, 0),  # Red
        ]
        cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=1024)

        # Normalize feature values according to ranges
        print(f'{c.OKBLUE}Linear normalization{c.ENDC}')
        norm = BoundaryNorm(
            boundaries=np.linspace(
                f_min, f_max,
                cmap.N
            ), ncolors=cmap.N
        )

        print(f'{c.OKBLUE}Normalizing...{c.ENDC}')
        for i, group in enumerate(groups):
            atlas = atlas_meshes[i]

            values = feature_values[i]['value']
            # nan values as the mean of the neighbors feature values
            # values[np.isnan(values)] = np.mean(values[~np.isnan(values)])

            vertex_colors = cmap(norm(values))
            atlas.visual.vertex_colors = vertex_colors

            path = data_path + f'ATLAS/{tissue}/FeaturesNormalized/{level}/{feature}/{group}_atlas_{feature}_normalized.ply'

            # Path checks
            split_path = path.split('/')
            for i in range(1, 3).__reversed__():
                if not os.path.exists('/'.join(split_path[:-i])):
                    os.mkdir('/'.join(split_path[:-i]))

            atlas.export(path)
            df = pd.DataFrame({
                'vertex_id': np.arange(len(values)),
                'value': values
            })
            df.to_csv(path.replace('.ply', '.csv'), index=False)

        # Save figures
        input_meshes = data_path + f'ATLAS/{tissue}/FeaturesNormalized/{level}/{feature}/'
        output_grid = input_meshes + 'grid/'

        if not os.path.exists(output_grid):
            os.makedirs(output_grid, exist_ok=True)

        save_mesh_views(input_meshes)
        create_feature_grid(input_meshes + 'output_images', output_grid, feature)

    except getopt.GetoptError:
        print_usage()

