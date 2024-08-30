import os
import sys
import getopt

import numpy as np
from scipy import ndimage

import pickle
import mcubes
import trimesh
import scipy
import skimage
import porespy as ps
import csv
from skimage import morphology
from skimage.morphology import disk
import pandas as pd
import time
import ast

from cellpose import models, core
from csbdeep.utils import normalize as deep_norm

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset, find_specimen
from auxiliary.utils.timer import LoadingBar, timed
from auxiliary.data import imaging

from filtering.cardiac_region import get_margins, crop_img, restore_img, filter_by_tissue
from feature_extraction.feature_extractor import filter_by_volume, filter_by_margin

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def median_3d_array(img, disk_size):
    if len(img.shape) == 4:
        img = img[:, :, :, 0]

    return ndimage.median_filter(img, size=disk_size)


if __name__ == '__main__':

    disk_size = 3
    ds = HtDataset()
    levels = ['Nuclei', 'Membrane']
    tissues = ['myocardium', 'splanchnic']

    for level in levels:
        print(f'{c.OKBLUE}Level{c.ENDC}: {level}')

        for tissue in tissues:
            print(f'{c.OKBLUE}Tissue{c.ENDC}: {tissue}')
            for group in ds.specimens.items():
                print(f'{c.BOLD}Group{c.ENDC}: {group[0]}')

                for specimen in group[1]:
                    print(f'\t{c.BOLD}Specimen{c.ENDC}: {specimen}')

                    try:
                        img_path, _ = ds.read_specimen(
                            specimen, level, 'Segmentation',
                            filtered=True if level == 'Nuclei' else False, verbose=1
                        )
                        img_path_raw, _ = ds.read_specimen(specimen, level, type='RawImages', verbose=1)
                        lines_path, _ = ds.read_line(specimen, verbose=1)
                        path_out = v.data_path + f'/{group[0]}/3DShape/{tissue}/2019{specimen}_{tissue}.ply'

                        img = imaging.read_image(img_path, axes='XYZ', verbose=1)
                        lines = imaging.read_image(lines_path, axes='XYZ', verbose=1)
                        metadata, _ = imaging.load_metadata(img_path_raw)

                        img = filter_by_tissue(
                            img, lines, tissue,
                            dilate=2 if level == 'Membrane' else 3, dilate_size=3,
                            verbose=1
                        )

                        props = ps.metrics.regionprops_3D(morphology.label(img))
                        meshes = []

                        bar = LoadingBar(len(props))

                        for i, p in enumerate(props):
                            coords = p.mask * 1
                            add = 10
                            aux = np.zeros(shape=tuple(np.asarray(coords.shape) + add), dtype=np.uint8)
                            aux[
                                add // 2: -add // 2,
                                add // 2: -add // 2,
                                add // 2: -add // 2
                            ] = coords.copy()
                            coords = aux.copy()
                            del aux

                            coords = median_3d_array(coords, disk_size)
                            vert, trian = mcubes.marching_cubes(mcubes.smooth(coords), 0)
                            vert -= vert.mean(axis=0)
                            vert += p.centroid
                            vert *= np.asarray([metadata['x_res'], metadata['y_res'], metadata['z_res']])

                            if len(vert) > 0 and len(trian) > 0:
                                mesh = trimesh.Trimesh(vertices=vert, faces=trian, process=False)
                                trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.5, iterations=20)
                                mesh = mesh.simplify_quadratic_decimation(int(100))
                                meshes.append(mesh)

                            bar.update()

                        bar.end()

                        combined_mesh = trimesh.util.concatenate(meshes)
                        combined_mesh.export(path_out)
                        print(f'{c.OKGREEN}Saved{c.ENDC}: {path_out}')


                    except Exception as e:
                        print(f'{c.FAIL}Error{c.ENDC}: {e}')
                        continue
