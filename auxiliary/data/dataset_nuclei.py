# Standard Packages
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

import math
import sys
import os
from skimage import io
from skimage import morphology

import ast

# Custom Packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.data import imaging
from auxiliary.utils.colors import bcolors as bc

from filtering import cardiac_region as cr


class NucleiDataset(tf.keras.utils.Sequence):
    def __init__(self, specimen, tissue='myocardium', resize=(50, 50), verbose=0):
        def parse_centroids(centroid):
            centroid = centroid.replace('[', '').replace(']', '').split(',')
            x, y, z = int(centroid[0]), int(centroid[1]), int(centroid[2])
            return np.array([x, y, z])

        self.N_CLASSES = 3
        self.CLASS_NAMES = ['Prophase/Metaphase', 'Anaphase/Telophase', 'Interphase']
        self.CLASSES = ['0', '1', '2']

        self.specimen = specimen

        self.batch_size = 32
        self.resize = (50, 50)

        ds = HtDataset()
        self.features = ds.get_features(specimen, 'Nuclei', tissue, verbose=verbose)

        if 'cell_id' not in self.features.columns:
            if 'original_labels' in self.features.columns:
                self.features.rename(
                    columns={'original_labels': 'cell_id'},
                    inplace=True
                )
            else:
                raise ValueError('cell_id not found in features')

        self.cell_ids = self.features['cell_id']
        centroids = self.features['centroids']
        self.centroids = np.array([parse_centroids(c) for c in centroids])

        self.seg_path, _ = ds.read_specimen(specimen, 'Nuclei', 'Segmentation', verbose=verbose)
        self.raw_path, _ = ds.read_specimen(specimen, 'Nuclei', 'RawImages', verbose=verbose)

        self.seg_img = imaging.read_image(self.seg_path, verbose=verbose)
        self.raw_img = imaging.read_image(self.raw_path, verbose=verbose)

        self.resize = resize

        if verbose:
            print(f'{bc.OKGREEN}Nuclei Dataset created successfully{bc.ENDC}')
            print(f'\t{bc.OKBLUE}Specimen: {bc.ENDC}{self.specimen}')
            print(f'\t{bc.OKBLUE}Tissue: {bc.ENDC}{tissue}')
            print(f'\t{bc.OKBLUE}Number of Nuclei: {bc.ENDC}{len(self.cell_ids)}')

    def __get_image(self, idx):
        cell_id = self.cell_ids[idx]
        c = self.centroids[idx]

        # Crop cell region + dilatation
        margins = cr.get_cell_margins(self.seg_img, cell_id, ma=0)
        mask = self.seg_img[
            c[0]-25: c[0]+25, c[1]-25: c[1]+25,
            int(margins[0][2]): int(margins[1][2])
        ]
        mask = np.where(mask == cell_id, 255, 0).astype(np.uint8)
        mask_dilated = morphology.opening(mask, morphology.ball(3))

        # Remove empty z-slices
        non_empty_slices = [
            i for i in range(mask.shape[2])
            if np.any(mask[..., i])
        ]
        mask = mask[..., non_empty_slices]
        mask_dilated = mask_dilated[..., non_empty_slices]

        # Crop raw image
        img_cell = self.raw_img[
            c[0]-25: c[0]+25, c[1]-25: c[1]+25,
            int(margins[0][2]): int(margins[1][2])
        ]
        img_cell = img_cell[..., non_empty_slices]

        # Erse background
        img_cell_no_bg = np.where(mask_dilated == 0, 0, img_cell)

        # IQR filter ver intensity
        img_cell_no_bg, intensities, thr = imaging.iqr_filter(
            img_cell_no_bg, get_params=True, verbose=0
        )
        mask_dilated = mask_dilated[..., intensities > thr]
        return img_cell_no_bg, mask_dilated

    def __get_cell_id(self, idx):
        return self.cell_ids[idx]

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, item):
        img, mask = self.__get_image(item)
        return (
            img.astype(np.uint8),
            mask.astype(np.uint8),
            self.__get_cell_id(item)
        )






