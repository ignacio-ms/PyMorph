import os
import sys
import getopt

from scipy import ndimage
from skimage import exposure
import numpy as np
import cv2
import gc

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

from filtering import cardiac_region as cr
from filtering.cardiac_region import get_margins, crop_img, restore_img
from feature_extraction.feature_extractor import filter_by_volume, filter_by_margin
from filtering.run_filter_tissue import filter_by_tissue


if __name__ == '__main__':
    ds = HtDataset()

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blur_kernel_size = (5, 5)
    sigma = 1.5
    iterations = 2

    for group in ds.specimens.keys():
        print(f'{c.BOLD}Group{c.ENDC}: {group}')

        for i, specimen in enumerate(ds.specimens[group]):
            if i > 0:
                sys.exit(0)
            print(f'\t{c.BOLD}Specimen{c.ENDC}: {specimen}')

            try:
                img_path, _ = ds.read_specimen(specimen, 'Nuclei', 'Segmentation', verbose=1)
                lines_path, _ = ds.read_line(specimen, verbose=1)

                path_split = img_path.split('/')
                img_path_out = '/'.join(path_split[:-1] + ['Filtered', path_split[-1]])

                img = imaging.read_image(img_path, axes='XYZ', verbose=1)
                lines = imaging.read_image(lines_path, axes='XYZ', verbose=1)

                filtered = filter_by_tissue(
                    img, lines,
                    ['myocardium', 'splanchnic'],
                    dilate=3, dilate_size=3,
                    verbose=1
                )

                smoothed = np.zeros_like(filtered, dtype=np.uint16)

                labels = np.unique(filtered)[1:]  # Exclude the background label (assumed to be 0)
                bar = LoadingBar(len(labels))
                for label in labels:

                    margins = cr.get_cell_margins(filtered, cell_id=label, ma=5)
                    cell_crop = cr.crop_img(filtered, margins)
                    cell_crop = np.where(cell_crop == label, 255, 0).astype(np.uint16)

                    for z in range(cell_crop.shape[-1]):
                        mask_slice = cell_crop[..., z]

                        mask_slice = cv2.dilate(mask_slice, dilate_kernel, iterations=iterations)
                        mask_slice = cv2.erode(mask_slice, erode_kernel, iterations=iterations)
                        mask_slice = cv2.GaussianBlur(mask_slice, blur_kernel_size, sigma)

                        _, mask_slice = cv2.threshold(
                            mask_slice, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )

                        cell_crop[..., z] = mask_slice

                    cell_crop[cell_crop == 255] = label
                    smoothed[
                        int(margins[0][0]):int(margins[1][0]),
                        int(margins[0][1]):int(margins[1][1]),
                        int(margins[0][2]):int(margins[1][2])
                    ] = cell_crop

                    bar.update()

                    del cell_crop, mask_slice
                    gc.collect()

                bar.end()
                imaging.save_nii(smoothed, img_path_out, verbose=1)

                del img, lines, filtered, smoothed, labels
                gc.collect()

            except Exception as e:
                print(f'{c.FAIL}Error{c.ENDC}: {e}')
                continue
