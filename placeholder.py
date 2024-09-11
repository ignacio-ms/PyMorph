import pandas as pd
import sys
import os

import tensorflow as tf
import json
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from cell_division.nets.transfer_learning import CNN
from auxiliary.data.dataset_cell import CellDataset
from auxiliary.data.dataset_unlabeled import UnlabeledDataset
from auxiliary import values as v
from auxiliary.utils.colors import bcolors as c
from auxiliary.utils import visualizer as vis
from auxiliary.data import imaging
from auxiliary.data.dataset_ht import HtDataset

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import GlobalAveragePooling2D
from cell_division.nets.custom_layers import (
    w_cel_loss,
    focal_loss,
    ExtendedLSEPooling,
    extended_w_cel_loss,
    LSEPooling
)

from cell_division.nets.cam import GradCAM, overlay_heatmap, CAM, GradCAMpp
from cell_division.semi_supervised import semi_supervised_learning as SSL

# GPU config
from auxiliary.utils.timer import LoadingBar
from auxiliary.gpu.gpu_tf import (
    increase_gpu_memory,
    set_gpu_allocator,
    clear_session
)


# img = np.zeros((1024, 1024, 500), dtype=float)
# imaging.save_prediction(img, v.data_path + 'Auxiliary/CellposeClusterTest/RawImages/Nuclei/black.tif', verbose=1)


# ds = HtDataset()
#
# img = ds.read_specimen('0806_E6', level='Nuclei', type='RawImages', verbose=1)

from filtering.cardiac_region import get_margins, crop_img

# margins = get_margins(
#     line_path=v.data_path + 'Gr4/Segmentation/LinesTissue/line_20190806_E6.nii.gz',
#     img_path=v.data_path + 'Gr4/RawImages/Nuclei/20190806_E6_DAPI_decon_0.5.nii.gz',
#     tissue='myocardium', verbose=1
# )
#
# img_path, _ = ds.read_specimen('0806_E6', level='Nuclei', type='RawImages', verbose=1)
# img = imaging.read_image(img_path, verbose=1)
# img = crop_img(img, margins, verbose=1)
#
# metadata, affine = imaging.load_metadata(img_path)
#
# imaging.save_nii(
#     img, v.data_path + 'Auxiliary/CellposeClusterTest/RawImages/Nuclei/0806_E6_myocardium.nii.gz',
#     affine=affine, verbose=1
# )
#
# img_path, _ = ds.read_specimen('0806_E6', level='Membrane', type='RawImages', verbose=1)
# img = imaging.read_image(img_path, verbose=1)
# imaging.nii2h5(img, img_path.replace('.nii.gz', '.h5'), verbose=1)

# from skimage import io
#
# img_paths = [
#     os.path.join(v.data_path + 'CellDivision/images_unlabeled_2d/', f)
#     for f in os.listdir(v.data_path + 'CellDivision/images_unlabeled_2d/')
#     if f.endswith('.tif')
# ]
#
# for i, img_path in enumerate(img_paths):
#     img = io.imread(img_path)
#     if img is not None:
#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         # Save image
#         io.imsave(img_path, img)

# img_path = v.data_path + 'Gr4/Segmentation/Nuclei/myocardium/20190806_E6_nuclei_mask_None_myocardium_myocardium.nii.gz'
# img_path_out = img_path.replace('.nii.gz', '_smoothed.nii.gz')
# img =  imaging.read_image(img_path, axes='XYZ', verbose=1)
# print(img.shape)
#
# from skimage import restoration
# import numpy as np
#
# def anisotropic_diffusion_filter(segmentation_mask, num_iter=10, kappa=50, gamma=0.1):
#     """
#     Applies anisotropic diffusion to a label image for edge-preserving smoothing.
#
#     Parameters:
#     - segmentation_mask: 3D numpy array, the labeled segmentation image.
#     - num_iter: Number of iterations to run the diffusion process.
#     - kappa: Conductance coefficient, controls sensitivity to edges.
#     - gamma: Controls the rate of diffusion.
#
#     Returns:
#     - Smoothed segmentation mask.
#     """
#     smoothed_segmentation = restoration.denoise_tv_chambolle(
#         segmentation_mask, weight=kappa, max_num_iter=num_iter
#     )
#     return smoothed_segmentation
#
# if __name__ == '__main__':
#     # Example usage:
#     # segmentation_mask: a 3D numpy array representing the instance segmentation
#     smoothed_segmentation = anisotropic_diffusion_filter(
#         img, num_iter=3, kappa=50, gamma=0.1
#     )
#     imaging.save_prediction(smoothed_segmentation, img_path_out, verbose=1)

def plot_kde_single(
        neg, title, include_hist=False,
        sub_index=None, n_rows=1, n_cols=1,
        xlim=None, labels=None, valley=True
):
    plt.figure(figsize=(8, 5))

    sns.rugplot(neg, color='green', alpha=.3, height=.1, linewidth=3.5, expand_margins=False)

    kde_1 = sns.kdeplot(neg, linewidth=4, color='green')

    if include_hist:
        sns.histplot(x=neg, kde=True, color='green', alpha=.0, linewidth=0)

    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    plt.tight_layout()

    # Get the KDE data
    kde_x, kde_y = kde_1.get_lines()[0].get_data()

    # Function to find the valley between peaks
    def find_valley(kde_x, kde_y):
        peaks, _ = find_peaks(kde_y)
        valleys, _ = find_peaks(-kde_y)

        if len(peaks) >= 2:
            peak_1, peak_2 = peaks[:2]
            valley = valleys[(valleys > peak_1) & (valleys < peak_2)]
            if len(valley) > 0:
                valley_x = kde_x[valley[0]]
                print(f"Valley between peaks at x = {valley_x}")
            else:
                valley_x = None
                print("No valley found between the peaks.")
        else:
            valley_x = None
            print("Fewer than two peaks detected.")

        return valley_x

    # Function to compute count percentages up to and after the valley
    def compute_count_percentages(neg, split_x):
        count_before = np.sum(neg <= split_x)
        count_after = np.sum(neg > split_x)
        total_count = len(neg)
        return count_before / total_count, count_after / total_count

    valley_x = find_valley(kde_x, kde_y)

    if valley_x and valley:
        plt.axvline(valley_x, color='black', linestyle='--')

    if valley_x is not None and valley:
        before_valley, after_valley = compute_count_percentages(neg, valley_x)
        print(f"Percentage of counts before valley: {before_valley * 100:.2f}%")
        print(f"Percentage of counts after valley: {after_valley * 100:.2f}%")

        # Annotate the percentages on the plot
        y_max = np.max(kde_y)
        plt.text(valley_x, y_max * 0.5, f'{before_valley * 100:.2f}%',
                 horizontalalignment='right', verticalalignment='center',
                 fontsize=12, color='black')
        plt.text(valley_x, y_max * 0.5, f'{after_valley * 100:.2f}%',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12, color='black')
    else:
        before_valley, after_valley = None, None
        print("No valley detected, unable to calculate count percentages.")

    return kde_1.get_lines()[0].get_data(), before_valley, after_valley
