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


ds = HtDataset()
#
# img = ds.read_specimen('0806_E6', level='Nuclei', type='RawImages', verbose=1)

from filtering.cardiac_region import get_margins, crop_img

margins = get_margins(
    line_path=v.data_path + 'Gr4/Segmentation/LinesTissue/line_20190806_E6.nii.gz',
    img_path=v.data_path + 'Gr4/RawImages/Nuclei/20190806_E6_DAPI_decon_0.5.nii.gz',
    tissue='myocardium', verbose=1
)

img_path, _ = ds.read_specimen('0806_E6', level='Nuclei', type='RawImages', verbose=1)
img = imaging.read_image(img_path, verbose=1)
img = crop_img(img, margins, verbose=1)

metadata, affine = imaging.load_metadata(img_path)

imaging.save_nii(
    img, v.data_path + 'Auxiliary/CellposeClusterTest/RawImages/Nuclei/0806_E6_myocardium.nii.gz',
    affine=affine, verbose=1
)

img_path, _ = ds.read_specimen('0806_E6', level='Membrane', type='RawImages', verbose=1)
img = imaging.read_image(img_path, verbose=1)
imaging.nii2h5(img, img_path.replace('.nii.gz', '.h5'), verbose=1)

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

