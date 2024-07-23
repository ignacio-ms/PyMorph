from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import os
from mrcnn.config import Config
from PIL import Image
import numpy as np
import nibabel as nib
from PIL import Image
from itertools import product
import cv2
import pickle


def tile(img, d):
    w, h, c = img.shape
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    l = []
    for i, j in grid:
        box = (j, i, j + d, i + d)
        l.append(box)
        # l.append(img.crop(box))
    crops = {}
    for u, x in enumerate(l):
        crops["{}_{}_{}_{}".format(x[0], x[2], x[1], x[3])] = img[
            x[0] : x[2], x[1] : x[3]
        ]
    return crops


class CaseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "testing"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    # STEPS_PER_EPOCH = 1000

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    # 3 si son imagenes en RGB
    # IMAGE_CHANNEL_COUNT = 1

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 64

    # El numero de objetos maximo de objetos que se pueden encontrar en una imagen
    TRAIN_ROIS_PER_IMAGE = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    MAX_GT_INSTANCES = 256

    POST_NMS_ROIS_INFERENCE = 2048
    POST_NMS_ROIS_TRAINING = 2048
    RPN_NMS_THRESHOLD = 0.8

    # Whether to use image augmentation in training mode
    # AUGMENT = True

    # Whether to use image scaling and rotations in training mode
    # SCALE = True

    # Random crop larger images
    # CROP = True
    # CROP_SHAPE = np.array([256, 256, 3])


m = "mask_rcnn_dapidv_all_0100.h5"
MODEL_DIR = "logs"


config = CaseConfig()
print(config.display())

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(m, by_name=True)


### INFERENCE
# folder = "/Users/dvarelat/Documents/MASTER/TFM/DATA/DECON_05/DAPI"
# ESPECIMEN = "20190504_E1"
especimens = [
    # "20190401_E2",
    # "20190504_E1",
    "20190404_E2",
    "20190520_E4",
    "20190516_E3",
    "20190806_E3",
    "20190520_E2",
    "20190401_E3",
    "20190517_E1",
    "20190520_E1",
    "20190401_E1",
]
for ESPECIMEN in especimens:

    gasp_mem = f"/homedtic/dvarela/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    nuclei = f"/homedtic/dvarela/DECON_05/DAPI/{ESPECIMEN}_DAPI_decon_0.5.nii.gz"

    DAPI = nib.load(nuclei).get_fdata()
    DAPI = DAPI[:, :, :, 0]
    print(DAPI.shape)

    pred_mem = nib.load(gasp_mem).get_fdata()
    mask_mem = np.where(pred_mem != 0, True, False)
    mask_nuclei = mask_mem * DAPI

    results = []
    print(mask_nuclei.shape[2])
    for i in range(mask_nuclei.shape[2]):
        print(i)
        s = cv2.cvtColor(mask_nuclei[:, :, i].astype("uint16"), cv2.COLOR_GRAY2RGB)
        crops = tile(s, 256)
        res_crops = []
        for c in crops.keys():
            print(c)
            r = model.detect([crops[c]], verbose=0)
            res_crops.append(r[0])
        results.append(res_crops)

    file_out = f"/homedtic/dvarela/division/INFERENCE/{ESPECIMEN}_E1_inference.pkl"
    with open(file_out, "wb") as f:
        pickle.dump(results, f)

# R = [i for i,r in enumerate(res_crops) if r["masks"].shape[2] != 0 ]
# results_good = {list(crops.keys())[i]:res_crops[i] for i in R}
# results_good.keys()

# prediction = np.zeros((1024,1024))
# for k in results_good.keys():
#     x = results_good[k]["masks"]
#     if x.shape[2] >1 :
#         print(x.shape)
#         x = np.sum(results_good[k]["masks"], axis=2)
#     x = x.reshape((256, 256))
#     print(x.shape)
#     kk = k.split("_")
#     kk = [int(i) for i in kk]
#     print(prediction[kk[0]:kk[1], kk[2]:kk[3]].shape)
#     prediction[kk[0]:kk[1], kk[2]:kk[3]] = x

# image_path = "/Users/dvarelat/Documents/MASTER/TFM/methods/daniela_division/SP8_ROIs/20190308_E4/169/DAPIdv/20190308_E4_DAPIdv_169_v0RGB.tif"
