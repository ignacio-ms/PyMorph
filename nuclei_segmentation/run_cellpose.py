# Standard libraries
import os
import sys
import getopt

import cv2
import nibabel as nib
from csbdeep.io import save_tiff_imagej_compatible
from skimage import exposure

import numpy as np
import matplotlib.pyplot as plt

from cellpose import models, core, plot, utils

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.colors import bcolors as c
from auxiliary.datasets import HtDataset
from auxiliary import imaging

# Configurations
use_gpu = core.use_gpu()
print(f"GPU activated: {use_gpu}")

from cellpose.io import logger_setup
logger_setup();


def load_img(img_path, img_type='.nii.gz', equalize_img=True, verbose=0):
    """
    Load and normalize image.
    :param img_path: Path to image.
    :param img_type: Extension of image[.nii.gz | .tif]. (Default: .nii.gz)
    :param equalize_img: Perform histogram equalization on image. (Default: True)
    :param verbose: Verbosity level.
    :return: Image.
    """
    img = imaging.read_nii(img_path) if img_type == '.nii.gz' else None  # To be implemented
    if equalize_img:
        img = exposure.equalize_hist(img)

    if verbose:
        print(f'{c.OKBLUE}Loading image{c.ENDC}: {img_path}')
        print(f'{c.BOLD}Image shape{c.ENDC}: {img.shape}')

    return img


def load_model(model_type='nuclei', model_path=None):
    """
    Load cellpose model.
    :param model_type: Type of model to load. (nuclei, cyto)
        -nuclei: nuclei model
        -(cyto, cyto2, cyto3): cytoplasm model
        -tissuenet_cp3: tissuenet dataset.
        -livecell_cp3: livecell dataset
        -yeast_PhC_cp3: YEAZ dataset
        -yeast_BF_cp3: YEAZ dataset
        -bact_phase_cp3: omnipose dataset
        -bact_fluor_cp3: omnipose dataset
        -deepbacs_cp3: deepbacs dataset
        -cyto2_cp3: cellpose dataset
    :param model_path: Path to model. (Default: None)
    :return: Model.
    """
    # Not needed at the moment
    if model_path is None:
        model_path = '../models/cellpose_models/'

    return models.Cellpose(gpu=use_gpu, model_type=model_type)


def run(
        model, img,
        diameter=25, channels=[0, 0],
        normalize=True,
):
    """
    Run cellpose on image.
    :param model: Cellpose model.
    :param img: Image.
    :param diameter: Diameter of nuclei. (Default: 0)
    :param channels: Channels to use. (Default: [0, 0])
    :param verbose: Verbosity level.
    :return: Masks.
    """

    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        normalize=normalize,
        do_3D=True
    )

    return masks


def save_prediction(labels, out_path, verbose=0):
    """
    Save prediction as tiff image.
    :param labels: Labels.
    :param out_path: Path to save prediction.
    :param verbose: Verbosity level.
    """

    save_tiff_imagej_compatible(out_path, labels, axes='XYZ')

    if verbose:
        print(f'{c.OKGREEN}Saving prediction{c.ENDC}: {out_path}')


def print_usage():
    """
    Print usage of script.
    """
    print(

    )


def arg_check(opt, arg, valid, valid_long, type=None):
    """
    Check input arguments.
    :param opt: Option.
    :param arg: Argument.
    :param valid: Valid option.
    :param valid_long: Valid long option.
    :param type: Dtype of argument.
    :return: Argument.
    """
    if opt in (valid, valid_long):
        if type is not None:
            try:
                arg = type(arg)
            except ValueError:
                print(f"{c.FAIL}Error{c.ENDC}: {valid_long} must be {type}.")
                print_usage()
        return arg


if __name__ == '__main__':
    argv = sys.argv[1:]

    img, group, model, normalize, equalize, diameter, channels, verbose = None, None, None, None, None, None, None, None

    try:
        opts, args = getopt.getopt(argv, "i:gr:m:n:e:d:c:v:", [
            "image=", "group=", "model=", "normalize=", "equalize=", "diameter=", "channels=", "verbose="
        ])

        if len(opts) == 0 or len(opts) > 8:
            print_usage()

        for opt, arg in opts:
            if opt in ("-i", "--image"):
                img = arg_check(opt, arg, "-i", "--image", str)
            if opt in ("-gr", "--group"):
                group = arg_check(opt, arg, "-gr", "--group", str)
            if opt in ("-m", "--model"):
                model = arg_check(opt, arg, "-m", "--model", str)
            if opt in ("-n", "--normalize"):
                normalize = arg_check(opt, arg, "-n", "--normalize", bool)
            if opt in ("-e", "--equalize"):
                equalize = arg_check(opt, arg, "-e", "--equalize", bool)
            if opt in ("-d", "--diameter"):
                diameter = arg_check(opt, arg, "-d", "--diameter", int)
            if opt in ("-c", "--channels"):
                channels = arg_check(opt, arg, "-c", "--channels", list)
            if opt in ("-v", "--verbose"):
                verbose = arg_check(opt, arg, "-v", "--verbose", int)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()



    except getopt.GetoptError:
        print_usage()
