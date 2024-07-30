# Standard Packages
import sys
import os

import numpy as np
import pandas as pd

from skimage import morphology
import SimpleITK as sitk
import porespy as ps
from radiomics import shape, firstorder

from collections import Counter

# Custom Packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.utils.colors import bcolors as c

from auxiliary.data.dataset_ht import HtDataset
from auxiliary.data import imaging

from auxiliary import values as v

from filtering import cardiac_region as c


def filter_by_volume(seg_img, percentile=98, verbose=0):
    """
    Filter cells by volume using the percentile.
    :param seg_img: Segmentation image.
    :param percentile: Percentile to filter. (default: 98)
    :param verbose: Verbosity level. (default: 0)
    :return: Filtered segmentation image.
    """
    if verbose:
        print(f'{c.OKBLUE}Filtering by volume...{c.ENDC}')

    props = ps.metrics.regionprops_3D(morphology.label(seg_img))
    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [seg_img[ce[0], ce[1], ce[2]] for ce in centroids]

    if verbose:
        print(f'\tFound{c.BOLD} {len(centroids)} {c.ENDC} cells')

    new_rows = []

    for i, p in enumerate(props):
        if centroids_labels[i] == 0 or p.volume <= 10:
            continue

        new_rows.append({
            "volumes": p.volume,
            "original_labels": centroids_labels[i],
        })

    df = pd.DataFrame(new_rows)

    lower_bound = np.percentile(df.volumes, 100 - percentile)
    upper_bound = np.percentile(df.volumes, percentile)

    remove = []
    remove += df[df.volumes > upper_bound].original_labels.tolist()
    remove += df[df.volumes < lower_bound].original_labels.tolist()

    remove = set(remove)

    if verbose:
        print(f'\tRemoving {c.BOLD}{len(remove)}{c.ENDC} cells...')

    # Optimize the removal of cells using integer labels and boolean mask
    # instead of iterating over the whole image
    seg_membrane_int = seg_img.astype(int)
    remove_int = np.array(list(remove)).astype(int)

    # Create a boolean mask for the integer labels
    max_label = np.max(seg_membrane_int)
    bool_mask = np.zeros(max_label + 1, dtype=bool)
    bool_mask[remove_int] = True

    # Apply the boolean mask
    mask = bool_mask[seg_membrane_int]
    seg_img[mask] = 0

    return seg_img

def compute_most_common(lines, props):
    """
    Compute the most common value in a list.
    :param lines: Lines image.
    :param props: List of properties.
    :return: Most common value.
    """
    counter = Counter(lines[props.slices].flatten())

    if len(list(counter)) == 1:
        m = list(counter)[0]
    else:
        d = {k: v for k, v in counter.items() if k != 0}
        m = max(d, key=d.get)

    return m


def shape_features(props, centroids_labels):
    pass


def extract(seg_img, raw_img, raw_img_path, verbose=0):
    """
    Extract features from the segmented image.
    :param seg_img: Segmented image.
    :param raw_img: Raw image.
    :param raw_img_path: Path to raw image.
    :param verbose: Verbosity level. (default: 0)
    :return: Features.
    """
    if verbose:
        print(f'{c.OKBLUE}Extracting features...{c.ENDC}')

    props = ps.metrics.regionprops_3D(morphology.label(seg_img))
    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [seg_img[ce[0], ce[1], ce[2]] for ce in centroids]

    if verbose:
        print(f'\tFound{c.BOLD} {len(props)} {c.ENDC} cells')

    new_rows = []



