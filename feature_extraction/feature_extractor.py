# Standard Packages
import sys
import os

import numpy as np
import pandas as pd
from numba import njit

from skimage import morphology
import SimpleITK as sitk
import porespy as ps
from radiomics import shape, firstorder

from collections import Counter

from skimage.measure import regionprops, label
# import cupy as cp
# from cucim.skimage.measure import label, regionprops

# Custom Packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from utils.misc.colors import bcolors as c
from utils.misc.timer import LoadingBar
from utils.data import imaging as cr


@njit
def compute_component_sizes(labeled_img, max_label):
    sizes = np.zeros(max_label + 1, dtype=np.int32)
    for i in range(labeled_img.size):
        label = labeled_img.flat[i]
        if label > 0:
            sizes[label] += 1
    return sizes


@njit
def filter_small_components(labeled_img, component_sizes, min_size, max_size):
    for i in range(labeled_img.size):
        label = labeled_img.flat[i]
        if label > 0 and (component_sizes[label] <= min_size or
                          (max_size is not None and component_sizes[label] >= max_size)):
            labeled_img.flat[i] = 0
    return labeled_img


def filter_connected_components_with_size(seg_img, min_size=20, max_size=None, verbose=0):
    filtered_img = np.zeros_like(seg_img)

    # Iterate through unique labels directly
    max_label = seg_img.max()

    bar = LoadingBar(max_label)
    for label_id in range(1, max_label + 1):

        if np.any(seg_img == label_id):
            # Extract mask for the current label
            label_mask = (seg_img == label_id)

            # Perform connected component analysis
            labeled_components, num_features = label(label_mask, connectivity=1, return_num=True)

            # Compute component sizes
            component_sizes = compute_component_sizes(labeled_components, num_features)

            # Filter components by size
            filtered_components = filter_small_components(labeled_components, component_sizes, min_size, max_size)

            # Retain the largest valid connected component
            remaining_labels = np.unique(filtered_components)
            remaining_labels = remaining_labels[remaining_labels != 0]

            if remaining_labels.size > 0:
                largest_component = max(remaining_labels, key=lambda l: component_sizes[l])
                filtered_img[filtered_components == largest_component] = label_id

        bar.update()

    bar.end()
    if verbose:
        print("Filtering completed. Retained components based on size thresholds.")

    return filtered_img


def filter_by_margin(seg_img, margin=1, verbose=0):
    """
    Remove all cells in contact with the image border.
    :param seg_img: Segmented image.
    :param margin: Pixels around the border. (default: 2)
    :param verbose: Verbosity level. (default: 0)
    :return: Filtered segmentation image.
    """
    if verbose:
        print(f'{c.OKBLUE}Filtering by margin...{c.ENDC}')

    # Get the cell ids in contact with the XY border according to the margin
    border = np.zeros(seg_img.shape, dtype=bool)
    border[:, :, :margin] = True
    border[:, :, -margin:] = True
    border[:, :margin, :] = True
    border[:, -margin:, :] = True

    border_cells = np.unique(seg_img[border])

    if verbose:
        print(f'\tRemoving{c.BOLD} {len(border_cells)} {c.ENDC} cells in contact with the border')

    # Set to zero the cells in contact with the border
    for cell in border_cells:
        if cell != 0:
            seg_img[seg_img == cell] = 0

    return seg_img


def compute_most_common(lines, props):
    """
    Compute the most common value in a list.
    :param lines: Lines image.
    :param props: List of properties.
    :return: Most common value.
    """
    most_commons = []
    for p in props:
        counter = Counter(lines[p.slices].flatten())

        if len(list(counter)) == 1:
            m = list(counter)[0]
        else:
            d = {k: v for k, v in counter.items() if k != 0}
            m = max(d, key=d.get)
        most_commons.append(m)
    return most_commons


def standard_features(lines, props, centroids, centroids_labels, type, verbose=0):
    """
    Compute shape features:
        - cell_in_props
        - volumes
        - sphericities
        - original_labels
        - centroids
        - lines
        - axis_major_length
        - axis_minor_length
        - solidity
        - feret_diameter_max
    :param lines: Lines image.
    :param props: Properties.
    :param centroids: Centroids.
    :param centroids_labels: Centroids labels.
    :param type: Type of features to extract [Membrane | Nuclei].
    :param verbose: Verbosity level. (default: 0)
    :return: Dataframe with shape features.
    """
    bar = LoadingBar(len(props))
    if verbose:
        print(f'{c.OKBLUE}Computing standard features...{c.ENDC}')

    new_rows = []
    if type == 'Membrane':
        most_commons = compute_most_common(lines, props)

    for i, p in enumerate(props):
        if centroids_labels[i] == 0 or p.volume <= 20:
            continue

        if type == 'Membrane':
            new_rows.append({
                "cell_in_props": i,
                "volumes": p.volume,
                "sphericities": p.sphericity,
                "cell_id": centroids_labels[i],
                "centroids": centroids[i],
                "lines": most_commons[i],
                "axis_major_length": p.axis_major_length,
                "axis_minor_length": p.axis_minor_length,
                "solidity": p.solidity
            })
        else:
            new_rows.append({
                "cell_in_props": i,
                "volumes": p.volume,
                "sphericities": p.sphericity,
                "cell_id": centroids_labels[i],
                "centroids": centroids[i],
            })

        if verbose:
            bar.update()

    bar.end()

    return pd.DataFrame(new_rows)


def resample_img(img, mask, raw_img_path):
    """
    Resample the image and mask. The resampling is done based on the XYZ resolutions.
    :param img: Image.
    :param mask: Mask.
    :param raw_img_path: Raw image path.
    :param verbose: Verbosity level. (default: 0)
    :return: Resampled image and mask.
    """
    metadata, _ = cr.load_metadata(raw_img_path)
    spacing = [
        float(metadata['x_res']),
        float(metadata['y_res']),
        float(metadata['z_res'])
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize([
        int(round(img.GetSize()[0] * spacing[0])),
        int(round(img.GetSize()[1] * spacing[1])),
        int(round(img.GetSize()[2] * spacing[2]))
    ])

    img = resampler.Execute(img)
    mask = resampler.Execute(mask)

    return img, mask


def shape_features(img, mask):
    """
    Compute shape features.
    :param img: Image.
    :param mask: Mask.
    :return: Results list with shape features.
    """
    sf = shape.RadiomicsShape(img, mask)
    sf.enableAllFeatures()
    result = sf.execute()

    return result


def first_order_features(img, mask):
    """
    Compute first order features.
    :param img: Image.
    :param mask: Mask.
    :return: Results list with first order features.
    """
    fo = firstorder.RadiomicsFirstOrder(img, mask)
    result = fo.execute()

    return result


def extract(seg_img, raw_img, lines, raw_img_path, metadata, f_type='Nuclei', verbose=0):
    """
    Extract features from the segmented image.
    :param seg_img: Segmented image.
    :param raw_img: Raw image.
    :param lines: Lines image.
    :param raw_img_path: Path to raw image.
    :param f_type: Type of features to extract[Membrane | Nuclei]. (default: Nuclei)
    :param verbose: Verbosity level. (default: 0)
    :return: Features.
    """
    if f_type not in ['Membrane', 'Nuclei']:
        raise ValueError(f'Invalid type: {f_type}')

    if verbose:
        print(f'{c.OKBLUE}Extracting features...{c.ENDC}')

    props = ps.metrics.regionprops_3D(seg_img) # morphology.label(seg_img)
    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [seg_img[ce[0], ce[1], ce[2]] for ce in centroids]

    if verbose:
        print(f'\tFound{c.BOLD} {len(props)} {c.ENDC} cells')

    df = standard_features(lines, props, centroids, centroids_labels, f_type, verbose=verbose)

    bar = LoadingBar(len(df))

    if verbose:
        print(f'{c.OKBLUE}Computing {f_type} features...{c.ENDC}')

    results = []

    for row in df.iterrows():
        cell = row[1].cell_in_props

        img = np.swapaxes(np.swapaxes(raw_img[props[cell].slice], 0, 2), 1, 2)
        m = np.swapaxes(np.swapaxes(props[cell].mask, 0, 2), 1, 2)

        sitk_img = sitk.GetImageFromArray(img)
        sitk_img = sitk.JoinSeries(sitk_img)[:, :, :, 0]

        sitk_mask = sitk.GetImageFromArray(m.astype("uint16"))
        sitk_mask = sitk.JoinSeries(sitk_mask)[:, :, :, 0]

        sitk_img, sitk_mask = resample_img(sitk_img, sitk_mask, raw_img_path)

        try:
            shape_feats = shape_features(sitk_img, sitk_mask)

            if f_type == 'Nuclei':
                first_order_feats = first_order_features(sitk_img, sitk_mask)
                shape_feats.update(first_order_feats)

            shape_feats['cell_in_props'] = cell
            results.append(shape_feats)
        except Exception:
            print(f'{c.WARNING}Error computing features for cell {cell}{c.ENDC}')

        if verbose:
            bar.update()

    bar.end()

    df_radiomics = pd.DataFrame(results)
    df = pd.merge(df, df_radiomics, on='cell_in_props', how='left')

    return df



