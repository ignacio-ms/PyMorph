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
from auxiliary.utils.timer import LoadingBar

from filtering import cardiac_region as cr


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
        print(f'Pre-filtering cells: {c.BOLD}{len(np.unique(seg_img))}{c.ENDC}')

    props = ps.metrics.regionprops_3D(morphology.label(seg_img))
    centroids = [[round(i) for i in p.centroid] for p in props]
    centroids_labels = [seg_img[ce[0], ce[1], ce[2]] for ce in centroids]

    if verbose:
        print(f'\tFound{c.BOLD} {len(centroids)} {c.ENDC} cells')

    new_rows = []
    remove = []

    for i, p in enumerate(props):
        if centroids_labels[i] == 0 or p.volume <= 10:
            remove.append(centroids_labels[i])
            continue

        new_rows.append({
            "volumes": p.volume,
            "original_labels": centroids_labels[i],
        })

    df = pd.DataFrame(new_rows)

    lower_bound = np.percentile(df.volumes, 100 - percentile)
    upper_bound = np.percentile(df.volumes, percentile)

    remove += df[df.volumes > upper_bound].original_labels.tolist()
    remove += df[df.volumes < lower_bound].original_labels.tolist()

    remove = set(remove)

    if verbose:
        print(f'\tRemoving {c.BOLD}{len(remove)}{c.ENDC} cells...')

    seg_membrane_int = seg_img.astype(int)
    remove_int = np.array(list(remove)).astype(int)

    mask = np.isin(seg_membrane_int, remove_int)
    seg_img[mask] = 0

    print(f'Post-filtering cells: {c.BOLD}{len(np.unique(seg_img))}{c.ENDC}')

    return seg_img


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
                "original_labels": centroids_labels[i],
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
                "original_labels": centroids_labels[i],
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


def extract(seg_img, raw_img, lines, raw_img_path, f_type='Nuclei', verbose=0):
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

    props = ps.metrics.regionprops_3D(morphology.label(seg_img))
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

        result = {}
        if f_type == 'Membrane' or f_type == 'Nuclei':
            result.update(shape_features(sitk_img, sitk_mask))
        if f_type == 'Nuclei':
            result.update(first_order_features(sitk_img, sitk_mask))

        result['original_labels'] = row[1].original_labels
        result['cell_in_props'] = cell
        results.append(result)  # :)

        if verbose:
            bar.update()

    bar.end()

    df_radiomics = pd.DataFrame(results)
    df = pd.merge(df, df_radiomics, on='cell_in_props', how='left')
    return df



