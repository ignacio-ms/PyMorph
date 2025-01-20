import sys
import os

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import numpy as np
from timagetk.components.labelled_image import LabelledImage
from timagetk.components.spatial_image import SpatialImage

import matplotlib.pyplot as plt

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging


def relabel_seg_img(img_seg, mapping_dict, ignore_missing_value=True):
    # provided solution @Divakar
    def map_values(a, bad_vals, update_vals):
        N = max(a.max(), max(bad_vals)) + 1
        mapar = np.empty(N, dtype=int)
        mapar[a] = a
        mapar[bad_vals] = update_vals
        out = mapar[a]
        return out

    # - Get type of the input
    dtype = 'uint16'

    # - Get image
    I = img_seg.get_array().copy().astype(dtype)

    # - Get mask of the missing values
    if ignore_missing_value:
        missing_val = list(set(img_seg.labels()) - set(mapping_dict))
        mask = np.isin(I, missing_val)

    # - Replace the value in image
    k = np.array(list(mapping_dict.keys()))
    v = np.array(list(mapping_dict.values()))

    I = map_values(I, k, v)

    if ignore_missing_value:
        I[mask] = img_seg.not_a_label

    return LabelledImage(SpatialImage(
        I, voxelsize=img_seg.voxelsize, dtype=dtype
    ), not_a_label=img_seg.not_a_label)


def visualize_comparison(
        target_img, reference_img, out_results, axis='z', slice_id=None,
        BACKGROUND_LABEL=1, save=False
):
    # fonction that determines the segmentation comparison state
    color_map = {
        'missing': 5, 'background': 4, 'correct': 0,
        'under_segmented': 2, 'over_segmented': 1, 'confused': 3
    }

    def segmentation_color(row):
        row_name = f'state'
        return color_map[row[row_name]]

    if slice_id is None:
        if axis == 'x':
            slice_id = int(reference_img.shape[0] / 2)
        elif axis == 'y':
            slice_id = int(reference_img.shape[1] / 2)
        else:
            slice_id = int(reference_img.shape[2] / 2)

    if target_img.is3D():
        target_img = LabelledImage(target_img.get_slice(slice_id, axis), not_a_label=0)

    if reference_img.is3D():
        reference_img = LabelledImage(reference_img.get_slice(slice_id, axis), not_a_label=0)

    vxs = reference_img.voxelsize
    im_extent = reference_img.extent
    mpl_extent = (-vxs[1] / 2, im_extent[1] + vxs[1] / 2, im_extent[0] + vxs[0] / 2, -vxs[0] / 2)

    # - Relabel target image according to the segmentation errors
    out_results['segmentation_color'] = out_results.apply(segmentation_color, axis=1)
    mapping_dict = {lab: c for lab_list, c in zip(out_results.target.values, out_results.segmentation_color.values) for
                    lab in lab_list}
    mapping_dict[BACKGROUND_LABEL] = 6
    mapping_dict = {**mapping_dict, **{lab: 5 for lab in target_img.labels() if lab not in mapping_dict}}

    target_comparison = relabel_seg_img(target_img, mapping_dict)

    if save:
        imaging.save_nii(target_comparison, 'target_comparison.nii.gz')

    fig, ax = plt.subplots(1, 3, figsize=(12, 12))

    list_img = [target_img, target_comparison, reference_img]
    title = ['Target Image', 'Segmentation Comparison', 'Reference Image']
    custom_map = ListedColormap(["tab:green", "tab:red", "tab:blue", "tab:gray", "lightgray", "yellow", "w"])
    cmap = ['prism', custom_map, 'prism']

    val_range = [(0, 255), (0, 7), (0, 255)]

    for ix, img in enumerate(list_img):
        ax[ix].imshow(img.get_array() % 255, cmap=cmap[ix], vmin=val_range[ix][0], vmax=val_range[ix][1],
                      extent=mpl_extent, interpolation='none')
        ax[ix].set_title(title[ix])

        # - add borders
        if ix in [0, 2]:
            for lab in img.labels():
                if lab != 1:
                    ax[ix].contour(img.get_array() == lab, linewidths=0.1, extent=mpl_extent, origin='upper',
                                   colors='k')
        else:
            for lab in list_img[0].labels():
                if lab != 1:
                    ax[ix].contour(list_img[0].get_array() == lab, linewidths=0.1, extent=mpl_extent, origin='upper',
                                   colors='k')

    fig.subplots_adjust(bottom=0.05, wspace=0.33)

    # - Add custom legend
    legend_elements = [Patch(facecolor='tab:green', edgecolor='k', label='Correct'),
                       Patch(facecolor='tab:red', edgecolor='k', label='Over-segmentation'),
                       Patch(facecolor='tab:blue', edgecolor='k', label='Under-segmentation'),
                       Patch(facecolor='tab:gray', edgecolor='k', label='Misc.'),
                       Patch(facecolor='lightgray', edgecolor='k', label='In background')]

    fig.axes[1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0, -1, 1, 1), ncol=2)

    fig.tight_layout()

    return fig


def transform_image(img):
    img_aux = img.copy()
    img_aux = img_aux + 1

    return LabelledImage(
        img_aux, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )


def save_comparison(target_img, out_results, out_path):
    target_img = transform_image(target_img)

    color_map = {
        'missing': [255, 255, 255],
        'background': [0, 0, 0],
        'correct': [0, 255, 0],
        'under_segmented': [0, 0, 255],
        'over_segmented': [255, 0, 0],
        'confused': [255, 255, 0]
    }

    def segmentation_color(row):
        row_name = f'state'
        return color_map[row[row_name]]

    out_results['segmentation_color'] = out_results.apply(segmentation_color, axis=1)

    mapping = {
        lab: c for lab_list, c
        in zip(out_results.target.values, out_results.segmentation_color.values)
        for lab in lab_list
    }
    mapping[1] = [0, 0, 0]
    mapping = {
        **mapping, **{
            lab: [255, 255, 255] for lab in target_img.labels()
            if lab not in mapping
        }
    }

    target_comparison = np.empty(target_img.shape + (3,), dtype=np.uint8)
    for lab in target_img.labels():
        target_comparison[target_img == lab] = mapping[lab]

    imaging.save_tiff_imagej_compatible(
        out_path, target_comparison,
        axes='XYZC'
    )
