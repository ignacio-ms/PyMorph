# Standard packages
import os
import sys

import cv2
import numpy as np

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.data.imaging import (
    read_nii, read_tiff,
    save_prediction, load_metadata
)
from auxiliary.utils.timer import LoadingBar
from auxiliary.utils.colors import bcolors as c
import auxiliary.values as v


def get_margins(line_path, img_path, tissue=None, ma=5, resolution=1024, verbose=0):
    """
    Get margins around the line to crop the image.
    :param line_path: Path to line image.
    :param img_path: Path to image to be cropped.
    :param tissue: Tissue to filter by. (Default: None)
    :param ma: Margin around the line to crop. (Default: 5)
    :param resolution: Resolution of image. (Default: 1024)
    :param verbose: Verbosity level.
    :return:
    """
    metadata_line = load_metadata(line_path)
    line = read_nii(line_path)
    metadata_img = load_metadata(img_path)

    if verbose:
        print(f'{c.BOLD}Margin around line{c.ENDC}: {ma}')
        print(f'{c.BOLD}Line metadata{c.ENDC}: {metadata_line}')
        print(f'{c.BOLD}Line shape{c.ENDC}: {line.shape}')

    try:
        coords = np.where(line == v.lines[tissue] if tissue else line > 1)
    except KeyError:
        print(f'{c.FAIL}Invalid tissue{c.ENDC}: {tissue}')
        print(f'{c.BOLD}Available tissues{c.ENDC}: {list(v.lines.keys())}')
        sys.exit(2)

    margins = (
        np.min(coords, axis=1),
        np.max(coords, axis=1)
    )

    for i in range(3):
        margins[0][i] = (
            margins[0][i] - ma
            if margins[0][i] - ma > 0
            else 0
        )
        margins[1][i] = (
            margins[1][i] + ma
            if margins[1][i] + ma < resolution
            else resolution
        )

        if i == 2:
            margins[1][i] = (
                margins[1][i] + ma
                if margins[1][i] + ma < metadata_img["z_size"]
                else metadata_img["z_size"]
            )

    return margins


def crop_img(img, margins, verbose=0):
    """
    Crop image.
    :param img: Path to image to be cropped or already loaded image.
    :param margins: Margins to crop the image.
    :param verbose: Verbosity level.
    :return: Cropped image.
    """
    if isinstance(img, str):
        img = read_nii(img) if img.endswith('.nii.gz') else read_tiff(img)

    img = img[
        int(margins[0][0]):int(margins[1][0]),
        int(margins[0][1]):int(margins[1][1]),
        int(margins[0][2]):int(margins[1][2])
    ]

    if verbose:
        print(f'{c.BOLD}Cropped image shape{c.ENDC}: {img.shape}')

    return img


def filter_by_tissue(img, lines, tissue_name='myocardium', dilate=0, dilate_size=3, verbose=0):
    """
    Filter image by tissue.
    :param img: Segmented image to be filtered.
    :param lines: Lines image.
    :param tissue_name: Tissue to filter by. (Default: myocardium)
        Available tissues:
        - background - myocardium - embryo - pocket - somatic - splanchnic - proximal - aort
        - lumen - middle plane - keep dorsal open myo - keep dorsal open spl
        - notochord
    :param dilate: Dilation of the mask. (Default: 0)
    :param dilate_size: Size of the dilation kernel. Must be an odd number. (Default: 3)
    :param verbose: Verbosity level.
    :return: Filtered image.
    """
    if verbose:
        print(f'{c.OKBLUE}Filtering image by tissue{c.ENDC}: {tissue_name}...')

    try:
        tissue = v.lines[tissue_name]
        if img.ndim == 4:
            img = img[..., 0]
        filtered = np.zeros_like(img)

        if dilate and dilate_size:
            if verbose:
                print(f'{c.BOLD}Dilating mask{c.ENDC}: ({dilate_size}x{dilate_size}) {dilate} times...')

            ds = dilate_size if dilate_size % 2 else dilate_size + 1
            kernel = np.ones((ds, ds), np.uint8)
            if lines.ndim == 2:
                lines = cv2.dilate(lines, kernel, iterations=dilate)
            else:
                for z in range(lines.shape[-1]):
                    lines[..., z] = cv2.dilate(lines[..., z], kernel, iterations=dilate)

        bar = LoadingBar(lines.shape[-1])

        cell_ids = np.unique(img[lines == tissue])

        for z in range(lines.shape[-1]):
            if verbose:
                bar.update()

            mask = np.isin(img[..., z], cell_ids)
            filtered[..., z] = np.where(mask, img[..., z], 0)

        bar.end()

    except KeyError:
        print(f'{c.FAIL}Invalid tissue{c.ENDC}: {tissue_name}')
        print(f'{c.BOLD}Available tissues{c.ENDC}: {list(v.lines.keys())}')
        sys.exit(2)

    return filtered


def main():
    line_path = v.data_path + 'Gr1/Segmentation/LinesTissue/line_20190521_E2.nii.gz'
    img_path = v.data_path + 'Gr1/Segmentation/Nuclei/20190521_E2_Fernando.tif'

    img = read_tiff(img_path, verbose=1)
    lines = read_nii(line_path, verbose=1)

    filtered_img = filter_by_tissue(
        img, lines, tissue_name='myocardium',
        dilate=1, dilate_size=3,
        verbose=1
    )
    save_prediction(filtered_img, 'filtering/filtered_example_d1.3.tif', verbose=1)


if __name__ == '__main__':
    main()
