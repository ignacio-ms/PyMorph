import os
import sys

import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
from rich.progress import Progress

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging as cr
from util.misc.colors import bcolors as c  # for printing colored messages (if desired)
from nuclei_segmentation.processing.preprocessing import Preprocessing
from util.data import imaging


def integrated_intensity(raw_img, mask):
    """
    Compute the integrated intensity (sum of intensities) for each cell.

    Parameters:
        raw_img : np.ndarray
            Preprocessed raw image.
        mask : np.ndarray
            Segmentation mask with cell labels.

    Returns:
        dict mapping cell id to integrated intensity.
    """
    intensity_dict = {}
    cell_ids = np.unique(mask)
    for cell in cell_ids:
        if cell == 0:
            continue
        intensity_dict[cell] = raw_img[mask == cell].sum()
    return intensity_dict


def volume(mask):
    """
    Compute the cell volume in pixels (i.e., the number of voxels).

    Parameters:
        mask : np.ndarray
            Segmentation mask with cell labels.

    Returns:
        dict mapping cell id to voxel count.
    """
    vol_dict = {}
    cell_ids = np.unique(mask)
    for cell in cell_ids:
        if cell == 0:
            continue
        vol_dict[cell] = (mask == cell).sum()
    return vol_dict


def vox2micron(metadata):
    """
    Compute the volume conversion factor (voxel volume in µm³) based on raw image resolutions.

    Parameters:
        metadata : dict
            Dictionary containing the voxel resolutions with keys: 'x_res', 'y_res', 'z_res'.

    Returns:
        voxel_volume : float
            Voxel volume in cubic microns.
    """
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    voxel_volume = x_res * y_res * z_res  # µm³ per voxel
    return voxel_volume


def run(raw_path, seg_path, out_path, name):
    _type = 'WT'
    data_path = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/ADULT {_type}/'
    out_path = data_path + out_path
    raw_path = data_path + raw_path
    seg_path = data_path + seg_path
    print(f"{c.OKBLUE}Loading raw image and segmentation mask...{c.ENDC}")

    # mkdir Results/{name} check
    if not os.path.exists(f'{data_path}Results/{name}'):
        os.makedirs(f'{data_path}Results/{name}')

    # Load metadata (e.g., voxel resolutions) from the raw image:
    metadata, _ = cr.load_metadata(raw_path)

    # Change metadata accordingly to resample of .4 zoom factor for X and Y
    metadata['x_res'] = float(metadata['x_res']) / .4
    metadata['y_res'] = float(metadata['y_res']) / .4

    seg_mask = imaging.read_image(seg_path, axes='XYZ', verbose=1)
    proc_img = Preprocessing([
        'resample',
        'intensity_calibration',
        # 'rescale_intensity',
    ]).run(raw_path, axes='XYZ', verbose=0)

    # Compute per-cell integrated intensity and volume (in pixels)
    int_dict = integrated_intensity(proc_img, seg_mask)
    vol_dict = volume(seg_mask)

    # Get conversion factor: voxel volume in µm³.
    voxel_vol = vox2micron(metadata)

    # For each cell, compute:
    #   - energy: integrated intensity divided by physical volume (µm³)
    #   - volume_microns: volume in µm³ (voxel count * voxel_vol)
    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Computing features...", total=len(int_dict))
        for cell in int_dict.keys():
            integrated_int = int_dict[cell]
            vol_pixels = vol_dict[cell]
            vol_micron = vol_pixels * voxel_vol
            energy = integrated_int / vol_micron if vol_micron != 0 else np.nan
            results.append({
                "cell_id": cell,
                "integrated_intensity": integrated_int,
                "energy": energy,
                "volume_pixels": vol_pixels,
                "volume_microns": vol_micron
            })
            progress.update(task, advance=1)

    # Save results to Excel using pandas
    df = pd.DataFrame(results)
    df.to_excel(out_path, index=False)
    print(f"{c.OKGREEN}Features saved to {out_path}{c.ENDC}")
1

if __name__ == '__main__':
    for name in [
        # 'EXW_1_BrdU001-Dapi',
        # 'EXW_1_BrdU002-Dapi',
        # 'EXW_1_BrdU003-Dapi',
        # 'EXW_1_BrdU004-Dapi',
        # 'EXW_1_BrdU006-Dapi',
        # 'EXW_1_BrdU007-Dapi',
        # 'EXW_1_BrdU-Dapi',
        # 'EXW_2_BrdU001-Dapi',
        # 'EXW_2_BrdU002-Dapi',
        # 'EXW_2_BrdU003-Dapi',
        # 'EXW_2_BrdU004-Dapi',
        # 'EXW_2_BrdU-Dapi',
        # 'EXW321_1_Brdu_001-Dapi',
        # 'EXW321_1_Brdu_002-Dapi',
        # 'EXW321_1_Brdu_-Dapi',
        # 'EXW_321b_Brdu_001-Dapi',
        # 'EXW_321b_Brdu_-Dapi',
        # 'EXW_324_Brdu__001-Dapi',
        # 'EXW_324_Brdu__002-Dapi',
        # 'EXW_324_Brdu__003-Dapi',
        # 'EXW_324_Brdu__004-Dapi',
        # 'EXW_324_Brdu__-Dapi',
        # 'EXW_324_Brdu___-Dapi',
        # 'EXW345_1_Brdu_002-Dapi',
        # 'EXW345_1_Brdu_003-Dapi',
        # 'EXW345_1_Brdu_004-Dapi',
        # 'EXW352_1_Brdu_-Dapi',
        # 'EXW_352_Brdu__001-Dapi',
        # 'EXW_352_Brdu__002-Dapi',
        # 'EXW_352_Brdu__-Dapi',
        # 'EXW370a(MYC)_BrdU001-Dapi',
        # 'EXW370a(MYC)_BrdU-Dapi',
        # 'EXW370b(MYC)_BrdU002-Dapi',
        # 'EXW370b(MYC)_BrdU003-Dapi',
        # 'EXW_3a_BrdU001-Dapi',
        # 'EXW_3a_BrdU-Dapi',
        # 'EXW_3b_BrdU-Dapi',
        # 'EXW_3c_BrdU001-Dapi',
        # 'EXW_3c_BrdU002-Dapi',
        # 'EXW_3c_BrdU003-Dapi',
        # 'EXW45_1_Brdu_001-Dapi',
        # 'EXW45_1_Brdu_-Dapi',


        'EXW348a_BrdU001-Dapi',
        'EXW348a_BrdU-Dapi',
        'EXW348b_BrdU001-Dapi',
        'EXW348b_BrdU-Dapi',
        'EXW348c_BrdU-Dapi',
        'EXW367a_BrdU-Dapi',
        'EXW367c_BrdU-Dapi',
        'EXW368a_BrdU001-Dapi',
        'EXW368a_BrdU-Dapi',
        'EXW368b_BrdU001-Dapi',
        'EXW382_1_Brdu_001-Dapi',
        'EXW382_1_Brdu_002-Dapi',
        'EXW382_1_Brdu_003-Dapi',
        'EXW382_1_Brdu_-Dapi',
        'EXW386_1_Brdu_001-Dapi',
        'EXW386_1_Brdu_-Dapi',
        'EXW386_2_Brdu_-Dapi',
        'EXW387_1_Brdu_001-Dapi',
        'EXW387_1_Brdu_002-Dapi',
        'EXW387_1_Brdu_-Dapi',
        'EXW_WT1_1_Brdu_-Dapi',
        'EXW_WT1_2_Brdu_001-Dapi',
        'EXW_WT1_2_Brdu_002-Dapi',
        'EXW_WT1_2_Brdu_003-Dapi',
        'EXW_WT1_2_Brdu_-Dapi',
        'EXW_WT2_Brdu_001-Dapi',
        'EXW_WT2_Brdu_002-Dapi',
        'EXW_WT2_Brdu_-Dapi',
    ]:
        raw_path = f"Raw/{name}.tif"
        seg_path = f"Segmentations/{name}_mask.nii.gz"
        out_path = f"Results/{name}/{name}.xlsx"

        try:
            run(raw_path, seg_path, out_path, name)
        except Exception as e:
            print(f"{c.FAIL}Error processing specimen:{c.ENDC} {name}")
