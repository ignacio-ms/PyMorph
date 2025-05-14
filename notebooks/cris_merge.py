#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from rich.progress import Progress

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging
from util.misc.colors import bcolors as c
from nuclei_segmentation.processing.preprocessing import Preprocessing

DOWN_SAMPLE_FACTOR = 1  # 2.5


###########################################################
# 1. Feature Extraction (Recalculation) Helpers
###########################################################

def integrated_intensity(raw_img, mask):
    """
    Compute the integrated intensity (sum of intensities) for each cell.
    raw_img, mask : 3D np.ndarray with shape (X, Y, Z).
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
    Number of voxels per cell (i.e., volume in pixels).
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
    Compute voxel volume in µm³ based on raw image resolutions: x_res, y_res, z_res.
    """
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    return x_res * y_res * z_res  # µm³ per voxel


def recalc_features(raw_path, seg_mask, out_path, label2class=None):
    """
    Re-run your standard preprocessing + feature extraction pipeline on the final (merged) seg_mask.
    Saves the results to an Excel file at out_path.
    """
    print(f"{c.OKBLUE}Recomputing features for final mask...{c.ENDC}")

    # Load metadata for voxel sizes
    metadata, _ = imaging.load_metadata(raw_path)

    if DOWN_SAMPLE_FACTOR != 1:
        metadata['x_res'] = float(metadata['x_res']) / .4
        metadata['y_res'] = float(metadata['y_res']) / .4

    # Preprocess the raw image to match the segmentation's shape (X, Y, Z).
    proc_img = Preprocessing([
        'intensity_calibration',
        'rescale_intensity',
    ]).run(raw_path, axes='XYZ', verbose=0)

    # Extract features
    int_dict = integrated_intensity(proc_img, seg_mask)
    vol_dict = volume(seg_mask)
    voxel_vol = vox2micron(metadata)

    results = []
    cell_ids = sorted(int_dict.keys())
    with Progress() as progress:
        task = progress.add_task("[cyan]Computing features...", total=len(cell_ids))
        for cell in cell_ids:
            integrated_int = int_dict[cell]
            vol_pixels = vol_dict[cell]
            vol_micron = vol_pixels * voxel_vol
            energy = integrated_int / vol_micron if vol_micron != 0 else np.nan
            row = {
                "cell_id": cell,
                "integrated_intensity": integrated_int,
                "energy": energy,
                "volume_pixels": vol_pixels,
                "volume_microns": vol_micron
            }
            if label2class is not None and cell in label2class:
                nuclei_type, brdu_status = label2class[cell]
                row["nuclei_type"] = nuclei_type
                row["brdu_status"] = brdu_status
            results.append(row)
            progress.update(task, advance=1)

    df = pd.DataFrame(results)
    df.to_excel(out_path, index=False)
    print(f"{c.OKGREEN}Features saved to {out_path}{c.ENDC}")


###########################################################
# 2. Annotation & Merging
###########################################################

def parse_csv_filename(csv_path):
    """
    Example: 'Results_MONO_NEG.csv' -> nuclei_type='MONO', brdu_status='NEG'.
    """
    basename = os.path.basename(csv_path)
    parts = basename.split('_')
    nuclei_type = parts[1]
    brdu_status = os.path.splitext(parts[2])[0]
    return nuclei_type.upper(), brdu_status.upper()


def read_annotation_csv(csv_path, downsample_factor=2.5):
    """
    Reads a CSV with columns: Label, Area, Mean, Min, Max, X, Y, ...
    Returns a list of (scaled_x, scaled_y) after dividing by downsample_factor.
    """
    df = pd.read_csv(csv_path)
    coords = []
    for idx, row in df.iterrows():
        x_orig = int(row['X'])
        y_orig = int(row['Y'])
        x_scaled = int(round(x_orig / downsample_factor))
        y_scaled = int(round(y_orig / downsample_factor))
        coords.append((x_scaled, y_scaled))
    return coords


def find_labels_in_3d(seg_mask, x, y):
    """
    For a 3D array (X, Y, Z), check seg_mask[x, y, :] across Z.
    Returns a set of labels (excluding 0).
    """
    if (x < 0 or x >= seg_mask.shape[0] or
            y < 0 or y >= seg_mask.shape[1]):
        return set()
    line_along_z = seg_mask[x, y, :]
    return set(line_along_z[line_along_z > 0])


def recompute_merge(seg_mask, labels_to_merge):
    """
    Merges all labels in labels_to_merge into one (choosing the smallest ID).
    Returns the updated seg_mask and the final label.
    """
    if not labels_to_merge:
        return seg_mask, None
    final_label = min(labels_to_merge)
    for old_label in labels_to_merge:
        seg_mask[seg_mask == old_label] = final_label
    return seg_mask, final_label


def annotate_segmentation(seg_mask, csv_files, out_seg_path=None, downsample_factor=2.5):
    """
    For each CSV file, parse the class, read coords, find overlapping labels in seg_mask,
    merge them if needed, and record a label2class mapping.
    Optionally saves the updated seg_mask.
    """
    print(f"{c.OKBLUE}Annotating/merging labels with CSV data...{c.ENDC}")
    label2class = {}
    for csv_path in csv_files:
        nuclei_type, brdu_status = parse_csv_filename(csv_path)
        coords = read_annotation_csv(csv_path, downsample_factor=downsample_factor)
        for (x, y) in coords:
            labels_found = find_labels_in_3d(seg_mask, x, y)
            if len(labels_found) == 0:
                continue
            elif len(labels_found) == 1:
                the_label = list(labels_found)[0]
                label2class[the_label] = (nuclei_type, brdu_status)
            else:
                seg_mask, final_label = recompute_merge(seg_mask, labels_found)
                if final_label is not None:
                    label2class[final_label] = (nuclei_type, brdu_status)
    if out_seg_path:
        imaging.save_nii(seg_mask, out_seg_path, axes='XYZ')
        print(f"{c.OKGREEN}Saved updated segmentation to {out_seg_path}{c.ENDC}")
    return seg_mask, label2class


###########################################################
# 3. Main Orchestrator - Automatic Analysis
###########################################################
def process_specimen(raw_path, seg_path, out_xlsx_path, csv_files):
    """
    Loads the raw image (for metadata and preprocessing), the segmentation mask,
    annotates and merges labels using CSV files, and then recalculates features.
    """
    print(f"{c.OKBLUE}Loading segmentation: {seg_path}{c.ENDC}")
    seg_mask = imaging.read_image(seg_path, axes='XYZ', verbose=1)
    seg_mask, label2class = annotate_segmentation(seg_mask, csv_files, downsample_factor=DOWN_SAMPLE_FACTOR)
    recalc_features(raw_path, seg_mask, out_xlsx_path, label2class=label2class)


def main():
    """
    Automatically processes each specimen for which annotation CSV files exist.
    It scans the Results folder for each specimen in the specified base directory.
    """
    _type = 'WT'
    base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/ADULT {_type}/'
    # base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/ADULT {_type}/'
    # Define subfolders:
    raw_dir = os.path.join(base_dir, "Raw")
    seg_dir = os.path.join(base_dir, "Segmentations")
    results_dir = os.path.join(base_dir, "Results")

    # Automatically loop over specimen folders in the Results directory
    for specimen in os.listdir(results_dir):
        specimen_path = os.path.join(results_dir, specimen)
        if not os.path.isdir(specimen_path):
            continue
        # Gather CSV annotation files from specimen's Results folder
        csv_files = [os.path.join(specimen_path, f) for f in os.listdir(specimen_path)
                     if f.endswith(".csv") and "Results_" in f]
        if len(csv_files) == 0:
            print(f"{c.WARNING}No CSV files found for specimen:{c.ENDC} {specimen}")
            continue
        # Define file paths:
        raw_path = os.path.join(raw_dir, f"{specimen}.tif")
        seg_path = os.path.join(seg_dir, f"{specimen}_mask.nii.gz")
        out_xlsx_path = os.path.join(specimen_path, f"{specimen}_annotated.xlsx")
        print(f"{c.OKBLUE}Processing specimen {specimen} with {len(csv_files)} annotation file(s)...{c.ENDC}")
        try:
            process_specimen(raw_path, seg_path, out_xlsx_path, csv_files)
        except Exception as e:
            print(f"{c.FAIL}Error processing specimen {specimen}:{c.ENDC} {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()