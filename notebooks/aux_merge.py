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


DOWN_SAMPLE_FACTOR = 1 # 2.5

###########################################################
# 1. Feature Extraction (Recalculation) Helpers
###########################################################

def integrated_intensity(raw_img, mask):
    """
    Compute the integrated intensity (sum of intensities) for each cell.
    raw_img, mask : 3D np.ndarray with shape (X, Y, Z) in your coordinate system.
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
    Re-run your standard preprocessing + feature extraction pipeline
    on the final (merged) seg_mask.
    Saves the results to an Excel file at out_path.

    label2class : dict or None
        A dictionary mapping label_id -> (nuclei_type, brdu_status).
        If provided, we add columns 'nuclei_type' and 'brdu_status' to the final .xlsx.
    """
    print(f"{c.OKBLUE}Recomputing features for final mask...{c.ENDC}")

    # Load metadata for voxel sizes
    metadata, _ = imaging.load_metadata(raw_path)

    if DOWN_SAMPLE_FACTOR != 1:
        metadata['x_res'] = float(metadata['x_res']) / .4
        metadata['y_res'] = float(metadata['y_res']) / .4

    # Preprocess the raw image (downsample, intensity calibration, etc.)
    # so that it matches the segmentation's shape (X, Y, Z).
    proc_img = Preprocessing([
        # 'resample',
        'intensity_calibration',
        'rescale_intensity',
    ]).run(raw_path, axes='XYZ', verbose=0)  # shape (X, Y, Z)

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
            # If we have class info for this cell, store it
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
    Adjust if your file naming differs.
    """
    basename = os.path.basename(csv_path)
    # e.g. "Results_MONO_NEG.csv" -> ["Results", "MONO", "NEG.csv"]
    print(basename)
    parts = basename.split('_')
    print(parts)
    nuclei_type = parts[1]  # 'MONO' or 'BI'
    brdu_status = os.path.splitext(parts[2])[0]  # 'NEG' or 'POS'
    return nuclei_type.upper(), brdu_status.upper()

def read_annotation_csv(csv_path, downsample_factor=2.5):
    """
    Reads a CSV with columns like: Label, Area, Mean, Min, Max, X, Y, ...
    Returns a list of (scaled_x, scaled_y) after dividing by downsample_factor.

    Adjust the factor if your segmentation is scaled differently
    from the original XY dimension (3636 -> 1454 is ~2.5).
    """
    df = pd.read_csv(csv_path)
    coords = []
    for idx, row in df.iterrows():
        # Original coords (X, Y) from the full-size image
        x_orig = int(row['X'])
        y_orig = int(row['Y'])
        # Scale them down to match the segmentation shape
        x_scaled = int(round(x_orig / downsample_factor))
        y_scaled = int(round(y_orig / downsample_factor))
        coords.append((x_scaled, y_scaled))
    return coords

def find_labels_in_3d(seg_mask, x, y):
    """
    For an array with shape (X, Y, Z), we check seg_mask[x, y, :] across Z.
    Returns a set of labels found (excluding 0).
    """
    # Check bounds
    if (x < 0 or x >= seg_mask.shape[0] or
        y < 0 or y >= seg_mask.shape[1]):
        return set()
    line_along_z = seg_mask[x, y, :]  # shape (Z,)
    return set(line_along_z[line_along_z > 0])

def recompute_merge(seg_mask, labels_to_merge):
    """
    Merges all labels in labels_to_merge into one. We pick the smallest ID
    and replace all others. Return the final label used.
    """
    if not labels_to_merge:
        return seg_mask, None
    final_label = min(labels_to_merge)
    for old_label in labels_to_merge:
        seg_mask[seg_mask == old_label] = final_label
    return seg_mask, final_label

def annotate_segmentation(
    seg_mask,
    csv_files,
    out_seg_path=None,
    downsample_factor=2.5
):
    """
    - For each CSV file, parse the class (Mono/Bi, POS/NEG),
      read coords, find overlapping labels in seg_mask, merge them if needed.
    - Returns the updated seg_mask and a dictionary label2class with final classes.
    - Optionally saves the updated mask.
    """
    print(f"{c.OKBLUE}Annotating / merging labels with CSV data...{c.ENDC}")
    label2class = {}  # label_id -> (nuclei_type, brdu_status)

    for csv_path in csv_files:
        nuclei_type, brdu_status = parse_csv_filename(csv_path)
        coords = read_annotation_csv(csv_path, downsample_factor=downsample_factor)
        for (x, y) in coords:
            labels_found = find_labels_in_3d(seg_mask, x, y)
            if len(labels_found) == 0:
                continue
            elif len(labels_found) == 1:
                # Exactly one label found
                the_label = list(labels_found)[0]
                # Store the class info for this label
                label2class[the_label] = (nuclei_type, brdu_status)
            else:
                # Merge them
                seg_mask, final_label = recompute_merge(seg_mask, labels_found)
                # Store class info for final_label
                if final_label is not None:
                    label2class[final_label] = (nuclei_type, brdu_status)

    if out_seg_path:
        imaging.save_nii(seg_mask, out_seg_path, axes='XYZ')
        print(f"{c.OKGREEN}Saved updated segmentation to {out_seg_path}{c.ENDC}")

    return seg_mask, label2class

###########################################################
# 3. Main Orchestrator
###########################################################

def process_specimen(raw_path, seg_path, out_xlsx_path, csv_files):
    """
    - Reads the raw image path (for metadata & reprocessing).
    - Reads the segmentation.
    - Annotates/merges using CSV files -> returns final seg_mask & label2class.
    - Recomputes features on the final mask -> saves final .xlsx with classes.
    """
    print(f"{c.OKBLUE}Loading segmentation: {seg_path}{c.ENDC}")
    seg_mask = imaging.read_image(seg_path, axes='XYZ', verbose=1)  # shape (X, Y, Z)

    # 1) Merge based on CSV annotations
    seg_mask, label2class = annotate_segmentation(seg_mask, csv_files, downsample_factor=DOWN_SAMPLE_FACTOR)

    # 2) Recompute features on the final mask & annotate with classes
    recalc_features(raw_path, seg_mask, out_xlsx_path, label2class=label2class)

def main():
    """
    Example main function that loops over your directory structure.
    Adapt to your actual file layout & naming patterns.
    """
    _type = 'WT'
    base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/ADULT {_type}/'
    # base_dir = f"/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/ADULT {_type}/"  # or the parent directory containing ADULT WT / ADULT MYC
    # Suppose you have something like:
    #   Raw/
    #   Segmentations/
    #   Results/
    # We’ll do a minimal example for multiple specimens:
    specimens = [
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


        # 'EXW348a_BrdU001-Dapi',
        # 'EXW348a_BrdU-Dapi',
        # 'EXW348b_BrdU001-Dapi',
        # 'EXW348b_BrdU-Dapi',
        # 'EXW348c_BrdU-Dapi',
        # 'EXW367a_BrdU-Dapi',
        # 'EXW367c_BrdU-Dapi',
        # 'EXW368a_BrdU001-Dapi',
        # 'EXW368a_BrdU-Dapi',
        # 'EXW368b_BrdU001-Dapi',
        # 'EXW382_1_Brdu_001-Dapi',
        # 'EXW382_1_Brdu_002-Dapi',
        # 'EXW382_1_Brdu_003-Dapi',
        # 'EXW382_1_Brdu_-Dapi',
        # 'EXW386_1_Brdu_001-Dapi',
        # 'EXW386_1_Brdu_-Dapi',
        # 'EXW386_2_Brdu_-Dapi',
        # 'EXW387_1_Brdu_001-Dapi',
        # 'EXW387_1_Brdu_002-Dapi',
        # 'EXW387_1_Brdu_-Dapi',
        # 'EXW_WT1_1_Brdu_-Dapi',
        # 'EXW_WT1_2_Brdu_001-Dapi',
        # 'EXW_WT1_2_Brdu_002-Dapi',
        # 'EXW_WT1_2_Brdu_003-Dapi',
        # 'EXW_WT1_2_Brdu_-Dapi',
        # 'EXW_WT2_Brdu_001-Dapi',
        # 'EXW_WT2_Brdu_002-Dapi',
        # 'EXW_WT2_Brdu_-Dapi',

        'AEI_ki67_001_dapi',  # WT
        'AEI_ki67_003_dapi',
        'AEI_ki67__dapi',
        'AEI_ki67_002_dapi',
        'AEI_ki67_004_dapi',

        # 'efj806__dapi', # MYC
        # 'efj806_ki67_004_dapi',
        # 'efj806_ki67__dapi.tif',
        # 'efj806_ki67_001_dapi',
        # 'efj806_ki67_005_dapi',
        # 'efj806_ki67_002_dapi',
        # 'efj806_ki67_006_dapi',
        # 'efj806_ki67_003_dapi',
        # 'efj806_ki67_007_dapi',
    ]
    for name in specimens:
        try:
            raw_path = os.path.join(base_dir, "Raw", f"{name}.tif")
            seg_path = os.path.join(base_dir, "Segmentations", f"{name}_mask.nii.gz")
            # We'll save the final .xlsx with appended classes
            out_xlsx_path = os.path.join(base_dir, "Results", name, f"{name}_annotated.xlsx")

            # Gather CSV files in the same folder, e.g.:
            #   Results_MONO_NEG.csv, Results_BI_POS.csv, etc.
            results_dir = os.path.join(base_dir, "Results", name)
            csv_files = []
            for fname in os.listdir(results_dir):
                if fname.endswith(".csv") and "Results_" in fname:
                    csv_files.append(os.path.join(results_dir, fname))

            # Process
            if len(csv_files) == 0:
                print(f"{c.WARNING}No CSV files found for specimen:{c.ENDC} {name}")
                continue

            process_specimen(raw_path, seg_path, out_xlsx_path, csv_files)
        except Exception as e:
            print(f"{c.FAIL}Error processing specimen:{c.ENDC} {name}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()