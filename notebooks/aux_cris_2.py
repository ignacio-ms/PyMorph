#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import warnings
from rich.progress import Progress

# Image processing and segmentation libraries
from skimage.measure import regionprops
from stardist.models import StarDist3D, StarDist2D
from csbdeep.utils import normalize

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from nuclei_segmentation.my_cellpose import load_img, load_model
from nuclei_segmentation.processing.preprocessing import Preprocessing
from util.data import imaging
from util.gpu.gpu_tf import increase_gpu_memory, set_gpu_allocator
from util.misc.colors import bcolors as c

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# =============================================================================
# PARAMETERS
# =============================================================================
# Segmentation and cropping parameters
CROP_SIZE = 100  # Crop a 100x100 pixel region around each annotation (XY)
HALF_CROP = CROP_SIZE // 2
MERGE_STRATEGY = "overlap"  # Options: "overlap" or "separate"

# (Keep your original parameters below)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/"

SEG_MODEL_TYPE = 'stardist'  # or 'stardist'
DO_3D = False
CELLPOSE_DIAMETER = None
CELLPROB_THRESHOLD = 0.0
STITCH_THRESHOLD = 0.1
FLOW_THRESHOLD = 0.5
DOWNSAMPLE_FACTOR = 1 # 2.5

# Pipeline parameters (as before)
PIPELINE = [
    # 'resample',
    'remove_bck',
    'intensity_calibration',
    'rescale_intensity',
    # 'equalization',
    # 'norm_adaptative',
    'bilateral',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_stardist_model(do_3D):
    try:
        increase_gpu_memory()
        set_gpu_allocator()
    except Exception as e:
        print(f'{c.WARNING}GPU mem. growth not available{c.ENDC}')
    if do_3D:
        _model_name = 'n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)'
        _model_path = os.path.join(current_dir, "..", "models", "stardist_models")
        return StarDist3D(None, name=_model_name, basedir=_model_path)
    _model_name = '2D_versatile_fluo'
    return StarDist2D.from_pretrained(_model_name)


def snitch_labels(mask, threshold=0.1):
    """
    As in your original script: stitches 2D label images into a 3D volume.
    """
    X, Y, Z = mask.shape
    # print(f'{c.OKGREEN}Mask shape{c.ENDC}: {mask.shape}')
    stitched = np.zeros_like(mask, dtype=np.int32)
    current_max_label = 0
    slice0 = mask[:, :, 0]
    unique_labels = np.unique(slice0)
    unique_labels = unique_labels[unique_labels != 0]
    for lab in unique_labels:
        current_max_label += 1
        stitched[:, :, 0][slice0 == lab] = current_max_label

    for z in range(1, Z):
        prev_slice = stitched[:, :, z - 1]
        curr_slice = mask[:, :, z]
        stitched_slice = np.zeros_like(curr_slice, dtype=np.int32)
        props = regionprops(curr_slice)
        for prop in props:
            if prop.label == 0:
                continue
            coords = prop.coords
            area = prop.area
            prev_labels = prev_slice[coords[:, 0], coords[:, 1]]
            counts = np.bincount(prev_labels)
            if len(counts) > 0:
                counts[0] = 0
            if counts.sum() == 0:
                current_max_label += 1
                new_lab = current_max_label
            else:
                best_lab = np.argmax(counts)
                best_count = counts[best_lab]
                if best_count / area >= threshold:
                    new_lab = best_lab
                else:
                    current_max_label += 1
                    new_lab = current_max_label
            stitched_slice[coords[:, 0], coords[:, 1]] = new_lab
        stitched[:, :, z] = stitched_slice

    return stitched


def read_annotation_csv(csv_path, downsample_factor=2.5):
    """
    Reads a CSV file (with annotated cell coordinates). Assumes the CSV contains columns 'X' and 'Y'.
    Returns a list of (x, y) coordinates (integers).
    """
    df = pd.read_csv(csv_path)
    coords = []
    for idx, row in df.iterrows():
        x_orig = float(row['X'])
        y_orig = float(row['Y'])
        x_scaled = int(round(x_orig / downsample_factor))
        y_scaled = int(round(y_orig / downsample_factor))
        coords.append((x_scaled, y_scaled))
    return coords


def crop_roi(image, x_center, y_center):
    """
    Given a 3D image array (assumed shape (Z, Y, X)) and a center coordinate (x, y) in XY,
    crop a region of size CROP_SIZE x CROP_SIZE in XY, keeping all Z slices.
    If the crop extends beyond image borders, it is clamped.
    Returns the ROI and the (x_min, y_min) offset in the full image.
    """
    X, Y, Z = image.shape
    x_min = max(0, x_center - HALF_CROP)
    x_max = min(X, x_center + HALF_CROP)
    y_min = max(0, y_center - HALF_CROP)
    y_max = min(Y, y_center + HALF_CROP)
    roi = image[x_min:x_max, y_min:y_max, :]
    return roi, (x_min, y_min)


def merge_roi_masks(full_mask, roi_mask, offset, current_label, merge_strategy="overlap"):
    """
    Merges a ROI segmentation mask (roi_mask) into the full_mask.
    - full_mask: the global segmentation mask (3D array) for the entire image.
    - roi_mask: segmentation result for the ROI (2D or 3D array). If 2D, assume it applies to all Z slices.
    - offset: tuple (x_min, y_min) indicating where the ROI was cropped from the full image.
    - current_label: current maximum label value used (int).
    - merge_strategy: "overlap" (default) or "separate".

    Returns:
      updated full_mask and updated current_label.
    """
    x_off, y_off = offset
    # Determine ROI dimensions (assume roi_mask shape is either (Z, h, w) or (h, w) for a single slice)
    if roi_mask.ndim == 2:
        roi_mask = np.expand_dims(roi_mask, axis=0)
    w, h, Z = roi_mask.shape
    # Extract the region from the full mask corresponding to the ROI
    # (Assume full_mask shape is (Z, Y, X) and ROI applies to all Z slices)
    full_region = full_mask[x_off:x_off + w, y_off:y_off + h, :]
    # Process each unique object in the roi_mask (ignoring background 0)
    for lab in np.unique(roi_mask):
        if lab == 0:
            continue
        roi_obj = (roi_mask == lab)
        # Check overlap in the corresponding region of full_mask
        overlap = full_region[roi_obj]
        overlap_labels = np.unique(overlap)
        overlap_labels = overlap_labels[overlap_labels != 0]
        if merge_strategy == "overlap" and overlap_labels.size > 0:
            # Merge: assign the smallest overlapping label to roi_obj region
            merge_lab = int(np.min(overlap_labels))
            full_region[roi_obj] = merge_lab
        else:
            # Assign a new unique label
            current_label += 1
            full_region[roi_obj] = current_label
    # Write the updated region back
    full_mask[x_off:x_off + w, y_off:y_off + h, :] = full_region
    return full_mask, current_label


# =============================================================================
# SEGMENTATION FUNCTION
# =============================================================================
def process_specimen(raw_path, seg_out_path, csv_files, preproc_dir, model_type='cellpose', **kwargs):
    """
    For a given specimen, process the corresponding preprocessed image:
      - Load the preprocessed image from PREPROCESSED_DIR (filename based on raw_path).
      - For each CSV annotation file (which contains XY coordinates), crop a 100x100 ROI.
      - Run segmentation (using the existing pipeline) on each ROI.
      - Merge all segmented ROIs into a single segmentation mask for the full image.
      - Save the final segmentation mask to seg_out_path (overwriting the existing one).

    kwargs are passed to the segmentation model eval (e.g. diameter, channels, etc.)
    """
    # Use the filename from raw_path to find the corresponding preprocessed image.
    img_filename = os.path.basename(raw_path)
    # preproc_path = os.path.join(preproc_dir, img_filename)
    # if not os.path.exists(preproc_path):
    #     print(f"{c.WARNING}Preprocessed image not found for {img_filename}{c.ENDC}")
    #     return
    print(raw_path)
    img = load_img(
        raw_path,
        PIPELINE,
        axes='ZYX' if model_type == 'cellpose' else 'XYZ',
        verbose=kwargs.get("verbose", 0)
    )
    # img = imaging.read_image(preproc_path, axes='ZYX' if model_type == 'cellpose' else 'XYZ')

    # Assume image shape is (Z, Y, X)
    X, Y, Z = img.shape
    # Initialize a global segmentation mask (zeros)
    full_mask = np.zeros_like(img, dtype=np.uint16)
    global_label = 0

    # Load segmentation model
    if model_type == 'cellpose':
        model = load_model(model_type='nuclei')
    elif model_type == 'stardist':
        model = load_stardist_model(do_3D=kwargs.get("do_3D", False))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Process each CSV file to get annotated coordinates
    for csv_path in csv_files:
        coords = read_annotation_csv(csv_path, downsample_factor=DOWNSAMPLE_FACTOR)
        with Progress() as progress:
            task = progress.add_task(f"Processing {csv_path.split('/')[-1]}", total=len(coords))
            for (x_coord, y_coord) in coords:
                # Crop ROI (100x100) around (x_coord, y_coord); note: these coordinates are in the segmented (preprocessed) image scale.
                roi, offset = crop_roi(img, x_coord, y_coord)
                # Run segmentation on this ROI using the same pipeline as before.
                if model_type == 'cellpose':
                    # Using cellpose eval as in your original code
                    mask_roi, _, _, _ = model.eval(
                        roi,
                        diameter=kwargs.get("diameter", CELLPOSE_DIAMETER),
                        channels=kwargs.get("channels", [0, 0]),
                        normalize=kwargs.get("normalize", False),
                        anisotropy=kwargs.get("anisotropy", None),
                        do_3D=kwargs.get("do_3D", False),
                        cellprob_threshold=kwargs.get("cellprob_threshold", CELLPROB_THRESHOLD),
                        stitch_threshold=kwargs.get("stitch_threshold", STITCH_THRESHOLD),
                        flow_threshold=kwargs.get("flow_threshold", FLOW_THRESHOLD),
                    )
                elif model_type == 'stardist':
                    # For StarDist, similar processing (adjust as needed)
                    # If ROI is 3D but using 2D model, process slice by slice.
                    if not kwargs.get("do_3D", False):
                        mask_slices = []
                        for z in range(roi.shape[-1]):
                            slice_img = roi[..., z]
                            mask_slice, _ = model.predict_instances(
                                slice_img,
                                n_tiles=(8, 8),
                                show_tile_progress=False,
                                axes='XY',
                            )
                            mask_slices.append(mask_slice)
                        # Stack slices in Z and optionally use snitch_labels to merge
                        mask_slices = np.swapaxes(mask_slices, 0, 2)
                        mask_roi = snitch_labels(mask_slices, threshold=STITCH_THRESHOLD)
                    else:
                        mask_roi, _ = model.predict_instances(
                            roi,
                            n_tiles=(8, 8, 1),
                            show_tile_progress=False,
                            axes='XYZ',
                        )
                else:
                    raise ValueError("Invalid model type")

                # Now, merge the ROI segmentation into the full mask.
                full_mask, global_label = merge_roi_masks(full_mask, mask_roi, offset, global_label,
                                                          merge_strategy=MERGE_STRATEGY)
                progress.update(task, advance=1)

    # Save the final segmentation mask (overwriting the existing one)
    imaging.save_nii(full_mask, seg_out_path, axes='ZYX' if model_type == 'cellpose' else 'XYZ',
                     verbose=kwargs.get("verbose", 0))
    print(f"{c.OKGREEN}Saved merged segmentation mask to {seg_out_path}{c.ENDC}")


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    # For each specimen folder (e.g., in ADULT WT and ADULT MYC) process images that have annotations.
    # Adjust the base directory as needed.
    base_dir = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/'
    # base_dir = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/'
    # Process both ADULT WT and ADULT MYC:
    for specimen_type in ['ADULT WT']: # , 'ADULT MYC'
        specimen_dir = os.path.join(base_dir, specimen_type)
        raw_dir = os.path.join(specimen_dir, "Raw")
        preproc_dir = os.path.join(raw_dir, "Processed")
        results_dir = os.path.join(specimen_dir, "Results")
        seg_out_dir = os.path.join(specimen_dir, "Segmentations")
        os.makedirs(seg_out_dir, exist_ok=True)
        # List all preprocessed .tif files (or .nii.gz if that is your convention)
        for fname in os.listdir(raw_dir):
            if not fname.lower().endswith('.tif'):
                continue
            img_name = os.path.splitext(fname)[0]
            if img_name.endswith('.tif'):
                img_name = img_name.replace('.tif', '')

            if img_name.endswith('_preprocessed'):
                img_name = img_name.replace('_preprocessed', '')

            # Check if there is a corresponding Results subfolder with CSV annotation files:
            img_results_dir = os.path.join(results_dir, img_name)
            if not os.path.isdir(img_results_dir):
                print(f"{c.WARNING}Skipping {img_name} (no Results folder){c.ENDC}")
                continue
            csv_files = [os.path.join(img_results_dir, f) for f in os.listdir(img_results_dir)
                         if f.endswith(".csv") and "Results_" in f]
            if len(csv_files) == 0:
                print(f"{c.WARNING}Skipping {img_name} (no annotation CSVs){c.ENDC}")
                continue
            # Define paths:
            raw_path = os.path.join(raw_dir, fname)
            seg_out_path = os.path.join(seg_out_dir, f"{img_name}_mask.nii.gz")
            print(f"{c.OKBLUE}Processing {img_name} with {len(csv_files)} annotation file(s)...{c.ENDC}")
            try:
                process_specimen(raw_path, seg_out_path, csv_files, preproc_dir, model_type=SEG_MODEL_TYPE,
                                 diameter=CELLPOSE_DIAMETER,
                                 anisotropy=None, cellprob_threshold=CELLPROB_THRESHOLD,
                                 stitch_threshold=STITCH_THRESHOLD,
                                 flow_threshold=FLOW_THRESHOLD, normalize=False, pipeline=PIPELINE, do_3D=DO_3D, verbose=1)
            except Exception as e:
                print(f"{c.FAIL}Error processing {img_name}:{c.ENDC} {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(__file__)
    except NameError:
        current_dir = os.getcwd()
    main()