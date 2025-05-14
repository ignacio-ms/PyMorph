#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import warnings
from rich.progress import Progress
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
CROP_SIZE = 100             # 100x100 pixel crop around each annotation (XY)
HALF_CROP = CROP_SIZE // 2
MERGE_STRATEGY = "overlap"  # Options: "overlap" or "separate"
ANNOTATION_MARGIN = 10      # Margin around annotated coordinates (in pixels)

# QA overlay parameters
QA_RADIUS = 2              # Radius for red circle overlay (in pixels)
QA_ALPHA = 0.3             # Transparency for segmentation overlay

# Folder for preprocessed images (relative to Raw folder)
PREPROCESSED_DIR = "Processed"

# Model and segmentation parameters
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/bin/"
SEG_MODEL_TYPE = 'stardist'  # or 'cellpose'
DO_3D = False
CELLPOSE_DIAMETER = None
CELLPROB_THRESHOLD = 0.0
STITCH_THRESHOLD = 0.1
FLOW_THRESHOLD = 0.5
DOWNSAMPLE_FACTOR = 1      # Adjust if needed (e.g., 2.5)

# Pipeline parameters (unchanged)
PIPELINE = [
    'remove_bck',
    # 'resample',
    'intensity_calibration',
    'rescale_intensity',
    'bilateral',
]

###########################################################
# 1. Feature Extraction (Recalculation) Helpers
###########################################################

def integrated_intensity(raw_img, mask):
    """
    Compute the integrated intensity (sum of intensities) for each cell.
    Both raw_img and mask must have the same shape (X, Y, Z).
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
    Compute number of voxels (volume in pixels) per cell.
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
    Compute voxel volume (in µm³) from metadata resolutions.
    """
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    return x_res * y_res * z_res

###########################################################
# 2. UTILITY FUNCTIONS
###########################################################

def parse_csv_filename(csv_path):
    """
    Parse CSV filename to extract nuclei type and brdu status.
    Example: 'Results_MONO_NEG.csv' -> ('MONO', 'NEG')
    """
    basename = os.path.basename(csv_path)
    parts = basename.split('_')
    nuclei_type = parts[1]
    brdu_status = os.path.splitext(parts[2])[0]
    return nuclei_type.upper(), brdu_status.upper()

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
    Stitch 2D label images into a 3D volume.
    """
    X, Y, Z = mask.shape
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

def read_annotation_csv(csv_path, metadata, scale_factor=1.0):
    """
    Reads a CSV file with columns 'X' and 'Y' and returns a list of (x, y) coordinates,
    scaled by 'scale_factor' if the image was resampled in XY by that factor.
    """
    df = pd.read_csv(csv_path)
    coords = []
    for idx, row in df.iterrows():
        x_orig = float(row['X'])
        y_orig = float(row['Y'])
        # Scale the original coordinates if the image was downsampled/resampled
        scale_factor_x = metadata['x_res']
        scale_factor_y = metadata['y_res']
        x_scaled = int(round(x_orig / scale_factor_x))
        y_scaled = int(round(y_orig / scale_factor_y))

        x_scaled = int(round(x_scaled * scale_factor))
        y_scaled = int(round(y_scaled * scale_factor))
        coords.append((x_scaled, y_scaled))
    return coords

def crop_roi(image, x_center, y_center):
    """
    Crop a region of size CROP_SIZE x CROP_SIZE (all Z slices) from image (assumed shape (X, Y, Z))
    centered at (x_center, y_center). Returns the ROI and the offset (x_min, y_min).
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
    Merge the ROI segmentation mask into the global full_mask.
    """
    x_off, y_off = offset
    if roi_mask.ndim == 2:
        roi_mask = np.expand_dims(roi_mask, axis=0)
    w, h, Z = roi_mask.shape
    full_region = full_mask[x_off:x_off+w, y_off:y_off+h, :]
    for lab in np.unique(roi_mask):
        if lab == 0:
            continue
        roi_obj = (roi_mask == lab)
        overlap = full_region[roi_obj]
        overlap_labels = np.unique(overlap)
        overlap_labels = overlap_labels[overlap_labels != 0]
        if merge_strategy == "overlap" and overlap_labels.size > 0:
            merge_lab = int(np.min(overlap_labels))
            full_region[roi_obj] = merge_lab
        else:
            current_label += 1
            full_region[roi_obj] = current_label
    full_mask[x_off:x_off+w, y_off:y_off+h, :] = full_region
    return full_mask, current_label

def generate_qa_overlay(raw_path, seg_mask, csv_files, metadata, qa_radius=QA_RADIUS, qa_alpha=QA_ALPHA):
    """
    Generate a QA image: maximum projection of the raw image with segmentation overlay,
    and red circles (of radius qa_radius) at each annotated coordinate.
    Saves the QA PNG in the same folder as raw_path.
    """
    raw_img = imaging.read_image(raw_path, axes='XYZ', verbose=0)
    mip_raw = np.swapaxes(np.max(raw_img, axis=2), 0, 1)
    mip_seg = np.swapaxes(np.max(seg_mask, axis=2), 0, 1)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(mip_raw, cmap='gray')
    ax.imshow(mip_seg, cmap='jet', alpha=qa_alpha)
    for csv_path in csv_files:
        coords = read_annotation_csv(csv_path, metadata, scale_factor=1)
        for (x, y) in coords:
            circ = Circle((x, y), radius=qa_radius, edgecolor='red', facecolor='none', linewidth=2)
            txt = f'{x},{y}'
            ax.add_patch(circ)
            ax.text(x + 1, y + 1, txt, color='red', fontsize=6)
    ax.set_axis_off()
    raw_folder = os.path.dirname(raw_path)
    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    out_png = os.path.join(raw_folder, f"{base_name}_qa.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"{c.OKGREEN}QA overlay saved to {out_png}{c.ENDC}")

###########################################################
# 3. SEGMENTATION & FEATURE EXTRACTION FUNCTION
###########################################################
def process_specimen(raw_path,
                     seg_out_path,
                     out_xlsx_path,
                     csv_files,
                     preproc_dir,
                     model_type='cellpose',
                     use_existing_mask=False,  # NEW: optional
                     **kwargs):
    """
    For a given specimen:
      - Optionally load an existing mask if use_existing_mask=True and seg_out_path file exists.
      - Load the preprocessed image from the Processed folder.
      - For each CSV annotation file, crop a CROP_SIZE x CROP_SIZE ROI centered on the scaled annotation coordinate.
        * If the annotation is already covered by the existing mask, skip segmenting.
      - Run segmentation on each ROI (if not skipped).
      - Merge newly segmented cells into the full (possibly pre-existing) mask.
      - Extract features only for newly segmented cells.
      - Save the final segmentation mask and merged features.
      - Generate a QA overlay image.
    """
    # 1) Possibly load existing segmentation if file exists and user asked for it
    if use_existing_mask and os.path.isfile(seg_out_path):
        print(f"{c.OKBLUE}[INFO] Loading existing segmentation mask for speedup...{c.ENDC}")
        existing_mask = imaging.read_image(seg_out_path, axes='ZYX', verbose=0)
        full_mask = existing_mask.copy()
        current_label = np.max(full_mask)
    else:
        # Create empty
        full_mask = None
        current_label = 0

    # 2) Load preprocessed image
    img_filename = os.path.basename(raw_path)
    preproc_path = os.path.join(preproc_dir, img_filename.replace('.tif', '_preprocessed.tif'))
    from_preproc = False
    img = load_img(
        preproc_path if from_preproc else raw_path,
        PIPELINE,
        axes='ZYX' if model_type=='cellpose' else 'XYZ',
        verbose=1,
        save=True,
    )
    X, Y, Z = img.shape
    if full_mask is None:
        full_mask = np.zeros_like(img, dtype=np.uint16)

    # Also load a raw version (with partial pipeline) for features
    raw_img = load_img(
        raw_path,
        pipeline=[
            # 'resample',
            'intensity_calibration',
            'rescale_intensity',
        ],
        axes='XYZ', verbose=1
    )
    SCALE_FACTOR = 1.0 # 0.67
    # DOWNSAMPLE_FACTOR = 1.0 / SCALE_FACTOR
    metadata, _ = imaging.load_metadata(raw_path)
    # metadata['x_res'] /= DOWNSAMPLE_FACTOR
    # metadata['y_res'] /= DOWNSAMPLE_FACTOR

    global_label = current_label
    features_list = []

    print(f'Raw image shape: {raw_img.shape}')
    print(f'Preprocessed image shape: {img.shape}')

    # 3) Load segmentation model
    if model_type == 'cellpose':
        model = load_model(model_type='nuclei')
    elif model_type == 'stardist':
        model = load_stardist_model(do_3D=kwargs.get("do_3D", False))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 4) Process CSV files
    total_coords = 0
    for csv_path in csv_files:
        these_coords = read_annotation_csv(csv_path, metadata, scale_factor=SCALE_FACTOR)
        total_coords += len(these_coords)
    for csv_path in csv_files:
        coords = read_annotation_csv(csv_path, metadata, scale_factor=SCALE_FACTOR)
        nuclei_type, brdu_status = parse_csv_filename(csv_path)

        with Progress() as progress:
            task = progress.add_task(f"Processing {os.path.basename(csv_path)}", total=len(coords))
            for (x_coord, y_coord) in coords:
                # 4a) Skip if existing mask covers this center (use_existing_mask==True)
                if use_existing_mask and np.uint16(full_mask.shape[0])>x_coord>=0 and np.uint16(full_mask.shape[1])>y_coord>=0:
                    # Only do this check if coords are in range
                    existing_label_at_point = full_mask[x_coord, y_coord, 0]
                    if existing_label_at_point != 0:
                        # Already segmented
                        progress.update(task, advance=1)
                        continue

                # 4b) Crop ROI
                roi, offset = crop_roi(img, x_coord, y_coord)
                if model_type == 'cellpose':
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
                    if not kwargs.get("do_3D", False):
                        mask_slices = []
                        for z in range(roi.shape[-1]):
                            slice_img = roi[..., z]
                            mask_slice, _ = model.predict_instances(
                                slice_img,
                                n_tiles=(1, 1),
                                show_tile_progress=False,
                                axes='XY',
                            )
                            mask_slices.append(mask_slice)
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

                roi_center = np.array([CROP_SIZE // 2, CROP_SIZE // 2])
                mask_roi = mask_roi.astype(np.uint8)
                unique_labels = np.unique(mask_roi)
                unique_labels = unique_labels[unique_labels != 0]

                if len(unique_labels) == 0:
                    print(f"{c.WARNING}No segmented cells in ROI centered at ({x_coord},{y_coord}).{c.ENDC}")
                    progress.update(task, advance=1)
                    continue

                # 4c) Find the "best label" near the annotation
                best_label = None
                best_dist = None
                for lab in unique_labels:
                    props = regionprops((mask_roi == lab).astype(np.int8))
                    for prop in props:
                        centroid = prop.centroid
                        dist = np.linalg.norm(roi_center - np.array(centroid)[:-1])
                        if best_dist is None or dist < best_dist:
                            best_label = lab
                            best_dist = dist

                if best_label is None:
                    print(f"{c.WARNING}No segmented cells in ROI centered at ({x_coord},{y_coord}).{c.ENDC}")
                    progress.update(task, advance=1)
                    continue

                annotated_cell_mask = (mask_roi == best_label)
                # 4d) Crop from raw_img for feature extraction
                raw_roi, _ = crop_roi(raw_img, x_coord, y_coord)

                int_dict = integrated_intensity(raw_roi, annotated_cell_mask)
                vol_dict = volume(annotated_cell_mask)
                voxel_vol = vox2micron(metadata)

                cell_ids = sorted(int_dict.keys())
                if not cell_ids:
                    progress.update(task, advance=1)
                    continue
                cell_id = cell_ids[0]
                integrated_int = int_dict[cell_id]
                vol_px = vol_dict[cell_id]
                vol_mic = vol_px * voxel_vol
                energy = integrated_int / vol_mic if vol_mic != 0 else np.nan

                feature_record = {
                    "specimen": os.path.basename(raw_path),
                    "ROI_center_x": x_coord,
                    "ROI_center_y": y_coord,
                    "cell_id": cell_id,
                    "integrated_intensity": integrated_int,
                    "energy": energy,
                    "volume_pixels": vol_px,
                    "volume_microns": vol_mic,
                    "nuclei_type": nuclei_type,
                    "brdu_status": brdu_status,
                }
                features_list.append(feature_record)

                # 4e) Merge new mask into full_mask
                full_mask, global_label = merge_roi_masks(full_mask, (mask_roi == best_label).astype(np.uint8),
                                                          offset, global_label, merge_strategy=MERGE_STRATEGY)
                progress.update(task, advance=1)

    # 5) Save final segmentation
    imaging.save_nii(full_mask, seg_out_path, axes='ZYX' if model_type=='cellpose' else 'XYZ',
                     verbose=kwargs.get("verbose", 0))
    print(f"{c.OKGREEN}Saved merged segmentation mask to {seg_out_path}{c.ENDC}")

    # 6) Save extracted features
    if features_list:
        df_features = pd.DataFrame(features_list)
        df_features.to_excel(out_xlsx_path, index=False)
        print(f"{c.OKGREEN}Saved annotated features to {out_xlsx_path}{c.ENDC}")
    else:
        print(f"{c.WARNING}No annotated features extracted.{c.ENDC}")

    # 7) QA overlay
    generate_qa_overlay(raw_path, full_mask, csv_files, metadata, qa_radius=QA_RADIUS, qa_alpha=QA_ALPHA)
    print(f'{c.OKBLUE}Segmented instances/total number of annotated cells: {len(features_list)} / {total_coords}{c.ENDC}')


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """
    Automatically processes each specimen for which annotation CSV files exist.
    It scans the Results folder for each specimen in the specified base directory.
    """
    _type = 'MYC'
    base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/ADULT {_type}/'
    # base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Copy/Nacho/ADULT {_type}/'
    raw_dir = os.path.join(base_dir, "Raw")
    seg_dir = os.path.join(base_dir, "Segmentations")
    results_dir = os.path.join(base_dir, "Results")

    # Loop over specimen folders in the Results directory
    # print(os.listdir(results_dir))
    # print(os.listdir(results_dir)[::-1])
    # n = len(os.listdir(results_dir))
    # dirlist = os.listdir(results_dir)[int(n/2):]
    # n_p = len(dirlist)
    # dirlist = dirlist[:int(n_p/2)]
    # dirlist = dirlist[::-1]
    # print(dirlist)
    dirlist = os.listdir(results_dir)
    for specimen in dirlist:
        if specimen in ['AEI1_dapi', 'AEI1_001_dapi', 'AEI2_dapi', 'AEI2_001_dapi', 'AEI2_002_dapi', 'AEI2_003_dapi']: # ['AEI2_dapi']: #
            print(f"{c.WARNING}Skipping specimen: {c.ENDC}{specimen}")
            continue
        specimen_path = os.path.join(results_dir, specimen)
        if not os.path.isdir(specimen_path):
            continue
        csv_files = [os.path.join(specimen_path, f) for f in os.listdir(specimen_path)
                     if f.endswith(".csv") and "Results_" in f]
        if len(csv_files) == 0:
            print(f"{c.WARNING}No CSV files found for specimen: {specimen}{c.ENDC}")
            continue
        raw_path = os.path.join(raw_dir, f"{specimen}.tif")
        seg_out_path = os.path.join(seg_dir, f"{specimen}_mask.nii.gz")
        out_xlsx_path = os.path.join(specimen_path, f"{specimen}_annotated.xlsx")
        print(f"{c.OKBLUE}Processing specimen {specimen} with {len(csv_files)} annotation file(s)...{c.ENDC}")
        try:
            # pass "use_existing_mask=True" if we want to skip re-segmenting existing cells
            process_specimen(raw_path,
                             seg_out_path,
                             out_xlsx_path,
                             csv_files,
                             preproc_dir=os.path.join(raw_dir, PREPROCESSED_DIR),
                             model_type=SEG_MODEL_TYPE,
                             diameter=CELLPOSE_DIAMETER,
                             anisotropy=None,
                             cellprob_threshold=CELLPROB_THRESHOLD,
                             stitch_threshold=STITCH_THRESHOLD,
                             flow_threshold=FLOW_THRESHOLD,
                             normalize=False,
                             pipeline=PIPELINE,
                             do_3D=DO_3D,
                             verbose=1,
                             use_existing_mask=False)  # <--- toggle here
        except Exception as e:
            print(f"{c.FAIL}Error processing specimen {specimen}:{c.ENDC} {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(__file__)
    except NameError:
        current_dir = os.getcwd()
    main()
