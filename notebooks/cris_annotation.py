import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.transform import rescale


try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging
from util.misc.colors import bcolors as c
from nuclei_segmentation.processing.preprocessing import Preprocessing

def rescale_raw_to_seg_shape(raw_img, seg_img):
    """
    Rescale the raw image (shape: (Xorig, Yorig, Z)) to match the segmentation shape (shape: (Xdown, Ydown, Z)).
    Assumes the Z-dimension is identical.
    """
    Xorig, Yorig, Z = raw_img.shape
    Xdown, Ydown, Zseg = seg_img.shape
    if Z != Zseg:
        raise ValueError(f"Z-dimensions do not match: raw Z={Z} vs seg Z={Zseg}")

    scale_x = Xdown / Xorig
    scale_y = Ydown / Yorig
    scaled_raw = np.zeros((Xdown, Ydown, Z), dtype=raw_img.dtype)

    for z in range(Z):
        slice_raw = raw_img[..., z]  # shape (Xorig, Yorig)
        slice_scaled = rescale(slice_raw, (scale_x, scale_y),
                               order=1,            # bilinear interpolation
                               preserve_range=True,
                               anti_aliasing=False)
        scaled_raw[..., z] = slice_scaled.astype(raw_img.dtype)

    return scaled_raw

def create_mip_overlay(raw_path, seg_path, annotated_excel, out_image_path, text_fontsize=8):
    """
    - Reads raw image (full resolution, shape: (Xorig, Yorig, Z)) and segmentation mask (downscaled, shape: (Xdown, Ydown, Z)).
    - Rescales raw image to match segmentation shape.
    - Computes maximum intensity projections (MIP) over Z.
    - Overlays segmentation boundaries on the raw MIP.
    - Annotates cells (only those present in the annotated Excel) with small text showing cell ID and biological label.
    - Saves the final overlay as a .tif using the imaging module.
    """
    if not os.path.isfile(raw_path) or not os.path.isfile(seg_path):
        print(f"{c.WARNING}Missing raw or segmentation file: {raw_path} / {seg_path}{c.ENDC}")
        return
    if not os.path.isfile(annotated_excel):
        print(f"{c.WARNING}Annotated Excel not found: {annotated_excel}{c.ENDC}")
        return

    # Load raw image (full resolution) and segmentation mask (downscaled)
    raw_img = Preprocessing([
        'rescale_intensity',
    ]).run(
        raw_path, axes='XYZ', verbose=0,
        out_range=(0, 255)
    ).astype(np.int8)  # shape: (Xorig, Yorig, Z)
    seg_mask = imaging.read_image(seg_path, axes='XYZ', verbose=0) # shape: (Xdown, Ydown, Z)

    # Rescale raw image to match segmentation shape (X,Y)
    scaled_raw = rescale_raw_to_seg_shape(raw_img, seg_mask)

    # Compute MIPs over Z
    mip_raw = np.max(scaled_raw, axis=2)  # shape: (Xdown, Ydown)
    mip_seg = np.max(seg_mask, axis=2)    # shape: (Xdown, Ydown)

    # Create overlay using segmentation boundaries
    overlay = mark_boundaries(mip_raw, mip_seg.astype(np.int32), color=(1,0,0), mode='thick')

    # Load annotated Excel to get biological labels for each cell
    df = pd.read_excel(annotated_excel)
    if 'cell_id' not in df.columns:
        print(f"{c.WARNING}Column 'cell_id' missing in {annotated_excel}. No labels will be drawn.{c.ENDC}")
        annotated_ids = set()
        label_dict = {}
    else:
        annotated_ids = set(df['cell_id'].dropna().astype(int))
        # Build a dictionary: cell_id -> "nuclei_type brdu_status"
        label_dict = {}
        for _, row in df.iterrows():
            if pd.isna(row.get('cell_id')):
                continue
            cid = int(row['cell_id'])
            nuc_type = str(row.get('nuclei_type', ''))
            brdu_status = str(row.get('brdu_status', ''))
            label_dict[cid] = f"{nuc_type} {brdu_status}".strip()

    # Compute region properties on the segmentation MIP
    props = regionprops(mip_seg.astype(int))

    # Plot overlay with annotations
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay, cmap='gray')

    for prop in props:
        lab_id = prop.label
        if lab_id in annotated_ids:
            # regionprops returns centroid as (row, col)
            row_cent, col_cent = prop.centroid
            # Compose label text: cell ID and biological label
            bio_text = label_dict.get(lab_id, "")
            text = f"{lab_id}: {bio_text}"
            ax.text(col_cent, row_cent, text, color='yellow', fontsize=text_fontsize, weight='bold')

    ax.axis('off')
    plt.tight_layout()

    # Convert figure to an image array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    overlay_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    # Save as .tif using imaging module; assume axes 'YXC' (height, width, channels)
    imaging.save_prediction(overlay_img, out_image_path, axes='YXC')
    print(f"{c.OKGREEN}MIP overlay saved to {out_image_path}{c.ENDC}")


def annotate_all_specimens():
    """
    Scans 'ADULT WT' and 'ADULT MYC' directories for subfolders,
    tries to locate raw image (Dapi.tif), segmentation mask (Dapi_mask.nii.gz),
    and annotated Excel file (Dapi_annotated.xlsx), then creates a MIP overlay.

    Adjust naming patterns to your actual file naming & directory structure.
    """
    bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/'
    base_dirs = [
        f"{bas_path}ADULT WT",
        f"{bas_path}ADULT MYC"
    ]
    for bd in base_dirs:
        # For example, each specimen might be recognized by "xxx-Dapi.tif" in "Raw/" subdir
        # We'll do a simple pattern check:
        for root, dirs, files in os.walk(bd):
            for f in files:
                if f.endswith(".tif") and "Dapi" in f:
                    # Potential raw image
                    raw_path = os.path.join(root, f)
                    base_name = f.replace(".tif","")
                    print(f'{c.OKBLUE}Processing specimen:{c.ENDC} {base_name}')
                    print(f"{c.OKBLUE}Raw image path:{c.ENDC} {raw_path}")

                    # Guess segmentation path (downscaled)
                    seg_name = base_name + "_mask.nii.gz"
                    seg_path = os.path.join(bd, "Segmentations", seg_name)
                    print(f"{c.OKBLUE}Segmentation path:{c.ENDC} {seg_path}")

                    # Guess annotated excel path
                    ann_name = base_name + "_annotated.xlsx"
                    ann_path = os.path.join(bd, "Results", base_name, ann_name)
                    print(f"{c.OKBLUE}Annotated Excel path:{c.ENDC} {ann_path}")

                    # Output path
                    out_image_path = os.path.join(bd, "Results", base_name, f"{base_name}_mip_overlay.tif")

                    if os.path.isfile(raw_path) and os.path.isfile(seg_path) and os.path.isfile(ann_path):
                        try:
                            create_mip_overlay(raw_path, seg_path, ann_path, out_image_path)
                        except Exception as e:
                            print(f"{c.FAIL}Error processing specimen:{c.ENDC} {base_name}")
                            continue
                    else:
                        print(f"{c.WARNING}Skipping specimen:{c.ENDC} raw={raw_path}, seg={seg_path}, ann={ann_path}")

def main():
    annotate_all_specimens()

if __name__ == "__main__":
    main()
