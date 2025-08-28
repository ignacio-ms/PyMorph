
from scipy.spatial import ConvexHull
from scipy import ndimage as ndi
import porespy as ps
import numpy as np
import pandas as pd

import os

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from util.misc.timer import LoadingBar
from util.data import imaging
from scipy.ndimage import binary_fill_holes
OVERLAP_THRESHOLD = 0.8


_skip_existing = False
_type = 'GDO enh2' # EBI TFP    FPN enh7-3        GDO enh2
base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_FSC/LAB/PERSONAL/imarcoss/LabMT/CovaBlasto/3.5E'
seg_dir = os.path.join(base_dir, 'Segmentation', _type)
out_hull_dir = os.path.join(base_dir, 'Segmentation', 'Tissue', _type)
out_seg_dir = os.path.join(base_dir, 'Results', _type)
csv_path = os.path.join(base_dir, 'Results', _type, 'intersected_cells.csv')


def get_coords(segmentation: np.ndarray) -> np.ndarray:
    """Get cell centroids"""
    props = ps.metrics.regionprops_3D(segmentation)
    return np.array([prop.centroid for prop in props])

def convex_hull_3d(segmentation: np.ndarray):
    """
    Compute the convex hull of a 3D segmentation mask.
    """
    # coords = np.column_stack(np.nonzero(segmentation))
    coords = get_coords(segmentation)
    hull = ConvexHull(coords)

    mins = coords.min(axis=0); mins = np.floor(mins).astype(int)
    maxs = coords.max(axis=0); maxs = np.ceil(maxs).astype(int)
    # Build grid in bounding box of centroids
    zs = np.arange(mins[0], maxs[0] + 1)
    ys = np.arange(mins[1], maxs[1] + 1)
    xs = np.arange(mins[2], maxs[2] + 1)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
    pts = np.vstack((zz.ravel(), yy.ravel(), xx.ravel())).T

    # Test points against hull planes
    eqs = hull.equations  # shape (n_facets, 4)
    inside = np.all(eqs[:, :-1].dot(pts.T) + eqs[:, -1][:, None] <= 1e-8, axis=0)
    region = inside.reshape(zz.shape)
    hull_mask = np.zeros_like(segmentation, dtype=bool)
    hull_mask[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1] = region
    # Fill any holes to ensure a closed mask
    hull_mask = binary_fill_holes(hull_mask)
    hull_mask = ndi.binary_dilation(hull_mask, structure=np.ones((3, 3, 3)))
    cleaned = segmentation.copy()
    for label in np.unique(segmentation):
        if label == 0:
            continue
        total = np.sum(segmentation == label)
        inside = np.sum((segmentation == label) & hull_mask)
        if inside / total < OVERLAP_THRESHOLD:
            cleaned[segmentation == label] = 0
    return cleaned, hull_mask


def _vox2micron(metadata):
    """
    Convert voxel dimensions to microns.
    """
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    return x_res * y_res * z_res


def summary(img, metadata, name):
    df = pd.read_csv(csv_path)

    cell_ids = np.unique(img)
    cell_ids = cell_ids[cell_ids > 0]

    volumes = {cell_id: np.sum(img == cell_id) for cell_id in cell_ids}
    volumes_mu = {cell_id: vol * _vox2micron(metadata) for cell_id, vol in volumes.items()}

    data = {
        'img': name,
        'n_cells': len(cell_ids),
        'cell_ids': list(cell_ids),
        'cell_volumes': [volumes_mu[cell_id] for cell_id in cell_ids]
    }

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(csv_path, index=False)

def main(input_path: str, output_path_seg: str, output_path_hull: str):
    """
    Main function to compute the convex hull of a 3D segmentation mask.

    Parameters
    ----------
    input_path : str
        Path to the input 3D segmentation mask in NIfTI format.
    output_path : str
        Path to save the output convex hull mask in NIfTI format.
    """
    # Load the segmentation mask
    segmentation = imaging.read_image(input_path, axes='XYZ').astype(np.uint32)
    hull_mask, contour_mask = convex_hull_3d(segmentation)

    _meta, _ = imaging.load_metadata(input_path, z_res='2.9997855')
    summary(hull_mask, _meta, os.path.basename(input_path).replace('.nii.gz', ''))

    imaging.save_nii(hull_mask, output_path_seg)
    imaging.save_nii(contour_mask.astype(np.uint8), output_path_hull)


if __name__ == "__main__":
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['img', 'n_cells', 'cell_ids', 'cell_volumes'])
        df.to_csv(csv_path, index=False)

    print(f'{_type} Embryo Hull Segmentation:')
    bar = LoadingBar(len(os.listdir(seg_dir)))

    for img in os.listdir(seg_dir):
        try:
            if not img.endswith('.tif') and not img.endswith('.nii.gz'):
                continue
            # if not img in ['EBI603_6_dapi_mask.nii.gz']: continue

            img_path = os.path.join(seg_dir, img)
            out_hull_path = os.path.join(out_hull_dir, img.replace('.nii.gz', '_hull_mask.nii.gz'))
            out_seg_path = os.path.join(out_seg_dir, img.replace('.nii.gz', '_filtered_hull.nii.gz'))
            if _skip_existing and os.path.exists(out_seg_path):
                print(f'Skipping {img} as output already exists.')
                bar.update()
                continue
            # print(f'Processing {img}...')
            main(img_path, out_seg_path, out_hull_path)
            # print(f'Saved hull mask to {out_path}')
            bar.update()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'Error processing {img}: {e}')
            bar.update()
            continue
    bar.end()
