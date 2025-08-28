import os
import subprocess
import sys

import numpy as np
import pandas as pd

from skimage.transform import rescale

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.data import imaging
from util.misc.colors import bcolors as c


_type = 'GDO' # EBI TFP      FPN enh7-3      GDO enh2
base_dir = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_FSC/LAB/PERSONAL/imarcoss/LabMT/CovaBlasto/6.5E_2/'
raw_dir = os.path.join(base_dir, _type)
cells_dir = os.path.join(base_dir, 'Segmentation', _type)
tissue_dir = os.path.join(base_dir, 'Segmentation', 'Tissue', 'Interpolated', _type)

out_dir = os.path.join(base_dir, 'Results', _type)
out_path = os.path.join(out_dir, 'intersected_cells.csv')

_skip_existing = True


def isotropy(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

    if 'image' in kwargs:
        img = kwargs['image']

    metadata = kwargs['metadata']
    resampling_factor = metadata['z_res'] / metadata['x_res']

    img_iso = rescale(
        img, (1, 1, resampling_factor),
        order=0, mode='constant',
        anti_aliasing=False,
        preserve_range=True
    )

    print(
        f'\t{c.OKBLUE}Image resolution{c.ENDC}: \n'
        f'\tX: {metadata["x_res"]} um/px\n'
        f'\tY: {metadata["y_res"]} um/px\n'
        f'\tZ: {metadata["z_res"]} um/px'
    )
    print(f'\tResampling factor: {resampling_factor}')
    print(f'\tOriginal shape: {img.shape}')
    print(f'\tResampled shape: {img_iso.shape}')

    return img_iso


def resample(img, **kwargs):
    default_kwargs = {
        'spacing': (.5, .5, 1.0),
        'order': 0,
        'mode': 'constant',
        'cval': 0,
        'anti_aliasing': False,
        'preserve_range': True,
    }
    default_kwargs.update(kwargs)

    spacing = default_kwargs['spacing']

    prev_shape = img.shape
    img = rescale(
        img, spacing,
        order=default_kwargs['order'],
        mode=default_kwargs['mode'],
        cval=default_kwargs['cval'],
        anti_aliasing=default_kwargs['anti_aliasing'],
        preserve_range=default_kwargs['preserve_range']
    )

    print(f'{c.OKBLUE}Resampling image with spacing:{c.ENDC} {spacing}')
    print(f'\tOriginal shape: {prev_shape}')
    print(f'\tResampled shape: {img.shape}')

    return img


def load_data(img_path, metadata, type='cells'):
    img = imaging.read_image(img_path, axes='XYZ')

    print(metadata)

    if type == 'tissue':
        img = resample(img)
        metadata['x_res'] *= 2
        metadata['y_res'] *= 2

        img = isotropy(img, metadata=metadata)

        imaging.save_tiff_imagej_compatible(
            img_path.replace('.tif', '_resampled.tif'), img,
            metadata=metadata, axes='XYZ'
        )

    return img, metadata


def intersect_cells(img, mask):
    filtered = np.zeros_like(img)
    mask = (mask > 0).astype(np.uint8)

    cell_ids = np.unique(img[mask > 0])

    for z in range(img.shape[-1]):
        mask = np.isin(img[..., z], cell_ids)
        filtered[..., z] = np.where(mask, img[..., z], 0)

    return filtered


def volume(mask):
    vol_dict = {}
    cell_ids = np.unique(mask)

    for cell in cell_ids:
        if cell == 0:
            continue
        vol_dict[cell] = (mask == cell).sum()
    return vol_dict

def vox2micron(metadata):
    x_res = float(metadata.get('x_res', 1))
    y_res = float(metadata.get('y_res', 1))
    z_res = float(metadata.get('z_res', 1))
    return x_res * y_res * z_res


def calculate_summary(img, metadata, name):
    df = pd.read_csv(out_path)

    cell_ids = np.unique(img)
    cell_ids = cell_ids[cell_ids > 0]

    volumes = volume(img)
    volumes_micron = {cell: vol * vox2micron(metadata) for cell, vol in volumes.items()}

    summary = {
        'embryo_id': name,
        'n_cells': len(cell_ids),
        'cell_ids': cell_ids.tolist(),
        'cell_volumes': volumes_micron,
    }

    print(f'{c.OKBLUE}Summary updated{c.ENDC}:')
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(out_path, index=False)


def process(cells_path, tissue_path, out_img_path, raw_path):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metadata, _ = imaging.load_metadata(raw_path)

    print(f'{c.OKBLUE}Loading images{c.ENDC}:')
    cells_img, cells_metadata = load_data(cells_path, metadata, type='cells')
    tissue_img, tissue_metadata = load_data(tissue_path, metadata, type='tissue')

    if cells_img.shape != tissue_img.shape:
        print(f'{c.FAIL}Error: Images have different shapes{c.ENDC}')
        print(f'Cells shape: {cells_img.shape}')
        print(f'Tissue shape: {tissue_img.shape}')
        return

    print(f'{c.OKBLUE}Intersecting cells{c.ENDC}:')
    intersected = intersect_cells(cells_img, tissue_img)
    print(f'Cells: {len(np.unique(intersected))} / {len(np.unique(cells_img))}')
    imaging.save_tiff_imagej_compatible(out_img_path, intersected, metadata=cells_metadata, axes='XYZ')
    print(f'{c.OKGREEN}Saving intersected cells{c.ENDC}: {out_img_path}')
    print(f'{c.OKBLUE}Calculating summary{c.ENDC}:')
    calculate_summary(
        intersected, cells_metadata,
        os.path.basename(cells_path).split('.')[0]
    )

def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_path):
        df = pd.DataFrame(columns=['embryo_id', 'n_cells', 'cell_ids', 'cell_volumes'])
        df.to_csv(out_path, index=False)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--one_image', action='store_true',
        help='Process only one image.'
    )
    parser.add_argument('cells_path', nargs='?', default=None)
    parser.add_argument('tissue_path', nargs='?', default=None)
    parser.add_argument('out_img_path', nargs='?', default=None)
    parser.add_argument('raw_path', nargs='?', default=None)
    args = parser.parse_args()

    if args.one_image:
        if not args.cells_path or not args.tissue_path or not args.out_img_path or not args.raw_path:
            print('Please provide paths for cells and tissue images.')
            return

        process(args.cells_path, args.tissue_path, args.out_img_path, args.raw_path)
        sys.exit(0)

    for img in os.listdir(cells_dir):
        try:
            if not img.endswith('.tif') or 'preprocessed' in img:
                continue

            img = img.replace(' ', '')
            img_name = img.split('.')[0]
            cells_path = os.path.join(cells_dir, img)
            raw_path = os.path.join(raw_dir, img_name.replace('_mask', '.tif'))
            #if _type == 'FPN enh7-3':
            #    tissue_path = os.path.join(tissue_dir, img_name.replace('_dapi_mask', '_labels.tif'))
            #else:
            tissue_path = os.path.join(tissue_dir, img_name.replace('_dapi_mask', '_labels.tif'))
            out_img_path = os.path.join(out_dir, img_name + '_intersected_cells.tif')

            if _skip_existing and os.path.exists(out_img_path):
                print(f'{c.OKGREEN}Skipping existing image{c.ENDC}: {img_name}')
                continue

            print(f'{c.OKBLUE}Checking files{c.ENDC}:')
            print(f'Cells path: {cells_path}')
            print(f'Tissue path: {tissue_path}')
            print(f'Output path: {out_img_path}')
            print(f'Raw path: {raw_path}')
            print(f'Name: {img_name}')

            print(f'{c.OKBLUE}Processing{c.ENDC}: {img_name}')
            cmd = [
                sys.executable,     # Python interpreter
                __file__,           # Current script
                '--one_image',
                cells_path,
                tissue_path,
                out_img_path,
                raw_path
            ]
            res = subprocess.run(cmd)
            print(f'{c.OKGREEN}Finished{c.ENDC}: {img_name}')
            if res.returncode != 0:
                print(f'Result code: {res.returncode}')
                print(f'{c.FAIL}Error processing image{c.ENDC}: {img_name}')
                import traceback
                traceback.print_exc()
                continue
        except Exception as e:
            print(f'{c.FAIL}Error processing image{c.ENDC}')
            print(e)
            continue



if __name__ == '__main__':
    main()