import os
import sys

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/"

import numpy as np
from rich.progress import Progress
from skimage.measure import regionprops

from stardist.models import StarDist3D, StarDist2D
from csbdeep.utils import normalize

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c
from nuclei_segmentation.my_cellpose import load_img, load_model
from nuclei_segmentation.processing.preprocessing import Preprocessing
from util.data import imaging
from util.gpu.gpu_tf import increase_gpu_memory, set_gpu_allocator


def load_stardist_model(do_3D):
    try:
        increase_gpu_memory()
        set_gpu_allocator()
    except Exception as e:
        print(f'{c.WARNING}GPU mem. growth not available{c.ENDC}')

    if do_3D:
        _model_name = 'n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)'
        _model_path = f'{current_dir}/../models/stardist_models/'
        return StarDist3D(
            None, name=_model_name,
            basedir=_model_path
        )

    _model_name = '2D_versatile_fluo'
    return StarDist2D.from_pretrained(_model_name)


def snitch_labels(mask, threshold=0.3):
    """
    Stitches a stack of 2D label images (in XYZ order) into a 3D label volume using skimage.measure.regionprops.

    Parameters:
      mask : numpy array
          A 3D numpy array of shape (X, Y, Z) where each slice (mask[:, :, z]) is a 2D labeled image.
      threshold : float, optional
          Minimum fraction of overlap (relative to the region's area) required to propagate the label from the previous slice.
          If the overlap is less than this threshold, a new label is assigned.

    Returns:
      stitched : numpy array
          A 3D numpy array of stitched labels with shape (X, Y, Z).
    """
    X, Y, Z = mask.shape
    print(f'{c.OKGREEN}Mask shape{c.ENDC}: {mask.shape}')
    stitched = np.zeros_like(mask, dtype=np.int32)
    current_max_label = 0

    # Process the first slice (z = 0)
    slice0 = mask[:, :, 0]
    unique_labels = np.unique(slice0)
    unique_labels = unique_labels[unique_labels != 0]
    for lab in unique_labels:
        current_max_label += 1
        stitched[:, :, 0][slice0 == lab] = current_max_label

    # Process subsequent slices (z = 1 to Z-1)
    with Progress() as progress:
        task = progress.add_task("[blue]Stitching slices...", total=Z - 1)
        for z in range(1, Z):
            prev_slice = stitched[:, :, z - 1]
            curr_slice = mask[:, :, z]
            stitched_slice = np.zeros_like(curr_slice, dtype=np.int32)

            # Use regionprops to get properties of connected regions in the current slice.
            props = regionprops(curr_slice)
            for prop in props:
                # prop.label corresponds to the label in curr_slice.
                if prop.label == 0:
                    continue
                coords = prop.coords  # an (N, 2) array of pixel coordinates (row, col)
                area = prop.area

                # Get the overlapping labels from the previous stitched slice.
                prev_labels = prev_slice[coords[:, 0], coords[:, 1]]
                # Count the overlaps using np.bincount.
                counts = np.bincount(prev_labels)
                # Ensure background (label 0) is not considered.
                if len(counts) > 0:
                    counts[0] = 0

                if counts.sum() == 0:
                    # No overlap with any object in the previous slice: assign a new label.
                    current_max_label += 1
                    new_lab = current_max_label
                else:
                    best_lab = np.argmax(counts)
                    best_count = counts[best_lab]
                    # If the best overlap represents a sufficient fraction of the region area, use that label.
                    if best_count / area >= threshold:
                        new_lab = best_lab
                    else:
                        current_max_label += 1
                        new_lab = current_max_label

                # Assign the new label to the current region.
                stitched_slice[coords[:, 0], coords[:, 1]] = new_lab

            stitched[:, :, z] = stitched_slice
            progress.update(task, advance=1)

    return stitched


def run(
        img_path, out_path,
        model_type='cellpose', channels=[0, 0],
        diameter=17, pipeline=None,
        do_3D=False, anisotropy=None,
        cellprob_threshold=.4, stitch_threshold=.4,
        flow_threshold=.4,
        normalize=False,
        verbose=0,
):
    _type = 'WT'
    data_path = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/{_type} new/Dapi/'
    # data_path = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/ADULT {_type}/Dapi/'
    # data_path = f'/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/cova/'

    # ----------------------
    # Loads & Pre-processing
    # ----------------------
    img = load_img(
        data_path + img_path, pipeline,
        axes='ZYX' if model_type == 'cellpose' else 'XYZ',
        verbose=verbose
    )

    if verbose:
        print(f'{c.OKGREEN}Loaded image{c.ENDC} - {img.shape}')
        print(f'{c.OKGREEN}Loading model:{c.ENDC} {model_type}')
    model = load_model(model_type='nuclei') if model_type == 'cellpose' else load_stardist_model(do_3D)

    # ----------------------
    # Segmentation
    # ----------------------
    if model_type == 'cellpose':
        if verbose:
            print(f'{c.OKGREEN}Running Cellpose segmentation: {c.ENDC}')
            print(f'\t{c.BOLD}Diameter{c.ENDC}: {diameter}')
            print(f'\t{c.BOLD}Channels{c.ENDC}: {channels}')
            print(f'\t{c.BOLD}Anisotropy{c.ENDC}: {anisotropy}')
            print(f'\t{c.BOLD}Do 3D{c.ENDC}: {do_3D}')
            print(f'\t{c.BOLD}Stitch threshold{c.ENDC}: {stitch_threshold}')
            print(f'\t{c.BOLD}Cellprob threshold{c.ENDC}: {cellprob_threshold}')
            print(f'\t{c.BOLD}Flow threshold{c.ENDC}: {flow_threshold}')
            print(f'\t{c.BOLD}Normalize{c.ENDC}: {normalize}')

            mask, _, _, _ = model.eval(
                img,
                diameter=diameter,
                channels=channels,
                normalize=normalize,
                anisotropy=anisotropy,
                do_3D=do_3D,
                cellprob_threshold=cellprob_threshold,
                stitch_threshold=stitch_threshold,
                flow_threshold=flow_threshold,
            )

    elif model_type == 'stardist':
        print(f'{c.OKGREEN}Running StarDist segmentation: {c.ENDC}')

        # pass from (1, x, y, z, 1) to (x, y, z)
        if img.ndim == 5:
            img = img[0, ..., 0]
        elif img.ndim == 4 and img.shape[0] == 1:
            img = img[0, ..., 0]
        elif img.ndim == 4 and img.shape[-1] == 1:
            img = img[..., 0]

        if normalize:
            img = normalize(img, 1, 99.8, axis=(0, 1, 2))

        if do_3D:
            mask, _ = model.predict_instances(
                img,
                n_tiles=(8, 8, 1),
                show_tile_progress=True,
                axes='XYZ',
            )
            mask = np.swapaxes(mask, 0, 2)

        else:
            mask_slices = []

            with Progress() as progress:
                task = progress.add_task("[blue]Segmenting slices...", total=img.shape[-1])
                for z in range(img.shape[-1]):
                    mask_slice, _ = model.predict_instances(
                        img[..., z],
                        n_tiles=(8, 8),
                        show_tile_progress=False,
                        axes='XY',
                    )
                    mask_slices.append(mask_slice)
                    progress.update(task, advance=1)

            if 'equalization' in pipeline:
                from skimage import morphology
                print(f'{c.OKGREEN}Applying erosion to mask slices{c.ENDC}')
                mask_slices = np.array(mask_slices)
                prev_dtype = mask_slices.dtype
                # for z in range(mask_slices.shape[0]):
                #     mask_slices[z] = morphology.erosion(mask_slices[z], morphology.disk(2))
                # mask_slices = mask_slices.astype(prev_dtype)

            # Reslice mask into XYZ order
            mask_slices = np.swapaxes(mask_slices, 0, 2)
            mask = snitch_labels(mask_slices, threshold=stitch_threshold)
            # mask = mask_slices

    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # ----------------------
    # Reconstruct image if rescaled
    # ----------------------
    # if 'resample' in pipeline:
    #     mask_resampled = Preprocessing().resample(
    #         mask,
    #         spacing=(1.0, 2.5, 2.5) if model_type == 'cellpose' else (2.5, 2.5, 1.0),
    #         order=0,  # Nearest-neighbor: preserves discrete labels.
    #         mode='edge',
    #         clip=True,
    #         anti_aliasing=False,  # No anti_aliasing for discrete labels.
    #         preserve_range=True,
    #         verbose=verbose
    #     ).astype(mask.dtype)
    #     mask = mask_resampled

    # ----------------------
    # Save segmentation
    # ----------------------
    imaging.save_nii(
        mask, data_path + 'test/' + out_path,
        axes='ZYX' if model_type == 'cellpose' else 'XYZ', # 'YXZ',
        verbose=verbose
    )


if __name__ == '__main__':
    # img_name = 'EXW_WT1_2_Brdu_001-Dapi.tif'
    # out_name = 'EXW_WT1_2_Brdu_001-Dapi_mask.nii.gz'

    # img_name = 'EXW348b_BrdU-Dapi.tif'
    # out_name = 'EXW348b_BrdU-Dapi_maks.nii.gz'

    # img_name = 'e3.5_raw_1.tif'
    # out_name = 'e3.5_mask_1.nii.gz'

    for img_name in [
        # 'EXW_1_BrdU001-Dapi.tif',
        # 'EXW_1_BrdU002-Dapi.tif',
        # 'EXW_1_BrdU003-Dapi.tif',
        # 'EXW_1_BrdU004-Dapi.tif',
        # 'EXW_1_BrdU006-Dapi.tif',
        # 'EXW_1_BrdU007-Dapi.tif',
        # 'EXW_1_BrdU-Dapi.tif',
        # 'EXW_2_BrdU001-Dapi.tif',
        # 'EXW_2_BrdU002-Dapi.tif',
        # 'EXW_2_BrdU003-Dapi.tif',
        # 'EXW_2_BrdU004-Dapi.tif',
        # 'EXW_2_BrdU-Dapi.tif',
        # 'EXW321_1_Brdu_001-Dapi.tif',
        # 'EXW321_1_Brdu_002-Dapi.tif',
        # 'EXW321_1_Brdu_-Dapi.tif',
        # 'EXW_321b_Brdu_001-Dapi.tif',
        # 'EXW_321b_Brdu_-Dapi.tif',
        # 'EXW_324_Brdu__001-Dapi.tif',
        # 'EXW_324_Brdu__002-Dapi.tif',
        # 'EXW_324_Brdu__003-Dapi.tif',
        # 'EXW_324_Brdu__004-Dapi.tif',
        # 'EXW_324_Brdu__-Dapi.tif',
        # 'EXW_324_Brdu___-Dapi.tif',
        # 'EXW345_1_Brdu_002-Dapi.tif',
        # 'EXW345_1_Brdu_003-Dapi.tif',
        # 'EXW345_1_Brdu_004-Dapi.tif',
        # 'EXW352_1_Brdu_-Dapi.tif',
        # 'EXW_352_Brdu__001-Dapi.tif',
        # 'EXW_352_Brdu__002-Dapi.tif',
        # 'EXW_352_Brdu__-Dapi.tif',
        # 'EXW370a(MYC)_BrdU001-Dapi.tif',
        # 'EXW370a(MYC)_BrdU-Dapi.tif',
        # 'EXW370b(MYC)_BrdU002-Dapi.tif',
        # 'EXW370b(MYC)_BrdU003-Dapi.tif',
        # 'EXW_3a_BrdU001-Dapi.tif',
        # 'EXW_3a_BrdU-Dapi.tif',
        # 'EXW_3b_BrdU-Dapi.tif',
        # 'EXW_3c_BrdU001-Dapi.tif',
        # 'EXW_3c_BrdU002-Dapi.tif',
        # 'EXW_3c_BrdU003-Dapi.tif',
        # 'EXW45_1_Brdu_001-Dapi.tif',
        # 'EXW45_1_Brdu_-Dapi.tif',

        # 'EXW348a_BrdU001-Dapi.tif',
        # 'EXW348a_BrdU-Dapi.tif',
        # 'EXW348b_BrdU001-Dapi.tif',
        # 'EXW348b_BrdU-Dapi.tif',
        # 'EXW348c_BrdU-Dapi.tif',
        # 'EXW367a_BrdU-Dapi.tif',
        # 'EXW367c_BrdU-Dapi.tif',
        # 'EXW368a_BrdU001-Dapi.tif',
        # 'EXW368a_BrdU-Dapi.tif',
        # 'EXW368b_BrdU001-Dapi.tif',
        # 'EXW382_1_Brdu_001-Dapi.tif',
        # 'EXW382_1_Brdu_002-Dapi.tif',
        # 'EXW382_1_Brdu_003-Dapi.tif',
        # 'EXW382_1_Brdu_-Dapi.tif',
        # 'EXW386_1_Brdu_001-Dapi.tif',
        # 'EXW386_1_Brdu_-Dapi.tif',
        # 'EXW386_2_Brdu_-Dapi.tif',
        # 'EXW387_1_Brdu_001-Dapi.tif',
        # 'EXW387_1_Brdu_002-Dapi.tif',
        # 'EXW387_1_Brdu_-Dapi.tif',
        # 'EXW_WT1_1_Brdu_-Dapi.tif',
        # 'EXW_WT1_2_Brdu_001-Dapi.tif',
        # 'EXW_WT1_2_Brdu_002-Dapi.tif',
        # 'EXW_WT1_2_Brdu_003-Dapi.tif',
        # 'EXW_WT1_2_Brdu_-Dapi.tif',
        # 'EXW_WT2_Brdu_001-Dapi.tif',
        # 'EXW_WT2_Brdu_002-Dapi.tif',
        # 'EXW_WT2_Brdu_-Dapi.tif',

        'AEI_ki67_001_dapi.tif', # WT
        'AEI_ki67_003_dapi.tif',
        'AEI_ki67__dapi.tif',
        'AEI_ki67_002_dapi.tif',
        'AEI_ki67_004_dapi.tif',

        # 'efj806__dapi.tif', # MYC
        # 'efj806_ki67_004_dapi.tif',
        # 'efj806_ki67__dapi.tif',
        # 'efj806_ki67_001_dapi.tif',
        # 'efj806_ki67_005_dapi.tif',
        # 'efj806_ki67_002_dapi.tif',
        # 'efj806_ki67_006_dapi.tif',
        # 'efj806_ki67_003_dapi.tif',
        # 'efj806_ki67_007_dapi.tif',
    ]:
        out_name = img_name.replace('.tif', '_mask.nii.gz')

        model_type = 'stardist'
        diameter = None
        do_3D = False
        anisotropy = None
        cellprob_threshold = .0
        flow_threshold = .5
        stitch_threshold = .1  # 2D only

        normalize = False
        pipeline = [
            'resample',
            'remove_bck',
            'intensity_calibration',
            # 'isotropy',
            'rescale_intensity',
            # 'norm_adaptative',
            'equalization',
            'bilateral',
        ]

        try:
            run(
                img_name, out_name,
                model_type=model_type,
                diameter=diameter,
                do_3D=do_3D,
                anisotropy=anisotropy,
                cellprob_threshold=cellprob_threshold,
                stitch_threshold=stitch_threshold,
                flow_threshold=flow_threshold,
                normalize=normalize,
                pipeline=pipeline,
                verbose=1
            )
        except Exception as e:
            print(f'{c.FAIL}Error processing image{c.ENDC}: {img_name}')
            import traceback
            traceback.print_exc()
            continue
