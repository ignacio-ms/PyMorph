from GASP.segmentation import (
    GaspFromAffinities,
    WatershedOnDistanceTransformFromAffinities,
)
from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
import h5py
import numpy as np
import time
from skimage.measure import label, regionprops
import nibabel as nib
import os
import json

input_type = "data_float32"
output_type = "labels"
save_directory = "output"
out_ext = ".h5"
algorithm_name = "GASP"
gasp_linkage_criteria = "average"  # "mutex_watershed"
beta = 0.5
ws_threshold = 0.4
ws_minsize = 30
ws_sigma = 0.8
n_threads = 6
post_minsize = 40
voxel_size = [1.0, 1.0, 1.0]


def normalize_01(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-12)


def adjust_input_type(data, input_type):
    if input_type == "labels":
        return data.astype(np.uint16)
    elif input_type in ["data_float32", "data_uint8"]:
        data = data.astype(np.float32)
        data = normalize_01(data)
        return data


def shift_affinities(affinities, offsets):
    rolled_affs = []
    for i, _ in enumerate(offsets):
        offset = offsets[i]
        shifts = tuple([int(off / 2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf

        padded_inverted_affs = np.pad(
            affinities, pad_width=((0, 0),) + tuple(padding), mode="constant"
        )

        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1])
            for ax in range(3)
        )

        padded_inverted_affs = np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[
            crop_slices
        ]
        rolled_affs.append(padded_inverted_affs)
        del padded_inverted_affs

    rolled_affs = np.stack(rolled_affs)
    return rolled_affs


def read_file(file_path):
    with h5py.File(file_path, "r") as f:
        ds = f["predictions"]
        data = ds[...]
    return data


def main(file_path):
    pmaps = read_file(file_path)
    data = np.nan_to_num(pmaps)
    data = adjust_input_type(data, input_type)
    if data.ndim == 4:
        data = data[0]
    print(data.dtype)
    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    affinities = np.stack([data, data, data], axis=0)
    affinities = shift_affinities(affinities, offsets=offsets)
    affinities = 1 - affinities
    superpixel_gen = WatershedOnDistanceTransformFromAffinities(
        offsets,
        threshold=ws_threshold,
        min_segment_size=ws_minsize,
        preserve_membrane=True,
        sigma_seeds=ws_sigma,
        stacked_2d=False,
        used_offsets=[0, 1, 2],
        offset_weights=[1, 1, 1],
        n_threads=n_threads,
    )
    runtime = time.time()

    run_GASP_kwargs = {
        "linkage_criteria": gasp_linkage_criteria,
        "add_cannot_link_constraints": False,
        "use_efficient_implementations": False,
    }

    # Init and run Gasp
    gasp_instance = GaspFromAffinities(
        offsets,
        superpixel_generator=superpixel_gen,
        run_GASP_kwargs=run_GASP_kwargs,
        n_threads=n_threads,
        beta_bias=beta,
    )
    # running gasp
    segmentation, _ = gasp_instance(affinities)

    # init and run size threshold
    size_threshold = SizeThreshAndGrowWithWS(post_minsize, offsets)
    segmentation = size_threshold(affinities, segmentation)

    # stop real world clock timer
    runtime = time.time() - runtime
    print(f"Clustering took {runtime:.2f} s")
    print(segmentation.shape)
    return segmentation


def post_filtering_size(segmentation):
    blobs_labels = label(segmentation, background=0)
    props = regionprops(blobs_labels)
    areas = [prop.area_bbox for prop in props]
    labels = [prop.label for prop in props]
    ## CONDICION PARA FILTRAR
    labels_back = [labels[i] for i, a in enumerate(areas) if a > 10 * np.mean(areas)]
    for l in labels_back:
        blobs_labels[np.where(blobs_labels[:, :, :] == l)] = 0
    return blobs_labels


if __name__ == "__main__":
    f = open("/homedtic/dvarela/specimens.json")
    data = json.load(f)
    flatten_list = [
        element
        for sublist in [data[i] for i in ["stage1", "stage2", "stage3", "stage4"]]
        for element in sublist
    ]
    mems = "/homedtic/dvarela/RESULTS/membranes/PNAS"
    gasp = "/homedtic/dvarela/RESULTS/membranes/GASP_PNAS"
    for sp in flatten_list:
        file_h5 = os.path.join(
            mems, f"2019{sp}_mGFP_CardiacRegion_0.5_ZXY_predictions.h5"
        )
        print(f"RUNNING {file_h5}")
        outfile = os.path.join(
            gasp,
            os.path.basename(file_h5)
            .replace(".h5", "_GASP.nii.gz")
            .replace("ZXY", "XYZ"),
        )
        print(outfile)
        seg = main(file_h5)
        seg_post = post_filtering_size(seg)
        seg_postXYZ = np.swapaxes(np.swapaxes(seg_post, 0, 2), 1, 0)
        ni_img = nib.Nifti1Image(
            seg_postXYZ.astype("uint16"),
            affine=np.eye(4),
        )
        nib.save(
            ni_img,
            outfile,
        )
        print("---------------")
