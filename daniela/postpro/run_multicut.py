import h5py
import numpy as np
import time
from skimage.measure import label, regionprops
import nibabel as nib
import nifty
import nifty.graph.rag as nrag
import numpy as np
from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import (
    multicut_kernighan_lin,
    transform_probabilities_to_costs,
)
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter


input_type = "data_float32"
beta = 0.5
ws_threshold = 0.5
ws_minsize = 30
ws_sigma = 0.8
ws_w_sigma = 0
post_minsize = 40


def adjust_input_type(data, input_type):
    if input_type == "labels":
        return data.astype(np.uint16)

    elif input_type in ["data_float32", "data_uint8"]:
        data = data.astype(np.float32)
        data = normalize_01(data)
        return data


def normalize_01(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-12)


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
    ## watershed
    runtime = time.time()

    ws, _ = distance_transform_watershed(
        data, ws_threshold, ws_sigma, sigma_weights=ws_w_sigma, min_size=ws_minsize
    )
    # Compute region adjacency graph of segmentation.
    rag = compute_rag(ws)
    features = nrag.accumulateEdgeMeanAndLength(
        rag, data, numberOfThreads=1
    )  # DO NOT CHANGE numberOfThreads
    probs = features[:, 0]  # mean edge prob
    edge_sizes = features[:, 1]
    # Prob -> edge costs
    # Transform probabilities to costs via negative log likelihood.
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=beta)
    # Creating graph
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())
    # Solving Multicut
    node_labels = multicut_kernighan_lin(graph, costs)
    segmentation = nifty.tools.take(node_labels, ws)
    if post_minsize > ws_minsize:
        segmentation, _ = apply_size_filter(segmentation, data, post_minsize)
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
    PROB = "/homedtic/dvarela/RESULTS/membranes/PNAS/20190404_E1_mGFP_CardiacRegion_0.5_ZYX_predictions.h5"
    outfile = "/homedtic/dvarela/RESULTS/membranes/MULTICUT_PNAS/20190404_E1_mGFP_CardiacRegion_0.5_ZYX_predictions_MULTICUT.nii.gz"

    
    seg = main(PROB)
    seg_post = post_filtering_size(seg)
    ## save as nii.qz

    ni_img = nib.Nifti1Image(
        seg_post.astype("uint16"),
        affine=np.eye(4),
    )

    nib.save(
        ni_img,
        outfile,
    )
    print(outfile)
