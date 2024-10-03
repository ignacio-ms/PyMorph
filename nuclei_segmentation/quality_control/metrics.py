import sys
import os

import numpy as np
import pandas as pd
import networkx as nx

from timagetk.components.labelled_image import LabelledImage
from timagetk.components.spatial_image import SpatialImage
from ctrl.algorithm.image_overlap import cell_overlap

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary.data import imaging
from auxiliary import values as v


def dice_coef(pred, gt, thr_overlap=.3):
    BCK_LABEL = 1
    pred = pred + 1
    gt = gt + 1

    target_img = LabelledImage(
        pred, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )

    reference_img = LabelledImage(
        gt, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )

    target_cells = [lab for lab in target_img.labels() if lab != BCK_LABEL]
    reference_cells = [lab for lab in reference_img.labels() if lab != BCK_LABEL]

    overlap = cell_overlap(
        target_img, reference_img,
        mother_label=target_cells, daughter_label=reference_cells,
        method='target_daughter', ds=1, verbose=0
    )

    df_overlap = pd.DataFrame(list(overlap.keys()), columns=['target', 'reference'])
    df_overlap['overlap'] = list(overlap.values())


    # Remove associations with an overlap lower than the threshold
    df_overlap = df_overlap.loc[df_overlap['overlap'] > thr_overlap]

    # Indentify for each cell, the rarget cell that maximizes their jaccard index
    df_overlap = df_overlap.loc[df_overlap.groupby('reference')['overlap'].idxmax()]

    # Ass the missing reference cells (no intersection with any target cells) and
    # calculate the volumes
    missing_cells = set(reference_cells) - set(df_overlap.reference.values)

    rows = []
    if len(missing_cells) > 0:
        for lab in missing_cells:
            rows.append({'target': 0, 'reference': lab, 'overlap': 0})

        rows = pd.DataFrame(rows)
        df_overlap = pd.concat([df_overlap, rows], ignore_index=True)

    print(df_overlap)

    dices = []
    df_dices = df_overlap
    df_dices.columns = ['target', 'reference', 'dice']
    for i, row in df_overlap.iterrows():
        if row.reference == 0 or row.target == 0:
            dices.append(0)
            continue

        gt_mask = gt == row.reference
        pred_mask = pred == row.target

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        # intersection = row.overlap
        total = np.sum(gt_mask) + np.sum(pred_mask)
        d = (2 * intersection / total)
        dices.append(d)

        df_dices.loc[i, 'dice'] = d

    dice = np.mean(dices)

    return dice, df_dices


def volume_jaccard_index(pred, gt):
    BCK_LABEL = 1
    pred = pred + 1
    gt = gt + 1

    target_img = LabelledImage(
        pred, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )
    reference_img = LabelledImage(
        gt, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )

    target_cells = [lab for lab in target_img.labels() if lab != BCK_LABEL]
    reference_cells = [lab for lab in reference_img.labels() if lab != BCK_LABEL]

    # Compute jaccard index between all possible pairs of ocerlapping cells in the image
    jaccard = cell_overlap(
        target_img, reference_img,
        mother_label=target_cells, daughter_label=reference_cells,
        method='jaccard', ds=1, verbose=0
    )

    df_jaccard = pd.DataFrame(list(jaccard.keys()), columns=['target', 'reference'])
    df_jaccard['jaccard'] = list(jaccard.values())

    # Indentify for each cell, the rarget cell that maximizes their jaccard index
    df_jaccard = pd.DataFrame(df_jaccard, columns=['target', 'reference', 'jaccard'])
    df_jaccard = df_jaccard.loc[df_jaccard.groupby('reference')['jaccard'].idxmax()]

    # Ass the missing reference cells (no intersection with any target cells) and
    # calculate the volumes
    missing_cells = set(reference_cells) - set(df_jaccard.reference.values)

    rows = []
    if len(missing_cells) > 0:
        for lab in missing_cells:
            rows.append({'target': 0, 'reference': lab, 'jaccard': 0})

        rows = pd.DataFrame(rows)
        df_jaccard = pd.concat([df_jaccard, rows], ignore_index=True)

    # cell_ref_volume = np.sum(np.ones_like(reference_img), reference_img, reference_img.labels())

    cell_ref_volume = np.array([np.sum(gt == lab) for lab in reference_img.labels()])
    cell_ref_volume = {lab: vol for lab, vol in zip(reference_img.labels(), cell_ref_volume)}

    df_jaccard['volume'] = df_jaccard.apply(lambda x: cell_ref_volume[x.reference], axis=1)

    # Calculate the weighted jaccard index by multiplying the jaccard index by the cell volume
    df_jaccard['weighted_jaccard'] = df_jaccard.jaccard * df_jaccard.volume

    # The volume averaged jaccard index is obtained by summing all the weighted jaccard index and
    # divide them by the total volume of the gt tissue
    total_cell_volume = sum(df_jaccard.volume.values)
    sum_weighted_ji = sum(df_jaccard.weighted_jaccard.values)
    vji = sum_weighted_ji / total_cell_volume

    return vji


def segmentation_stats(pred, gt, thr_bck=.60, thr_overlap=.5, ignore_bck=True, verbose=False):
    BCK_LABEL = 1
    pred = pred + 1
    gt = gt + 1

    target_img = LabelledImage(
        pred, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )
    reference_img = LabelledImage(
        gt, not_a_label=0, axes_order='XYZ',
        origin=[0, 0, 0], voxelsize=[1.0, 1.0, 1.0],
        unit=1e-06
    )

    target = cell_overlap(
        target_img, reference_img,
        method='target_mother', ds=1, verbose=0
    )
    df_target = pd.DataFrame(list(target.keys()), columns=['target', 'reference'])
    df_target['target_in_reference'] = list(target.values())

    reference = cell_overlap(
        target_img, reference_img,
        method='target_daughter', ds=1, verbose=0
    )
    df_reference = pd.DataFrame(list(reference.keys()), columns=['target', 'reference'])
    df_reference['reference_in_target'] = list(reference.values())

    # Remove associations with an overlap lower than the threshold
    df_target = df_target[df_target['target_in_reference'] > thr_overlap]
    df_reference = df_reference[df_reference['reference_in_target'] > thr_overlap]

    # Background assiciations
    target_bacground = df_target.target.loc[
        (df_target.reference == BCK_LABEL) & (df_target.target_in_reference > thr_bck)
        ].values.tolist()

    reference_bacground = df_reference.reference.loc[
        (df_reference.target == BCK_LABEL) & (df_reference.reference_in_target > thr_bck)
        ].values.tolist()

    target_bacground.append(BCK_LABEL)
    reference_bacground.append(BCK_LABEL)

    # Remove cells associated with the background
    df_target = df_target.loc[
        ~((df_target.target.isin(target_bacground)) | (df_target.reference == BCK_LABEL))
    ].copy()

    df_reference = df_reference.loc[
        ~((df_reference.reference.isin(reference_bacground)) | (df_reference.target == BCK_LABEL))
    ].copy()

    # For the remaining reference cells, we find target cells that maximize its inclusion rate
    df_target = df_target.loc[df_target.groupby('target')['target_in_reference'].idxmax()]
    df_reference = df_reference.loc[df_reference.groupby('reference')['reference_in_target'].idxmax()]

    target_in_reference = df_target[['target', 'reference']].set_index('target').to_dict()['reference']
    reference_in_target = df_reference[['reference', 'target']].set_index('reference').to_dict()['target']

    # Then a bipartitate graph is built where:
    # - The left nodes represents the target cells
    # - The right nodes represents the reference cells
    # - The edges represents one-to-one associations between target and reference cells using the previous maximization procedure.

    # Reindex labels
    target_labels = list(set(df_target.target.values) | set(df_reference.target.values))
    reference_labels = list(set(df_target.reference.values) | set(df_reference.reference.values))

    label_tp_list = [(m, 'l') for m in target_labels] + [(d, 'r') for d in reference_labels]
    lg2nid = dict(zip(label_tp_list, range(len(label_tp_list))))

    G = nx.Graph()
    G.add_nodes_from([(nid, {'label': lab, 'group': g}) for (lab, g), nid in lg2nid.items()])

    target_to_ref_list = [
        (lg2nid[(i, 'l')], lg2nid[(j, 'r')])
        for i, j in target_in_reference.items()
    ]
    G.add_edges_from(target_to_ref_list)

    ref_to_target_list = [
        (lg2nid[(i, 'r')], lg2nid[(j, 'l')])
        for i, j in reference_in_target.items()
    ]
    G.add_edges_from(ref_to_target_list)

    # We find all the connected sub-graphes of the bipartite graph.
    # The connected sub-graphes corresponds to target and reference cells that are associated

    connected_graph = [list(G.subgraph(c)) for c in nx.connected_components(G)]

    # Gather all connected subgraph con reidex according to image labels
    nid2lg = {v: k for k, v in lg2nid.items()}

    out_results = []
    for c in connected_graph:
        # c = c[0] if not isinstance(c[0], int) else c
        if len(c) > 1:
            target, reference = [], []
            for nid in c:
                if nid2lg[nid][1] == 'l':
                    target.append(nid2lg[nid][0])
                else:
                    reference.append(nid2lg[nid][0])

            out_results.append({'target': target, 'reference': reference})

    # Add background associations
    for lab in target_bacground:
        if lab != BCK_LABEL:
            out_results.append({'target': [lab], 'reference': []})

    for lab in reference_bacground:
        if lab != BCK_LABEL:
            out_results.append({'target': [], 'reference': [lab]})

    out_results = pd.DataFrame(out_results)

    def segmentation_state(row):
        if len(row.reference) == 0:
            return 'background'
        elif len(row.target) == 0:
            return 'missing'
        elif len(row.target) == 1:
            if len(row.reference) == 1:
                return 'correct'
            else:
                return 'under_segmented'
        else:
            if len(row.reference) == 1:
                return 'over_segmented'
            else:
                return 'confused'

    out_results['state'] = out_results.apply(segmentation_state, axis=1)

    cell_stats = {'correct': 0, 'under_segmented': 0, 'over_segmented': 0, 'confused': 0}
    if not ignore_bck:
        cell_stats['background'] = BCK_LABEL

    state_target = {
        lab: state for list_lab, state
        in zip(out_results.target.values, out_results.state.values)
        for lab in list_lab if state in cell_stats
    }

    for lab, state in state_target.items():
        cell_stats[state] += 1

    total_cells = len(state_target)
    cell_stats = {state: np.around(val / total_cells * 100, 2) for state, val in cell_stats.items()}

    # Add missing cells
    total_reference_cells = len(np.unique([
        item for sublist in out_results.reference.values for item in sublist]
    ))

    missing_cells = [
        item for sublist
        in out_results.reference.loc[out_results.state == 'missing'].values
        for item in sublist
    ]

    total_missing = len(missing_cells)
    cell_stats['missing'] = np.around(total_missing / total_reference_cells * 100, 2)

    if verbose:
        print(f'NO. Cells {total_cells}')
        print(f'% Correct: {cell_stats["correct"]}')
        print(f'% Under-segmented: {cell_stats["under_segmented"]}')
        print(f'% Over-segmented: {cell_stats["over_segmented"]}')
        print(f'% Missing GT cells: {cell_stats["missing"]}')
        print(f'% Confused: {cell_stats["confused"]}')

    return out_results

