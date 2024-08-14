import sys
import os

import pandas as pd
from scipy import stats

import tensorflow as tf
import json
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from cell_division.nets.transfer_learning import CNN
from auxiliary.data.dataset_cell import CellDataset
from auxiliary import values as v
from auxiliary.utils.colors import bcolors as c
from auxiliary.utils import visualizer as vis
from auxiliary.data import imaging

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import GlobalAveragePooling2D
from cell_division.nets.custom_layers import (
    w_cel_loss,
    focal_loss,
    ExtendedLSEPooling,
    extended_w_cel_loss,
    LSEPooling
)

from cell_division.nets.cam import GradCAM, overlay_heatmap, CAM, GradCAMpp

# GPU config
from auxiliary.utils.timer import LoadingBar
from auxiliary.gpu.gpu_tf import (
    increase_gpu_memory,
    set_gpu_allocator,
    clear_session
)

increase_gpu_memory()
set_gpu_allocator()


def pre_train(model, train, val, batch_size=32, verbose=0):
    """
    Train the model with the labeled data.
    :param model: Model to train
    :param train: Training data generator
    :param val: Validation data generator
    :param batch_size: Batch size
    :param verbose: Verbosity level
    :return:
    """
    train_generator = train
    val_generator = val

    model.fit(
        train_generator,
        val_generator,
        epochs=100,
        verbose=verbose,
        batch_size=batch_size
    )

    return model


def pseudo_labeling(model, unlabeled, threshold=.9, undersample=True, verbose=0):
    """
    Predict the labels of the unlabeled data with the pretrained model.
    Those instances with a probability higher than the threshold will be pseudo-labeled
    and added to the labeled data.

    To avoid class imbalance, the pseudo-labeled data will be undersampled if specified.

    The data input is composed by 3D cell images. To make a prediction, the model will make
    a prediction for each z-stack image and average the results by a majority vote.
    :param model: Pretrained model
    :param unlabeled: Unlabeled data generator
    :param threshold: Threshold to consider a prediction as valid
    :param undersample: Whether to undersample the pseudo-labeled data (default: True)
    :param verbose: Verbosity level (default: 0)
    :return:
    """
    unlabeled_generator = unlabeled

    predictions = []
    predictions_prob = []
    two_d_predictions = []
    for img in unlabeled_generator:
        img = img[0] * 255
        img = img.astype(np.uint8)
        if img.ndim == 3:
            for z in range(img.shape[2]):
                aux = cv2.cvtColor(img[..., z], cv2.COLOR_GRAY2RGB)
                # Creating axis for the batch
                aux = np.expand_dims(aux, axis=0)
                two_d_predictions.append(model.model.predict(aux))
        else:
            aux = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # Creating axis for the batch
            aux = np.expand_dims(aux, axis=0)
            two_d_predictions.append(model.model.predict(aux))

        two_d_predictions = np.array(two_d_predictions)

        # Voting by majority
        predicted_classes = np.argmax(two_d_predictions, axis=-1).flatten()
        predicted_classes_prob = np.max(two_d_predictions, axis=-1).flatten()
        majority_vote = stats.mode(predicted_classes, keepdims=True)[0][0]

        predictions.append(majority_vote)
        predictions_prob.append(predicted_classes_prob[0])
        two_d_predictions = []

    pseudo_labels = np.array(predictions)
    pseudo_labels_prob = np.array(predictions_prob)

    mask = pseudo_labels_prob > threshold
    pseudo_labels = pseudo_labels[mask]

    if verbose:
        print(f'{c.OKBLUE}Pseudo-labeling results{c.ENDC}')
        print(f'Pseudo-labels: {len(pseudo_labels)}')
        print(f'Pseudo-labels distribution: {np.bincount(pseudo_labels)}')

    # Saving the image names and pseudo labels into csv
    pseudo_labels_df = pd.DataFrame({
        'id': unlabeled.img_short_names[mask].astype(str),
        'label': pseudo_labels
    })

    # Get instances pseudo-labeled index to remove from the unlabeled dataset
    idxs = np.where(mask == True)[0]
    unlabeled.remove_images(idxs)

    if undersample:
        X = np.array(pseudo_labels_df['id']).reshape(-1, 1)
        y = np.array(pseudo_labels_df['label'])

        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        pseudo_labels_df = pd.DataFrame({
            'id': X_res.flatten(),
            'label': y_res
        })

        if verbose:
            print(f'{c.OKBLUE}Undersampling results{c.ENDC}')
            print(f'Pseudo-labels: {len(pseudo_labels_df)}')
            print(f'Pseudo-labels distribution: {np.bincount(pseudo_labels_df["label"])}')

    pseudo_labels_df.to_csv(v.data_path + 'CellDivision/undersampled/pseudo_labels.csv', index=False)


def split_pseudo_labeled_images(
        data_path=None,
        pseudo_labels_file='pseudo_labels.csv',
        verbose=0
):
    """
    Images in the unlabeled dataset are 3D stacks. This function splits the images with pseudo-labels
    into 2D images with the corresponding pseudo-label.
    Apart from assigning the pseudo-labels to the 2D stacks, the 2D stacks are saved into a new folder.
    :param pseudo_labels_file: File with the pseudo-labels and images ids.
    :return: Dataset with the splitted images in 2D
    """
    if data_path is None:
        data_path = v.data_path

    df = pd.read_csv(data_path + f'CellDivision/undersampled/{pseudo_labels_file}')
    img_path = data_path + 'CellDivision/images_unlabeled/'
    new_path = data_path + 'CellDivision/images_unlabeled_2d/'

    new_df = pd.DataFrame(columns=['id', 'label'])

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    if verbose:
        print(f'{c.OKBLUE}Splitting pseudo-labeled images into 2D slices...{c.ENDC}')
    bar = LoadingBar(len(df))

    for idx, row in df.iterrows():
        img = imaging.read_image(img_path + row['id'])
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        for z in range(img.shape[2]):
            imaging.save_prediction(
                img[..., z],
                new_path + f'{row["id"].replace(".tif", "")}_{z}.tif',
                axes='XY'
            )

            new_row = pd.DataFrame({
                'id': [f'{row["id"].replace(".tif", "")}_{z}.tif'],
                'label': [row['label']]
            })
            new_df = pd.concat([new_df, new_row], ignore_index=True)

        if verbose:
            bar.update()

    if verbose:
        bar.end()
        print(f'{c.OKGREEN}Splitting completed{c.ENDC}')
        print(f'\t{c.BOLD}Total instances:{c.ENDC} {len(new_df)}')

    new_df.to_csv(data_path + f'CellDivision/undersampled/{pseudo_labels_file.replace(".csv", "_2d.csv")}', index=False)


def merge_datasets(train, pseudo_labels_file='pseudo_labels_2d.csv'):
    """
    Merge the labeled and pseudo-labeled data into a single dataset.
    :param pseudo_labels_file: Path to the pseudo-labels file.
    :param train: Labeled dataset generator.
    :return:
    """
    train.add_pseudo_labels(
        v.data_path + 'CellDivision/images_unlabeled/',
        v.data_path + f'CellDivision/undersampled/{pseudo_labels_file}'
    )
    return train


def print_iter_results(model, train, test, val):
    print(f'\t\t{c.OKBLUE}Model trained:{c.ENDC}')
    print(f'\t\t{c.BOLD}Evaluating...{c.ENDC}')

    train_res = model.model.evaluate(train, verbose=1)
    print(f'\t\t{c.BOLD}Train AUC:{c.ENDC} {train_res[1]}')

    val_res = model.model.evaluate(val, verbose=1)
    print(f'\t\t{c.BOLD}Validation AUC:{c.ENDC} {val_res[1]}')

    test_res = model.model.evaluate(test, verbose=1)
    print(f'\t\t{c.BOLD}Test AUC:{c.ENDC} {test_res[1]}')


def semi_supervised_learning(
        model, train, val, test, unlabeled,
        max_iter=10, batch_size=32, verbose=0
):
    """
    Semi-supervised learning algorithm.
        1. Pre-train the model with the labeled data.
        2. Predict the pseudo-labels of the unlabeled data.
        3. Merge the labeled and pseudo-labeled data.
        4. Train the model with the merged data.
        5. Iterate until convergence | max_iter.
    :param model: Model to train
    :param train: Labeled training data generator
    :param val: Validation data generator
    :param test: Test data generator
    :param unlabeled: Unlabeled data generator
    :param batch_size: Batch size (default: 32)
    :param verbose: Verbosity level (default: 0)
    :return:
    """

    results = []
    iters_without_improvement = 0
    for i in range(max_iter):
        if verbose:
            print(f'{c.OKGREEN}Iteration:{c.ENDC} {i + 1}')
            print(f'\t{c.OKBLUE}Pre-training...{c.ENDC}')

        # Pre-train the model with the labeled data
        model = pre_train(model, train, val, batch_size=batch_size, verbose=verbose)
        results.append(model.model.evaluate(val, verbose=0)[1])

        # Early stopping and saving the model
        if i == 0:
            model.model.save(f'../models/cellular_division_models/vgg16_semi_{i}.h5')

        elif results[-1] > np.max(results[:-1]):
            model.model.save('../models/cellular_division_models/vgg16_semi_{i}.h5')
            iters_without_improvement = 0

        if i > 0 and results[-1] < np.max(results[:-1]):
            iters_without_improvement += 1
            if iters_without_improvement == 3:
                print(f'{c.OKGREEN}Early stopping at iteration {i + 1}{c.ENDC}')
                break

        if verbose:
            print_iter_results(model, train, test, val)
            print(f'\t{c.OKBLUE}Pseudo-labeling...{c.ENDC}')

        pseudo_labeling(model, unlabeled, verbose=verbose)
        split_pseudo_labeled_images(verbose=verbose)
        train = merge_datasets(train)

    return model
