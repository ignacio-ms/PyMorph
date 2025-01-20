import math
import sys
import os

import pandas as pd

import numpy as np
import cv2

from imblearn.under_sampling import RandomUnderSampler

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util import values as v
from util.misc.colors import bcolors as c
from util.data import imaging

# GPU config
from util.misc.timer import LoadingBar
from util.gpu.gpu_tf import (
    increase_gpu_memory,
    set_gpu_allocator
)

increase_gpu_memory()
set_gpu_allocator()


def pre_train(model, train, val, batch_size=32, verbose=0):
    """
    Train the model with the labeled data.
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


def pseudo_labeling(model, unlabeled, threshold=0.9, undersample=True, verbose=0, iteration=0):
    """
    Predict the labels of the unlabeled data with the pretrained model.
    Implements weighted averaging of slice predictions, giving more weight to middle slices.
    Processes all slices in batches to speed up computation.
    Incorporates RandomUnderSampler to address class imbalance.

    :param model: Pretrained CNN model.
    :param unlabeled: Unlabeled data generator (UnlabeledDataset instance).
    :param threshold: Confidence threshold for selecting pseudo-labels.
    :param undersample: Whether to apply undersampling to the pseudo-labeled data.
    :param verbose: Verbosity level.
    :param iteration: Current iteration number (used for dynamic thresholding).
    :return: DataFrame containing pseudo-labels with 'id', 'label', and 'confidence'.
    """
    unlabeled_generator = unlabeled

    all_slices = []
    slice_image_indices = []
    slice_weights = []
    total_images = len(unlabeled_generator.img_names)

    if verbose:
        print(f'{c.OKBLUE}Collecting all slices from unlabeled images...{c.ENDC}')
        bar = LoadingBar(total_images)

    # Collect all slices and their associated image indices and weights
    for img_idx in range(total_images):
        img = unlabeled.__get_image(unlabeled.img_names[img_idx]) * 255
        img = img.astype(np.uint8)

        if img.ndim == 3:
            n_slices = img.shape[2]
            mid_slice = n_slices // 2
            # Generate Gaussian weights centered at the middle slice
            sigma = n_slices / 4.0  # Adjust sigma as needed
            slice_indices = np.arange(n_slices)
            weights = np.exp(-0.5 * ((slice_indices - mid_slice) / sigma) ** 2)
            weights /= np.sum(weights)  # Normalize weights to sum to 1

            for z in range(n_slices):
                aux = cv2.cvtColor(img[..., z], cv2.COLOR_GRAY2RGB)
                all_slices.append(aux)
                slice_image_indices.append(img_idx)
                slice_weights.append(weights[z])
        else:
            aux = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            all_slices.append(aux)
            slice_image_indices.append(img_idx)
            slice_weights.append(1.0)  # Full weight for single-slice images

        if verbose:
            bar.update()

    if verbose:
        bar.end()
        print(f'{c.OKGREEN}Total slices collected: {len(all_slices)}{c.ENDC}')

    # Convert lists to arrays
    all_slices = np.array(all_slices)  # Shape: (total_slices, height, width, channels)
    slice_image_indices = np.array(slice_image_indices)  # Shape: (total_slices,)
    slice_weights = np.array(slice_weights)  # Shape: (total_slices,)

    # Predict on all slices in batches
    batch_size = 32  # Adjust based on your GPU memory
    num_slices = len(all_slices)
    all_predictions = []

    if verbose:
        print(f'{c.OKBLUE}Predicting on all slices...{c.ENDC}')
        bar = LoadingBar(math.ceil(num_slices / batch_size))

    for start_idx in range(0, num_slices, batch_size):
        end_idx = min(start_idx + batch_size, num_slices)
        batch_slices = all_slices[start_idx:end_idx]
        batch_preds = model.model.predict(batch_slices, batch_size=batch_size, verbose=0)
        all_predictions.append(batch_preds)

        if verbose:
            bar.update()

    if verbose:
        bar.end()

    all_predictions = np.vstack(all_predictions)  # Shape: (total_slices, n_classes)

    # Group predictions by image and compute weighted averages
    num_images = total_images
    image_predictions = [[] for _ in range(num_images)]
    image_confidences = np.zeros(num_images)
    image_ids = unlabeled.img_short_names

    for img_idx in range(num_images):
        mask = slice_image_indices == img_idx
        img_slice_predictions = all_predictions[mask]  # Shape: (n_slices_per_image, n_classes)
        img_slice_weights = slice_weights[mask]  # Shape: (n_slices_per_image,)

        # Compute weighted average of the predicted probabilities
        weighted_avg_probs = np.average(img_slice_predictions, axis=0, weights=img_slice_weights)  # Shape: (n_classes,)
        confidence = np.max(weighted_avg_probs)

        image_predictions[img_idx] = weighted_avg_probs.tolist()
        image_confidences[img_idx] = confidence

    # Create DataFrame with pseudo-labels
    pseudo_labels_df = pd.DataFrame({
        'id': image_ids,
        'label': image_predictions,  # Soft labels as lists
        'confidence': image_confidences
    })

    # Dynamic threshold for curriculum learning
    dynamic_threshold = max(0.5, threshold - 0.05 * iteration)
    mask = pseudo_labels_df['confidence'] > dynamic_threshold

    selected_pseudo_labels_df = pseudo_labels_df[mask].copy()

    if verbose:
        print(f'{c.OKBLUE}Pseudo-labeling results{c.ENDC}')
        print(f'Pseudo-labels selected: {len(selected_pseudo_labels_df)}')

    if undersample:
        # Convert soft labels to hard labels for undersampling
        selected_pseudo_labels_df['hard_label'] = selected_pseudo_labels_df['label'].apply(lambda x: np.argmax(x))

        X = selected_pseudo_labels_df['id'].values.reshape(-1, 1)  # Features (IDs)
        y = selected_pseudo_labels_df['hard_label'].values  # Target labels

        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        # Retrieve the resampled IDs
        resampled_ids = X_res.flatten()

        # Filter the DataFrame to include only resampled IDs
        resampled_df = selected_pseudo_labels_df[selected_pseudo_labels_df['id'].isin(resampled_ids)]

        if verbose:
            print(f'{c.OKBLUE}After undersampling{c.ENDC}')
            print(f'Pseudo-labels after undersampling: {len(resampled_df)}')
            print(f'Class distribution after undersampling:\n{resampled_df["hard_label"].value_counts()}')

        # Drop the 'hard_label' column as it's no longer needed
        resampled_df = resampled_df.drop(columns=['hard_label'])

        # Update the pseudo_labels_df to be the undersampled DataFrame
        pseudo_labels_df = resampled_df.copy()

    # Remove pseudo-labeled instances from the unlabeled dataset
    selected_ids = pseudo_labels_df['id'].values
    idxs = [np.where(unlabeled.img_short_names == img_id)[0][0] for img_id in selected_ids]
    unlabeled.remove_images(idxs)

    # Save the pseudo-labels dataframe
    pseudo_labels_df.to_csv(v.data_path + 'CellDivision/undersampled/pseudo_labels.csv', index=False)

    return pseudo_labels_df


def split_pseudo_labeled_images(
        data_path=None,
        pseudo_labels_file='pseudo_labels.csv',
        verbose=0
):
    """
    Splits pseudo-labeled 3D images into 2D slices and assigns the corresponding soft labels.
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

        # The soft label is stored as a string representation of a list in 'label' column
        soft_label = np.array(eval(row['label']))  # Convert string to numpy array

        n_slices = img.shape[2]
        mid_slice = n_slices // 2
        # Generate Gaussian weights centered at the middle slice
        sigma = n_slices / 4.0  # Adjust sigma as needed
        slice_indices = np.arange(n_slices)
        weights = np.exp(-0.5 * ((slice_indices - mid_slice) / sigma) ** 2)
        weights /= np.sum(weights)

        for z in range(img.shape[2]):
            imaging.save_prediction(
                img[..., z],
                new_path + f'{row["id"].replace(".tif", "")}_{z}.tif',
                axes='XY'
            )

            # Optionally, you can adjust the soft labels per slice if needed
            # For simplicity, we assign the same soft label to each slice

            new_row = pd.DataFrame({
                'id': [f'{row["id"].replace(".tif", "")}_{z}.tif'],
                'label': [soft_label.tolist()]
            })
            new_df = pd.concat([new_df, new_row], ignore_index=True)

        if verbose:
            bar.update()

    if verbose:
        bar.end()
        print(f'{c.OKGREEN}Splitting completed{c.ENDC}')
        print(f'\t{c.BOLD}Total instances:{c.ENDC} {len(new_df)}')

    # Save the new dataframe with soft labels
    new_df.to_csv(data_path + f'CellDivision/undersampled/{pseudo_labels_file.replace(".csv", "_2d.csv")}', index=False)


def merge_datasets(train, pseudo_labels_file='pseudo_labels_2d.csv'):
    """
    Merge the labeled and pseudo-labeled data into a single dataset.
    """
    train.add_pseudo_labels(
        v.data_path + 'CellDivision/images_unlabeled_2d/',
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
    """
    results = []
    iters_without_improvement = 0
    initial_threshold = 0.9
    decay_rate = 0.05  # Adjust as needed

    for i in range(max_iter):
        dynamic_threshold = max(0.5, initial_threshold - decay_rate * i)
        if verbose:
            print(f'{c.OKGREEN}Iteration:{c.ENDC} {i + 1}')
            print(f'\t{c.OKBLUE}Pre-training...{c.ENDC}')
            print(f'\tUsing confidence threshold: {dynamic_threshold}')

        # Pre-train the model with the labeled data
        model = pre_train(model, train, val, batch_size=batch_size, verbose=verbose)
        results.append(model.model.evaluate(val, verbose=0)[1])

        # Early stopping and saving the model
        if i == 0:
            model.model.save(f'../models/cellular_division_models/vgg16_semi_{i}.h5')

        elif results[-1] > np.max(results[:-1]):
            model.model.save(f'../models/cellular_division_models/vgg16_semi_{i}.h5')
            iters_without_improvement = 0

        if i > 0 and results[-1] < np.max(results[:-1]):
            iters_without_improvement += 1
            if iters_without_improvement == 5:
                print(f'{c.OKGREEN}Early stopping at iteration {i + 1}{c.ENDC}')
                break

        if verbose:
            print_iter_results(model, train, test, val)
            print(f'\t{c.OKBLUE}Pseudo-labeling...{c.ENDC}')

        pseudo_labels_df = pseudo_labeling(
            model, unlabeled,
            threshold=dynamic_threshold,
            verbose=verbose,
            iteration=i
        )
        split_pseudo_labeled_images(verbose=verbose)
        train = merge_datasets(train)

    return model
