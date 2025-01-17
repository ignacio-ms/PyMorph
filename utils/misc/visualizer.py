import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cf_matrix, title='Confusion Matrix', sub_index=None, nrows=1, ncols=2):
    """
    Plot the confusion matrix.
    :param cf_matrix: Confusion matrix
    :param title: Title of the plot
    :param sub_index: Subplot index (Default: None)
    :return: None
    """
    if sub_index:
        plt.subplot(nrows, ncols, sub_index)
    else:
        plt.figure(figsize=(4, 4))

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)

    plt.imshow(
        cf_matrix, interpolation='nearest',
        alpha=0.7, cmap='Blues'
    )
    plt.xticks([0, 1, 2], ['P/M', 'A/T', 'I'], rotation=45)
    plt.yticks([0, 1, 2], ['P/M', 'A/T', 'I'])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, labels[i, j], ha='center', va='center', color='black', fontsize=12)

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.grid(False)
    plt.tight_layout()