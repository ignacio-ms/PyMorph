# Standard Packages
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

import math
import sys
import os
from skimage import io

# Custom Packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v


class CellDataset(tf.keras.utils.Sequence):
    def __init__(self, img_path, labels_path, batch_size=32, resize=(50, 50)):
        self.N_CLASSES = 3
        self.CLASS_NAMES = ['Prophase/Metaphase', 'Anaphase/Telophase', 'Interphase']
        self.CLASSES = ['0', '1', '2']

        img_names, img_labels = [], []

        def format_names(idx):
            return f'{idx:03}.tif'

        if isinstance(labels_path, pd.DataFrame):
            for idx, row in labels_path.iterrows():
                img_names.append(os.path.join(img_path, format_names(row['id'])))
                img_labels.append(row['label'])

        else:
            with open(labels_path, 'r') as f:
                for line in f:
                    if line.startswith('id'):
                        continue

                    items = line.split()[0].split(',')

                    img_names.append(os.path.join(img_path, format_names(int(items[0]))))
                    img_labels.append(items[1])

        self.img_names = np.array(img_names)
        self.img_labels = np.array(img_labels)

        self.batch_size = batch_size
        self.resize = resize

    def add_pseudo_labels(self, img_path, labels_path):
        img_names, img_labels = [], []

        if isinstance(labels_path, pd.DataFrame):
            for idx, row in labels_path.iterrows():
                img_names.append(os.path.join(img_path, row['id'] + '.tif'))
                img_labels.append(row['label'])

        else:
            with open(labels_path, 'r') as f:
                for line in f:
                    if line.startswith('id'):
                        continue

                    items = line.split()[0].split(',')

                    img_names.append(os.path.join(img_path, items[0] + '.tif'))
                    img_labels.append(items[1])

        self.img_names = np.concatenate((self.img_names, np.array(img_names)))
        self.img_labels = np.concatenate((self.img_labels, np.array(img_labels)))

    def __get_image(self, idx):
        img = io.imread(idx)
        if img is not None:
            if self.resize:
                img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
            img = img / 255.0
            img = img.astype(np.float16)
        return img

    def __get_label(self, idx):
        return tf.one_hot(np.int16(idx), self.N_CLASSES, dtype=tf.int16)

    def __len__(self):
        return math.ceil(len(self.img_names) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.img_names[item * self.batch_size:(item + 1) * self.batch_size]
        batch_y = self.img_labels[item * self.batch_size:(item + 1) * self.batch_size]

        x = [self.__get_image(idx) for idx in batch_x]
        y = [self.__get_label(idx) for idx in batch_y]

        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    def oh2class(self, oh):
        return self.CLASSES[np.argmax(oh)]

    def oh2name(self, oh):
        return self.CLASS_NAMES[np.argmax(oh)]

    def class2oh(self, c):
        return tf.one_hot(np.int8(c), self.N_CLASSES, dtype=tf.int8)
