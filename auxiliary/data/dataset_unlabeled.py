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
from auxiliary.data import imaging


class UnlabeledDataset(tf.keras.utils.Sequence):
    def __init__(self, img_path=None, batch_size=32, resize=(50, 50)):
        self.N_CLASSES = 3
        self.CLASS_NAMES = ['Prophase/Metaphase', 'Anaphase/Telophase', 'Interphase']
        self.CLASSES = ['0', '1', '2']

        if img_path is None:
            img_path = v.data_path + 'CellDivision/images_unlabeled/'

        img_names = [
            os.path.join(img_path, f)
            for f in os.listdir(img_path)
            if f.endswith('.tif')
        ]

        self.img_short_names = np.array([os.path.basename(f) for f in img_names])
        self.img_names = np.array(img_names)
        self.batch_size = batch_size
        self.resize = resize

    def __get_image(self, idx):
        img = imaging.read_image(idx)
        if img is not None:
            if self.resize:
                img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
            img = img / 255.0
            img = img.astype(np.float32)
        return img

    def remove_images(self, idxs):
        self.img_names = np.delete(self.img_names, idxs)
        self.img_short_names = np.delete(self.img_short_names, idxs)

    def __getitem__(self, item):
        batch_names = self.img_names[item * self.batch_size:(item + 1) * self.batch_size]
        batch_images = [self.__get_image(name) for name in batch_names]

        return np.array(batch_images)

    def __len__(self):
        return math.ceil(len(self.img_names) / self.batch_size)
