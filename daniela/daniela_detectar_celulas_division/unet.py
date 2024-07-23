import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
import tensorflow as tf
import os
from PIL import Image
import cv2
from keras.utils import to_categorical


def setup_gpu():
    print("BUSCANDO GPU")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        print("FOUND")
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            print(e)


def create(fileraw, filemask, hw):
    p_img = np.array(Image.open(fileraw)).astype("uint16")
    p_mask = np.array(Image.open(filemask)).astype("uint16")
    p_img = cv2.resize(p_img, (hw, hw))
    p_mask = cv2.resize(p_mask, (hw, hw))
    # p_mask = cv2.cvtColor(p_mask,cv2.COLOR_BGR2GRAY)
    p_img = cv2.cvtColor(p_img, cv2.COLOR_GRAY2RGB)
    print(p_mask.shape)
    print(p_img.shape)
    return p_img, p_mask


def Generator(listfileraw, listfilemask, batch_size, hw):
    while 1:
        b = 0
        all_i = []
        all_m = []
        # random.shuffle(X_list)
        for i in range(batch_size):
            image, mask = create(listfileraw[i], listfilemask[i], hw)
            print("here")
            print(image.shape)
            # image,mask = create(X_list[i-1],'dataset/data1/')
            all_i.append(image)
            mask = mask.reshape((hw, hw, 1))
            all_m.append(mask)
            b += 1
        all_i = np.array(all_i)
        all_m = np.array(all_m)
        all_m = to_categorical(all_m)
        # print(all_m[0])
        # print(all_m.shape)
        yield np.array(all_i), np.array(all_m)


num_classes = 2
channels = 3
hw = 1024
setup_gpu()


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size)
    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


folderRAW = "raws"
folderMASK = "masks"
list_raw = [os.path.join(folderRAW, i) for i in os.listdir(folderRAW)]
list_masks = [
    os.path.join(folderMASK, r.replace("_v0.tif", "_mask.tif"))
    for r in os.listdir(folderRAW)
]
model = get_model((1024, 1024, 3), num_classes)
gen = Generator(list_raw, list_masks, 2, 1024)
model.fit(gen, steps_per_epoch=48, epochs=2)
model.save("model.h5")
