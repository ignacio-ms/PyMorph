# Standard Packages
import focal_loss
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Dense,
    Flatten, Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Custom Packages
from cell_division.nets.custom_layers import LSEPooling, w_cel_loss, focal_loss
from auxiliary import values as v


class CNN:
    def __init__(self, base=None, n_classes=3, input_shape=(50, 50, 3), fine_tune=False):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.model = None
        if base is None:
            base = tf.keras.applications.DenseNet169

        self.base_model = base(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet'
        )

        for layer in self.base_model.layers:
            layer.trainable = fine_tune

    def build_top(self, activation='softmax', b_type='CAM'):
        x = self.base_model.output

        if b_type == 'CAM':
            x_trans = Conv2D(
                1024, kernel_size=3, strides=1, padding='same',
                activation='relu', name='transition_layer'
            )(x)
            x_pool = LSEPooling(name='lse_pooling')(x_trans)
            predictions = Dense(
                self.n_classes, activation=activation,
                name='prediction_layer'
            )(x_pool)

        else:
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.n_classes, activation=activation)(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def compile(self, lr=1e-3, metrics=None, loss=None, optimizer=None):
        if metrics is None:
            metrics = [tf.keras.metrics.AUC(name='auc')]

        if loss is None:
            loss = w_cel_loss()

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def fit(self, train_gen, val_gen, epochs=100, batch_size=32, save=True, verbose=0):
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_auc', patience=3, factor=0.1, verbose=1)
        ]

        if save:
            callbacks.append(
                ModelCheckpoint(
                    v.data_path + f'models/cellular_division_models/{self.base_model.name}.h5',
                    save_best_only=True, save_weights_only=True,
                    monitor='val_auc'
                )
            )

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        if verbose > 1:
            # self.model.summary()

            acc = history.history['auc']
            val_acc = history.history['val_auc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training AUC')
            plt.plot(val_acc, label='Validation AUC')
            plt.legend(loc='lower right')
            plt.title('Training and Validation AUC')

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        return history

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                'LSEPooling': LSEPooling,
                'w_cel_loss': w_cel_loss()
            }
        )
