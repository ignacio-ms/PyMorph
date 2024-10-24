# Standard Packages
import focal_loss
import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Dense,
    Flatten, Dropout,
    BatchNormalization,
    Activation
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

import matplotlib.pyplot as plt

# Custom Packages
from cell_division.layers.calibration_layers import (
    VectorScalingLayer,
    TemperatureScalingLayer,
    MatrixScalingLayer,
    DirichletCalibrationLayer
)
from cell_division.layers.custom_layers import (
    LSEPooling,
    w_cel_loss,
    focal_loss,
    extended_w_cel_loss,
    ExtendedLSEPooling,
    extended_w_cel_loss_soft
)


class CNN:
    def __init__(self, base=None, n_classes=3, input_shape=(50, 50, 3), fine_tune=False):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.model = None
        if base is None:
            base = tf.keras.applications.VGG16

        self.base_model = base(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet'
        )

        for layer in self.base_model.layers:
            layer.trainable = fine_tune

        # Completely remove last 8 layers
        for i in range(8):
            self.base_model.layers.pop()

    def build_top(self, activation='softmax', b_type='CAM', pooling=None):
        x = self.base_model.output

        if b_type == 'CAM':
            if pooling is None:
                pooling = ExtendedLSEPooling

            x_trans = Conv2D(
                1024, kernel_size=1, strides=1,
                padding='same', name='transition_layer'
            )(x)
            x_trans = BatchNormalization()(x_trans)
            x_trans = Activation('relu')(x_trans)
            x_trans = Dropout(rate=0.5)(x_trans)

            x_pool = pooling(name='lse_pooling')(x_trans)
            x_pool = Flatten(name='lse_flatten')(x_pool)
            x_pool = Dropout(rate=0.5)(x_pool)

            predictions = Dense(
                self.n_classes, activation=activation,
                name='prediction_layer'
            )(x_pool)

        else:
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.n_classes, activation=activation)(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        n_layers = 8 if b_type == 'CAM' else 8
        for layer in self.model.layers[-n_layers:]:
            layer.trainable = True

    def compile(self, lr=1e-3, metrics=None, loss=None, optimizer=None):
        if metrics is None:
            metrics = [tf.keras.metrics.AUC(name='auc', multi_label=True)]

        if loss is None:
            loss = extended_w_cel_loss(from_logits=False)

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def fit(self, train_gen, val_gen, epochs=100, batch_size=32, save=False, verbose=0):
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_auc', patience=3, factor=0.1, verbose=1)
        ]

        if save:
            callbacks.append(
                ModelCheckpoint(
                    f'../models/cellular_division_models/{self.base_model.name}.h5',
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_auc'
                )
            )

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

        if verbose > 1:
            self.model.summary()

            acc = history.history['auc']
            val_acc = history.history['val_auc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(12, 8))
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

    def add_calibration_layers(self, method='temperature'):
        if self.model.layers[-1].activation != tf.keras.activations.softmax and method != 'dirichlet':
            logits = Dense(
                self.n_classes,
                activation=None,
                name='logits_layer'
            )(self.model.layers[-2].output)
        else:
            logits = self.model.output

        if method == 'temperature':
            temperature_layer = TemperatureScalingLayer(name='temperature_scaling')
            scaled_logits = temperature_layer(logits)
        elif method == 'vector':
            vector_layer = VectorScalingLayer(name='vector_scaling')
            scaled_logits = vector_layer(logits)
        elif method == 'matrix':
            matrix_layer = MatrixScalingLayer(name='matrix_scaling')
            scaled_logits = matrix_layer(logits)
        elif method == 'dirichlet':
            dirichlet_layer = DirichletCalibrationLayer(name='dirichlet_scaling')
            scaled_logits = dirichlet_layer(logits)
            self.calibrated_model = Model(inputs=self.model.input, outputs=scaled_logits)
            return
        else:
            raise ValueError("Unsupported calibration method")

        calibrated_probs = Activation('softmax', name='calibrated_softmax')(scaled_logits)
        self.calibrated_model = Model(inputs=self.model.input, outputs=calibrated_probs)

    def compile_calibrated_model(self, lr=5e-2, metrics=None, loss=None, optimizer=None):
        if self.calibrated_model is None:
            raise ValueError("Calibrated model is not created. Call add_temperature_scaling() first.")

        if metrics is None:
            metrics = [tf.keras.metrics.AUC(name='auc', multi_label=True)]

        if loss is None:
            loss = extended_w_cel_loss(from_logits=False)

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Freeze all layers except the calibration layers
        for layer in self.calibrated_model.layers:
            if layer.name not in [
                'temperature_scaling', 'vector_scaling',
                'matrix_scaling', 'dirichlet_scaling'
            ]:
                layer.trainable = False
            else:
                layer.trainable = True

        print("Trainable variables: ", len(self.calibrated_model.trainable_variables))
        for var in self.calibrated_model.trainable_variables:
            print(var.name)

        self.calibrated_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def fit_calibrated_model(self, X_calib, y_calib, epochs=100, batch_size=32):
        if self.calibrated_model is None:
            raise ValueError("Calibrated model is not created. Call add_temperature_scaling() first.")

        callbacks = [
            EarlyStopping(monitor='auc', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='auc', patience=5, factor=0.1, verbose=1)
        ]

        # Train only the temperature parameter
        self.calibrated_model.fit(
            X_calib, y_calib,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def calibrate(self, X_calib, y_calib, method='temperature'):
        assert self.model is not None, "Model is not trained yet"
        assert method in ['temperature', 'vector', 'matrix', 'dirichlet'], "Unsupported calibration method"

        self.add_calibration_layers(method)
        self.compile_calibrated_model()
        self.fit_calibrated_model(X_calib, y_calib)

    def predict_proba(self, X):
        assert self.model is not None, "Model is not trained yet"

        if hasattr(self, 'calibrated_model') and self.calibrated_model is not None:
            # Use the calibrated model for predictions
            return self.calibrated_model.predict(X)
        else:
            return self.model.predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                'LSEPooling': LSEPooling,
                'ExtendedLSEPooling': ExtendedLSEPooling,
                'weighted_cross_entropy_with_logits': w_cel_loss(),
                'focal_loss': focal_loss(),
                'ext_weighted_cross_entropy_with_logits': extended_w_cel_loss(),
                'ext_weighted_cross_entropy_with_logits_soft': extended_w_cel_loss_soft(),
                'TemperatureScalingLayer': TemperatureScalingLayer,
                'VectorScalingLayer': VectorScalingLayer,
                'MatrixScalingLayer': MatrixScalingLayer,
                'DirichletCalibrationLayer': DirichletCalibrationLayer
            }
        )
