import tensorflow as tf

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import numpy as np


class TemperatureScalingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureScalingLayer, self).__init__(**kwargs)
        # Use log_temperature to ensure positivity
        self.temperature = tf.Variable(
            1.0, trainable=True, dtype=tf.float32,
            name='temperature'
        )

    def call(self, logits):
        # Apply temperature scaling with positive temperature
        # temperature = tf.exp(self.log_temperature)
        return tf.math.divide(logits, self.temperature)


class VectorScalingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VectorScalingLayer, self).__init__(**kwargs)
        self.num_classes = 3
        # Initialize scaling vector and bias vector
        self.scale = tf.Variable(
            tf.ones([self.num_classes]),
            trainable=True,
            name='scale',
            dtype=tf.float32
        )
        self.bias = tf.Variable(
            tf.zeros([self.num_classes]),
            trainable=True,
            name='bias',
            dtype=tf.float32
        )

    def call(self, logits):
        # Apply element-wise scaling and bias
        return self.scale * logits + self.bias


class MatrixScalingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatrixScalingLayer, self).__init__(**kwargs)
        self.num_classes = 3
        # Initialize weight matrix W and bias vector b
        # W is initialized as an identity matrix
        initializer = tf.keras.initializers.Identity()
        self.W = tf.Variable(initial_value=initializer(shape=(self.num_classes, self.num_classes), dtype='float32'),
                             trainable=True, name='W')
        self.b = tf.Variable(initial_value=tf.zeros([self.num_classes], dtype='float32'),
                             trainable=True, name='b')

    def call(self, logits):
        # Apply matrix scaling transformation
        scaled_logits = tf.matmul(logits, self.W) + self.b
        return scaled_logits


class DirichletCalibrationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DirichletCalibrationLayer, self).__init__(**kwargs)
        self.num_classes = 3
        # Initialize alpha parameters (positive values)
        self.alpha = self.add_weight(
            name='alpha',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )

    def call(self, probabilities):
        # Ensure alpha parameters are positive
        alpha = tf.nn.softplus(self.alpha)
        # Apply Dirichlet calibration
        adjusted_probs = tf.pow(probabilities, alpha)
        # Normalize to ensure probabilities sum to 1
        adjusted_probs /= tf.reduce_sum(adjusted_probs, axis=1, keepdims=True)
        return adjusted_probs

