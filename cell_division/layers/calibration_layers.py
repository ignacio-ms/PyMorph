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
        # Initialize calibration parameters
        # Parameters a and b are vectors of size num_classes
        # Parameter c is a scalar bias
        initializer = tf.keras.initializers.Identity()
        self.A = self.add_weight(name='A',
                                 shape=(self.num_classes, self.num_classes),
                                 initializer=initializer,
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.num_classes,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, logits):
        # Apply the Dirichlet calibration transformation
        # Transformation: scaled_logits = softmax(A * logits + b)
        # Note: A is a matrix, b is a bias vector
        scaled_logits = tf.matmul(logits, self.A) + self.b
        return scaled_logits
