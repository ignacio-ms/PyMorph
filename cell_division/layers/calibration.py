import tensorflow as tf

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import numpy as np


class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureScaling, self).__init__(**kwargs)
        # Use log_temperature to ensure positivity
        self.temperature = tf.Variable(
            1.0, trainable=True, dtype=tf.float32,
            name='temperature'
        )

    def call(self, logits):
        # Apply temperature scaling with positive temperature
        # temperature = tf.exp(self.log_temperature)
        return tf.math.divide(logits, self.temperature)


class SigmoidCalibrationLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(SigmoidCalibrationLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        # Initialize alpha and beta parameters for each class
        self.alpha = self.add_weight(
            shape=(num_classes,),
            initializer='ones',
            trainable=True,
            name='alpha'
        )
        self.beta = self.add_weight(
            shape=(num_classes,),
            initializer='zeros',
            trainable=True,
            name='beta'
        )

    def call(self, inputs):
        # Apply per-class calibration: calibrated_logits = alpha * logits + beta
        calibrated_logits = inputs * self.alpha + self.beta
        # Apply sigmoid activation to get calibrated probabilities
        calibrated_probs = tf.nn.sigmoid(calibrated_logits)
        return calibrated_probs


class BetaCalibration:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None

    def fit(self, prob_pos, y_true):
        # Avoid probabilities of 0 or 1
        prob_pos = np.clip(prob_pos, 1e-15, 1 - 1e-15)
        y_true = y_true.astype(int)
        X = np.vstack([np.log(prob_pos), np.log(1 - prob_pos)]).T
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept
        y = y_true

        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(X, y)
        self.a = lr.coef_[0][0]
        self.b = lr.coef_[0][1]
        self.c = lr.intercept_[0]

    def predict(self, prob_pos):
        # Avoid probabilities of 0 or 1
        prob_pos = np.clip(prob_pos, 1e-15, 1 - 1e-15)
        X = np.vstack([np.log(prob_pos), np.log(1 - prob_pos)]).T
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add intercept
        log_odds = self.a * X[:, 0] + self.b * X[:, 1] + self.c
        return 1 / (1 + np.exp(-log_odds))

