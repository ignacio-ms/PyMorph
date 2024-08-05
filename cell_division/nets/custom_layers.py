import tensorflow as tf


class LSEPooling(tf.keras.layers.Layer):
    def __init__(self, axis=None, r=10, **kwargs):
        super(LSEPooling, self).__init__(**kwargs)
        if axis is None:
            axis = [1, 2]
        self.axis = axis
        self.r = r

    def call(self, inputs, *args, **kwargs):
        # return tf.math.reduce_logsumexp(inputs, axis=self.axis)
        max_input = tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        lse = max_input + (1.0 / self.r) * tf.math.log(tf.reduce_mean(tf.exp(self.r * (inputs - max_input)), axis=self.axis, keepdims=True))
        return lse

    def get_config(self):
        config = super(LSEPooling, self).get_config()
        config.update({'axis': self.axis})
        return config


class ExtendedLSEPooling(tf.keras.layers.Layer):
    def __init__(self, axis=None, r=10, **kwargs):
        super(ExtendedLSEPooling, self).__init__(**kwargs)
        if axis is None:
            axis = [1, 2]
        self.axis = axis
        self.r = r

    def call(self, inputs, *args, **kwargs):
        S = tf.cast(tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.float32)
        x_star = tf.reduce_max(inputs, axis=self.axis, keepdims=True)

        lse = x_star + (1.0 / self.r) * tf.math.log(tf.reduce_sum(tf.exp(self.r * (inputs - x_star)), axis=self.axis, keepdims=True) / S)
        return lse

    def get_config(self):
        config = super(ExtendedLSEPooling, self).get_config()
        config.update({'axis': self.axis})
        return config


def w_cel_loss():
    def weighted_cross_entropy_with_logits(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        positive_weight = tf.reduce_sum(y_true) / tf.cast(tf.size(y_true), tf.float32)
        weight = 1 / (positive_weight + 1e-9)

        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=weight)
        return tf.reduce_mean(loss)

    return weighted_cross_entropy_with_logits


def extended_w_cel_loss():
    def ext_weighted_cross_entropy_with_logits(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        beta_p = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(y_true, axis=0) + 1e-6)
        beta_n = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(1 - y_true, axis=0) + 1e-6)

        loss = beta_p * y_true * tf.math.log(y_pred + 1e-6) + beta_n * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
        return -tf.reduce_mean(loss)

    return ext_weighted_cross_entropy_with_logits


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

