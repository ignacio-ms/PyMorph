import tensorflow as tf


class LSEPooling(tf.keras.layers.Layer):
    def __init__(self, axis=None, **kwargs):
        super(LSEPooling, self).__init__(**kwargs)
        if axis is None:
            axis = [1, 2]
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        # return tf.math.log(tf.reduce_sum(tf.exp(inputs), axis=self.axis))
        return tf.math.reduce_logsumexp(inputs, axis=self.axis)

    def get_config(self):
        config = super(LSEPooling, self).get_config()
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

