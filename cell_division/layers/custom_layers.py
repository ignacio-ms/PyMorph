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
        # Ensure same dtype for all operations
        inputs = tf.cast(inputs, tf.float16)

        S = tf.cast(tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.float16)
        x_star = tf.reduce_max(inputs, axis=self.axis, keepdims=True)

        lse = x_star + (1.0 / self.r) * tf.math.log(tf.reduce_sum(tf.exp(self.r * (inputs - x_star)), axis=self.axis, keepdims=True) / S)
        return lse

    def get_config(self):
        config = super(ExtendedLSEPooling, self).get_config()
        config.update({'axis': self.axis})
        return config


def w_cel_loss():
    def weighted_cross_entropy_with_logits(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float16)
        y_pred = tf.cast(y_pred, tf.float16)

        positive_weight = tf.reduce_sum(y_true) / tf.cast(tf.size(y_true), tf.float16)
        weight = 1 / (positive_weight + 1e-9)

        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=weight)
        return tf.reduce_mean(loss)

    return weighted_cross_entropy_with_logits


def extended_w_cel_loss(from_logits=False):
    def ext_weighted_cross_entropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float16)
        y_pred = tf.cast(y_pred, tf.float16)

        beta_p = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(y_true, axis=0) + 1e-6)
        beta_n = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(1 - y_true, axis=0) + 1e-6)

        loss = beta_p * y_true * tf.math.log(y_pred + 1e-6) + beta_n * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
        return -tf.reduce_mean(loss)

    def ext_weighted_cross_entropy_with_logits(y_true, logits):
        y_true = tf.cast(y_true, tf.float16)
        logits = tf.cast(logits, tf.float16)

        # Compute probabilities from logits using sigmoid for multilabel classification
        y_pred = tf.nn.sigmoid(logits)

        beta_p = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(y_true, axis=0) + 1e-6)
        beta_n = (tf.reduce_sum(1 - y_true, axis=0) + tf.reduce_sum(y_true, axis=0)) / (tf.reduce_sum(1 - y_true, axis=0) + 1e-6)

        loss = beta_p * y_true * tf.math.log(y_pred + 1e-6) + beta_n * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
        return -tf.reduce_mean(loss)

    return ext_weighted_cross_entropy_with_logits if from_logits else ext_weighted_cross_entropy


def extended_w_cel_loss_multiclass(from_logits=False):
    """
    Returns a Weighted Cross-Entropy Loss function tailored for multi-class classification.

    Parameters
    ----------
    num_classes : int
        Number of classes in your multi-class classification problem.
    from_logits : bool
        If True, applies softmax to logits before computing loss.
        Otherwise, assumes y_pred is already probabilities.

    Returns
    -------
    A callable that takes (y_true, y_pred) or (y_true, logits)
    and returns a scalar loss value.
    """

    def ext_weighted_cross_entropy_multiclass(y_true, y_pred):
        """
        Weighted Cross-Entropy for multi-class classification when y_pred is already probabilities.

        y_true: one-hot encoded labels, shape = (batch_size, num_classes).
        y_pred: probabilities, shape = (batch_size, num_classes).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. Compute total count of each class across the batch
        class_counts = tf.reduce_sum(y_true, axis=0)  # shape = (num_classes,)
        total_count = tf.reduce_sum(class_counts)  # scalar

        # 2. Compute per-class weights: higher weight for rarer classes
        #    Avoid division by zero by adding a small epsilon
        eps = 1e-6
        weights = total_count / (class_counts + eps)  # shape = (num_classes,)

        # 3. Compute per-sample cross-entropy
        #    weighted_ce[i] = - sum_c [ w_c * y_true[i, c] * log(y_pred[i, c]) ]
        weighted_ce = -tf.reduce_sum(y_true * tf.math.log(y_pred + eps) * weights, axis=1)

        # 4. Return mean loss across the batch
        return tf.reduce_mean(weighted_ce)

    def ext_weighted_cross_entropy_multiclass_with_logits(y_true, logits):
        """
        Weighted Cross-Entropy for multi-class classification when logits are provided.

        y_true: one-hot encoded labels, shape = (batch_size, num_classes).
        logits: raw model outputs, shape = (batch_size, num_classes).
        """
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)

        # 1. Convert logits -> probabilities with softmax
        y_pred = tf.nn.softmax(logits, axis=-1)  # shape = (batch_size, num_classes)

        # 2. Compute total count of each class across the batch
        class_counts = tf.reduce_sum(y_true, axis=0)  # shape = (num_classes,)
        total_count = tf.reduce_sum(class_counts)  # scalar

        # 3. Compute per-class weights
        eps = 1e-6
        weights = total_count / (class_counts + eps)  # shape = (num_classes,)

        # 4. Weighted cross-entropy
        weighted_ce = -tf.reduce_sum(y_true * tf.math.log(y_pred + eps) * weights, axis=1)

        # 5. Return mean loss across the batch
        return tf.reduce_mean(weighted_ce)

    # Return the correct function based on from_logits
    return (
        ext_weighted_cross_entropy_multiclass_with_logits
        if from_logits
        else ext_weighted_cross_entropy_multiclass
    )



# def extended_w_cel_loss(from_logits=False):
#     def ext_weighted_cross_entropy(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float16)
#         y_pred = tf.cast(y_pred, tf.float16)
#
#         # Calculate beta_p and beta_n for each class
#         total_count = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(1 - y_true, axis=0)
#         beta_p = total_count / (tf.reduce_sum(y_true, axis=0) + 1e-6)
#         beta_n = total_count / (tf.reduce_sum(1 - y_true, axis=0) + 1e-6)
#
#         # Calculate per-class loss
#         loss = beta_p * y_true * tf.math.log(y_pred + 1e-6) + \
#                beta_n * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
#         return -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # Summing over classes
#
#     def ext_weighted_cross_entropy_with_logits(y_true, logits):
#         y_true = tf.cast(y_true, tf.float16)
#         logits = tf.cast(logits, tf.float16)
#
#         # Compute probabilities from logits using softmax for multi-class classification
#         y_pred = tf.nn.softmax(logits, axis=-1)
#
#         # Calculate beta_p and beta_n for each class
#         total_count = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(1 - y_true, axis=0)
#         beta_p = total_count / (tf.reduce_sum(y_true, axis=0) + 1e-6)
#         beta_n = total_count / (tf.reduce_sum(1 - y_true, axis=0) + 1e-6)
#
#         # Calculate per-class loss
#         loss = beta_p * y_true * tf.math.log(y_pred + 1e-6) + \
#                beta_n * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
#         return -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # Summing over classes
#
#     return ext_weighted_cross_entropy_with_logits if from_logits else ext_weighted_cross_entropy


def extended_w_cel_loss_soft():
    def ext_weighted_cross_entropy_with_logits_soft(y_true, y_pred):
        """
        Extended weighted cross-entropy loss that handles soft labels.
        """
        y_true = tf.cast(y_true, tf.float16)
        y_pred = tf.cast(y_pred, tf.float16)

        # Compute class weights based on the batch
        class_totals = tf.reduce_sum(y_true, axis=0)
        class_weights = tf.reduce_max(class_totals) / (class_totals + 1e-6)

        # Compute the categorical cross-entropy
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply class weights
        weighted_loss = tf.reduce_sum(class_weights * loss)

        # Return the mean loss over the batch
        return weighted_loss / tf.cast(tf.shape(y_true)[0], tf.float16)

    return ext_weighted_cross_entropy_with_logits_soft


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float16)
        y_pred = tf.cast(y_pred, tf.float16)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

