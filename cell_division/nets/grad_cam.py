from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    output = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), alpha, heatmap, 1 - alpha, 0)

    return output


class GradCAM:
    def __init__(self, model):
        self.model = model

    def compute_heatmap(self, image, class_idx, normalize=True, eps=1e-9):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer('transition_layer').output,
                self.model.get_layer('prediction_layer').output
            ]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]

        cast_conv_outputs = tf.cast(conv_outputs > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        h, w = image.shape[1], image.shape[2]
        heatmap = cv2.resize(cam.numpy(), (h, w))
        if normalize:
            heatmap = np.maximum(heatmap, 0) / (heatmap.max() + eps)

        return heatmap
