from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image = np.uint8(255 * image) if image.dtype != np.uint8 else image
    output = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), alpha, heatmap, 1 - alpha, 0)

    return output


class GradCAM:
    def __init__(self, model):
        self.model = model

    def compute_heatmap(self, image, class_idx, eps=1e-9):
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

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None, check the model and input data.")

        grads = grads[0]

        cast_conv_outputs = tf.cast(conv_outputs > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        h, w = image.shape[1], image.shape[2]
        heatmap = cv2.resize(cam.numpy(), (w, h))
        heatmap = np.maximum(heatmap, 0) / (heatmap.max() + eps)

        return heatmap


class CAM:
    def __init__(self, model, layer_name='transition_layer'):
        self.model = model
        self.layer_name = layer_name
        self.feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        self.weights = self.model.get_layer('prediction_layer').get_weights()[0]

    def compute_heatmap(self, image, class_idx, normalize=True, eps=1e-9):
        conv_outputs = self.feature_model.predict(image)
        conv_outputs = np.squeeze(conv_outputs)

        class_weights = self.weights[:, class_idx]

        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i, w in enumerate(class_weights):
            cam += w * conv_outputs[:, :, i]

        heatmap = cv2.resize(cam, (image.shape[2], image.shape[1]))
        if normalize:
            heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)

        return heatmap
