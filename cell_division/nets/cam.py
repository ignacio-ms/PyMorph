from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_HSV):
    image = np.uint8(255 * image) if image.dtype != np.uint8 else image
    heatmap = np.uint8(255 * heatmap) if heatmap.dtype != np.uint8 else heatmap

    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), alpha, heatmap, 1 - alpha, 0)

    return output


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


class GradCAM:
    def __init__(self, model):
        self.model = model

    def compute_heatmap(self, image, class_idx, eps=1e-9):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer('transition_layer').output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None, check the model and input data.")

        cast_conv_outputs = tf.cast(conv_outputs > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        h, w = image.shape[1], image.shape[2]
        heatmap = cv2.resize(cam.numpy(), (w, h))

        num = heatmap - np.min(heatmap)
        den = (heatmap.max() - heatmap.min()) + eps
        heatmap = num / den

        return (heatmap * 255).astype(np.uint8)


class GradCAMpp:
    def __init__(self, model):
        self.model = model

    def compute_heatmap(self, image, class_idx, eps=1e-9):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer('transition_layer').output,
                self.model.output
            ]
        )

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    inputs = tf.cast(image, tf.float32)
                    (conv_output, predictions) = grad_model(inputs)
                    loss = predictions[:, class_idx]

                    conv_first_grad = gtape3.gradient(loss, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

        global_sum = np.sum(conv_output, axis=(0, 1, 2))

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
        alphas /= alpha_normalization_constant

        weights = np.maximum(conv_first_grad[0], 0.0)

        deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
        grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

        h, w = image.shape[1], image.shape[2]
        heatmap = cv2.resize(grad_cam_map, (w, h))

        num = heatmap - np.min(heatmap)
        den = (heatmap.max() - heatmap.min()) + eps
        heatmap = num / den

        return (heatmap * 255).astype(np.uint8)
