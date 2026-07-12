import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_for_gradcam(image_path):
    """
    Load and preprocess image for Grad-CAM.
    """

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def get_last_conv_layer(model):
    """
    Automatically find the last Conv2D layer.
    """

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(model, image_path):
    """
    Generate Grad-CAM heatmap for a prediction.
    """

    img_array = preprocess_for_gradcam(image_path)

    last_conv_layer = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_index = 0

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    original = cv2.imread(image_path)

    original = cv2.resize(original, IMG_SIZE)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay