import numpy as np
from PIL import Image
import tensorflow as tf

# Image size expected by DenseNet121
IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Load and preprocess an image for DenseNet121 model.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image).astype("float32")

    # DenseNet121 preprocessing
    image = tf.keras.applications.densenet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


def preprocess_image_densenet(image_path):
    """
    Alternative preprocessing using TensorFlow's
    DenseNet preprocessing function.
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image = np.array(image).astype("float32")

    image = tf.keras.applications.densenet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


def decode_prediction(probability):
    """
    Convert prediction probability into class label.

    Parameters:
        probability (float)

    Returns:
        tuple -> (prediction, confidence)
    """

    if probability >= 0.5:
        prediction = "PNEUMONIA"
        confidence = probability * 100
    else:
        prediction = "NORMAL"
        confidence = (1 - probability) * 100

    return prediction, confidence


def allowed_file(filename):
    """
    Check whether uploaded file has a valid image extension.
    """

    allowed_extensions = {"png", "jpg", "jpeg"}

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def get_prediction(model, image_path):
    """
    Complete prediction pipeline.

    Parameters:
        model : Loaded TensorFlow model
        image_path : Path to image

    Returns:
        prediction, confidence
    """

    image = preprocess_image(image_path)

    probability = float(model.predict(image, verbose=0)[0][0])

    prediction, confidence = decode_prediction(probability)

    return prediction, confidence