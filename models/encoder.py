import tensorflow as tf
from tensorflow import keras

from models.loaders.config import Config


def get_encoder(config: Config) -> keras.Model:
    """
    Constructs the encoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the encoder
    """

    width = config['data']['images']['width']
    height = config['data']['images']['height']
    channels = config['data']['images']['channels']

    convs = config['models']['vq_vae']['conv2d']

    inputs = keras.Input(shape=(width, height, channels))

    # Instantiate layers from first convs specification
    encoder_layers = keras.layers.Conv2D(
        filters=convs[0]['filters'],
        kernel_size=convs[0]['kernel_size'],
        strides=convs[0]['strides'],
        padding="same",
    )(inputs)

    for conv in convs[1:]:
        encoder_layers = keras.layers.Conv2D(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            strides=conv['strides'],
            padding="same",
        )(encoder_layers)
        encoder_layers = keras.layers.ReLU()(encoder_layers)

    encoder = tf.keras.Model(inputs, encoder_layers, name="encoder")
    return encoder
