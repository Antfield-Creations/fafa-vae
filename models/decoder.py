import numpy as np
from tensorflow import keras

from models.loaders.config import Config


def get_decoder(config: Config) -> keras.Model:
    """
    Constructs the decoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the decoder
    """

    latent_dim = config['models']['vae']['latent_dim']

    latent_inputs = keras.Input(shape=(latent_dim,))

    convs = config['models']['vae']['conv2d']
    img_width = config['images']['width']
    img_height = config['images']['height']
    img_channels = config['images']['channels']

    reductions = np.product([conv['strides'] for conv in convs])
    width_over_deconv = int(img_width / reductions)
    height_over_deconv = int(img_height / reductions)

    x = keras.layers.Dense(
        width_over_deconv * height_over_deconv * convs[-1]['filters'], activation="relu")(latent_inputs)
    x = keras.layers.Reshape(
        (width_over_deconv, height_over_deconv, convs[-1]['filters']))(x)

    # The conv2d layers in reverse:
    for conv in reversed(convs):
        x = keras.layers.Conv2DTranspose(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            strides=conv['strides'],
            padding="same"
        )(x)
        x = keras.layers.LeakyReLU()(x)

    decoder_outputs = keras.layers.Conv2DTranspose(
        filters=img_channels,
        kernel_size=3,
        activation="tanh",
        strides=1,
        padding="same",
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
