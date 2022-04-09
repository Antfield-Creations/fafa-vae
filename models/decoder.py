import numpy as np
from tensorflow import keras

from models.loaders.config import Config


def get_decoder(config: Config) -> keras.Model:
    """
    Constructs the decoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the decoder
    """

    latent_size = config['models']['vae']['latent_size']

    latent_inputs = keras.Input(shape=(latent_size,))

    convs = config['models']['vae']['conv2d']
    img_width = config['images']['width']
    img_height = config['images']['height']
    img_channels = config['images']['channels']

    reductions = np.product([conv['strides'] for conv in convs])
    width_over_deconv = int(img_width / reductions)
    height_over_deconv = int(img_height / reductions)

    decoder_layers = keras.layers.Dense(
        width_over_deconv * height_over_deconv * convs[-1]['filters'], activation="relu")(latent_inputs)
    decoder_layers = keras.layers.Reshape(
        (width_over_deconv, height_over_deconv, convs[-1]['filters']))(decoder_layers)

    # The conv2d layers in reverse:
    for conv in reversed(convs):
        decoder_layers = keras.layers.Conv2DTranspose(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            strides=conv['strides'],
            padding="same",
            kernel_initializer='random_normal',
        )(decoder_layers)
        decoder_layers = keras.layers.LeakyReLU()(decoder_layers)

    decoder_outputs = keras.layers.Conv2DTranspose(
        filters=img_channels,
        kernel_size=3,
        # activation="tanh",
        strides=1,
        padding="same",
    )(decoder_layers)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
