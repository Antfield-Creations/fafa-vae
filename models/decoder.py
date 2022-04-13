from tensorflow import keras

from models.encoder import get_encoder
from models.loaders.config import Config


def get_decoder(config: Config) -> keras.Model:
    """
    Constructs the decoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the decoder
    """

    latent_inputs = keras.Input(shape=get_encoder(config).output.shape[1:])

    convs = config['models']['vqvae']['conv2d']
    # The conv2d layers in reverse:
    convs = list(reversed(convs))
    img_channels = config['images']['channels']

    decoder_layers = keras.layers.Conv2DTranspose(
        filters=convs[0]['filters'],
        kernel_size=convs[0]['kernel_size'],
        strides=convs[0]['strides'],
        padding="same",
    )(latent_inputs)

    for conv in convs[1:]:
        decoder_layers = keras.layers.Conv2DTranspose(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            strides=conv['strides'],
            padding="same",
        )(decoder_layers)
        decoder_layers = keras.layers.LeakyReLU()(decoder_layers)

    decoder_outputs = keras.layers.Conv2DTranspose(
        filters=img_channels,
        kernel_size=3,
        strides=1,
        padding="same",
    )(decoder_layers)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
