import keras
from keras import Model, layers

from models.loaders import Config


def get_decoder(config: Config) -> Model:
    """
    Constructs the decoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the decoder
    """

    latent_dim = config['models']['vae']['latent_dim']

    latent_inputs = keras.Input(shape=(latent_dim,))

    convs = config['models']['vae']['conv2d']
    x = layers.Dense(7 * 7 * convs[1]['filters'], activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, convs[1]['filters']))(x)

    # The conv2d layers in reverse:
    x = layers.Conv2DTranspose(
        filters=convs[1]['filters'], kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(
        filters=convs[0]['filters'], kernel_size=3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
