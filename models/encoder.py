import tensorflow as tf
from tensorflow import keras

from models.loaders.config import Config
from models.sampling import Sampling


def get_encoder(config: Config) -> keras.Model:
    """
    Constructs the encoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the encoder
    """

    width = config['images']['width']
    height = config['images']['height']
    channels = config['images']['channels']

    convs = config['models']['vae']['conv2d']
    dense = config['models']['vae']['dense']

    inputs = keras.Input(shape=(width, height, channels))

    x = keras.layers.Conv2D(
        filters=convs[0]['filters'],
        kernel_size=convs[0]['kernel_size'],
        activation="relu",
        strides=2,
        padding="same"
    )(inputs)
    x = keras.layers.Conv2D(
        filters=convs[1]['filters'],
        kernel_size=convs[1]['kernel_size'],
        activation="relu",
        strides=2,
        padding="same"
    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(dense['size'], activation="relu")(x)

    latent_dim = config['models']['vae']['latent_dim']
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder
