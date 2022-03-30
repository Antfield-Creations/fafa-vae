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
    # No-op: extract out encoder layers so we can loop over them
    encoder_layers = keras.layers.Lambda(lambda x: x)(inputs)

    for conv in convs:
        encoder_layers = keras.layers.Conv2D(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            activation="relu",
            strides=2,
            padding="same"
        )(encoder_layers)

    encoder_layers = keras.layers.Flatten()(encoder_layers)
    encoder_layers = keras.layers.Dense(dense['size'], activation="relu")(encoder_layers)

    latent_dim = config['models']['vae']['latent_dim']
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoder_layers)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(encoder_layers)
    z = Sampling()((z_mean, z_log_var))

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder
