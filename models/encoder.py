import keras
from keras import layers, Model

from models.loaders import Config
from models.sampling import Sampling


def get_encoder(config: Config) -> Model:
    """
    Constructs the encoder part of the variational auto-encoder

    :param config: a config_loader constructed Config dict

    :return: A Keras model instance as the encoder
    """

    width = config['images']['width']
    height = config['images']['height']
    channels = config['images']['channels']

    inputs = keras.Input(shape=(width, height, channels))

    convs = config['models']['vae']['conv2d']
    x = layers.Conv2D(filters=convs[0]['filters'], kernel_size=3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(filters=convs[1]['filters'], kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)

    latent_dim = 2
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder
