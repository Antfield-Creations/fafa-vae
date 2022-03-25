from tensorflow import keras

from models.loaders import Config


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

    x = keras.layers.Dense(
        int(img_width / 4) * int(img_height / 4) * convs[1]['filters'], activation="relu")(latent_inputs)
    x = keras.layers.Reshape(
        (int(img_width / 4), int(img_height / 4), convs[1]['filters']))(x)

    # The conv2d layers in reverse:
    x = keras.layers.Conv2DTranspose(
        filters=convs[1]['filters'],
        kernel_size=convs[1]['kernel_size'],
        activation="relu",
        strides=2,
        padding="same"
    )(x)
    x = keras.layers.Conv2DTranspose(
        filters=convs[0]['filters'],
        kernel_size=convs[0]['kernel_size'],
        activation="relu",
        strides=2,
        padding="same"
    )(x)
    decoder_outputs = keras.layers.Conv2DTranspose(img_channels, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder
