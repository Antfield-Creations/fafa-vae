from keras.optimizer_v2.adam import Adam

from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders import Config, FAFADataGenerator, load_metadata
from models.vae import VAE


def train(config: Config) -> None:
    """
    Trains the VAE model on the images

    :param config: a Config object from the load_config function

    :return: None
    """

    encoder = get_encoder(config)
    encoder.summary()

    decoder = get_decoder(config)
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam())

    fafa_loader = FAFADataGenerator()
    metadataframe = load_metadata(config['images']['folder'])

    for epoch in range(config['models']['vae']['epochs']):
        vae.fit(fafa_loader.flow_from_dataframe(dataframe=metadataframe))
