import os.path

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

    # Input data specifics
    fafa_loader = FAFADataGenerator()
    img_folder = config['images']['folder']
    metadataframe = load_metadata(
        img_folder=img_folder,
        include_tags=config['images']['filter']['include'],
        exclude_tags=config['images']['filter']['exclude'],
    )

    data_generator = fafa_loader.flow_from_dataframe(
        dataframe=metadataframe,
        class_mode=None,
        directory=img_folder,
        target_size=(config['images']['width'], config['images']['height'])
    )

    # Checkpoints
    checkpoint_folder = config['models']['vae']['checkpoints']['folder']

    for epoch in range(config['models']['vae']['epochs']):
        vae.fit(data_generator, initial_epoch=epoch + 1)
        vae.encoder.save(filepath=os.path.join(checkpoint_folder, f'encoder-epoch-{epoch + 1}'))
        vae.decoder.save(filepath=os.path.join(checkpoint_folder, f'decoder-epoch-{epoch + 1}'))
