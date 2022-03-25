import logging
import os.path

from keras_preprocessing.image import save_img
from tensorflow import keras
from tqdm import tqdm

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
    # Compile the encoder separately to get rid of uncompiled metrics warnings
    # See also: https://stackoverflow.com/questions/67970389
    encoder.compile(optimizer=keras.optimizers.Adam())
    encoder.summary()

    decoder = get_decoder(config)
    decoder.compile(optimizer=keras.optimizers.Adam())
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    # Input data specifics
    fafa_loader = FAFADataGenerator()
    img_folder = config['images']['folder']
    img_metadata = load_metadata(
        img_folder=img_folder,
        include_tags=config['images']['filter']['include'],
        exclude_tags=config['images']['filter']['exclude'],
    )

    batch_size = config['models']['vae']['data_generator']['fit_samples']
    data_generator = fafa_loader.flow_from_dataframe(
        dataframe=img_metadata,
        class_mode=None,
        directory=img_folder,
        target_size=(config['images']['width'], config['images']['height']),
        batch_size=batch_size,
    )

    logging.info(f'Fitting {batch_size} samples for image loader normalization...')
    fafa_loader.fit(data_generator.next())

    # Reset data generator to training mode batch sizes
    data_generator.batch_size = config['models']['vae']['batch_size']

    logging.info(f'Image preprocessor featurewise: std {fafa_loader.std}')
    logging.info(f'Image preprocessor featurewise: mean {fafa_loader.mean}')

    # Checkpoints
    checkpoint_folder = config['models']['vae']['checkpoints']['folder']
    steps_per_epoch = config['models']['vae']['batches_per_epoch']

    for epoch in tqdm(range(config['models']['vae']['epochs'])):
        vae.fit(data_generator, verbose=1, initial_epoch=epoch + 1, steps_per_epoch=steps_per_epoch)

        # Save encoder and decoder models
        epoch_folder = os.path.join(checkpoint_folder, f'epoch-{epoch + 1}')

        vae.encoder.save(filepath=os.path.join(epoch_folder, 'encoder'))
        vae.decoder.save(filepath=os.path.join(epoch_folder, f'epoch-{epoch + 1}', 'decoder'))

        # Save samples
        reconstructions_folder = os.path.join(epoch_folder, 'reconstructions')
        os.makedirs(reconstructions_folder, exist_ok=True)

        for img_idx in tqdm(range(config['models']['vae']['predict_samples'])):
            batch = data_generator.next()
            encoded = vae.encoder.predict(batch, batch_size=batch.shape[0])
            latent_z = encoded[2]
            reconstructions = vae.decoder.predict(latent_z, batch_size=batch.shape[0])
            save_img(os.path.join(reconstructions_folder, f'{img_idx + 1}.png'), reconstructions[0])
