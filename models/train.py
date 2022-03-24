import logging
import os.path

import numpy as np
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

    data_generator = fafa_loader.flow_from_dataframe(
        dataframe=img_metadata,
        class_mode=None,
        directory=img_folder,
        target_size=(config['images']['width'], config['images']['height'])
    )

    for _ in range(100):
        fafa_loader.fit(data_generator.next())

    logging.info(f'Image preprocessor featurewise: std {fafa_loader.std}')
    logging.info(f'Image preprocessor featurewise: mean {fafa_loader.mean}')

    # Checkpoints
    checkpoint_folder = config['models']['vae']['checkpoints']['folder']

    for epoch in tqdm(range(config['models']['vae']['epochs'])):
        vae.fit(data_generator, initial_epoch=epoch + 1)

        # Save encoder and decoder models
        epoch_folder = os.path.join(checkpoint_folder, f'epoch-{epoch + 1}')

        vae.encoder.save(filepath=os.path.join(epoch_folder, 'encoder'))
        vae.decoder.save(filepath=os.path.join(epoch_folder, f'epoch-{epoch + 1}', 'decoder'))

        # Save samples
        reconstructions_folder = os.path.join(epoch_folder, 'reconstructions')
        os.makedirs(reconstructions_folder, exist_ok=True)

        for img_idx in tqdm(range(config['models']['vae']['predict_samples'])):
            generator_next = data_generator.next()[0]
            generator_next = np.expand_dims(generator_next, axis=0)
            samples = vae.decoder.predict(vae.encoder.predict(generator_next)[0])
            sample = samples[0]

            denormalized = np.multiply(sample, fafa_loader.std)
            denormalized = np.sum((denormalized, fafa_loader.mean))
            save_img(os.path.join(reconstructions_folder, f'{img_idx + 1}.png'), denormalized)
