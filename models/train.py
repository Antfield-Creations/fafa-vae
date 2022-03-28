import logging
import os.path
from typing import Optional

from keras_preprocessing.image import save_img
from tensorflow import keras
from tensorflow.python.keras.callbacks import History

from models.callbacks import tensorboard_callback
from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders import Config, FAFADataGenerator, load_metadata
from models.vae import VAE


def train(config: Config) -> Optional[History]:
    """
    Trains the VAE model on the images

    :param config: a Config object from the load_config function

    :return: None
    """

    encoder = get_encoder(config)
    # Compile the encoder separately to get rid of "uncompiled metrics" warnings
    # See also: https://stackoverflow.com/questions/67970389
    encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=config['models']['vae']['learning_rate']))
    encoder.summary()

    decoder = get_decoder(config)
    decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=config['models']['vae']['learning_rate']))
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=config['models']['vae']['learning_rate']))

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
    artifact_folder = config['models']['vae']['artifacts']['folder']
    checkpoint_folder = os.path.join(artifact_folder, 'checkpoints')

    epochs = config['models']['vae']['epochs']
    steps = config['models']['vae']['batches_per_epoch']

    history = None

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1} of {epochs} in {steps} steps of batch size {data_generator.batch_size}:")
        tensorboard_cb = tensorboard_callback(artifacts_folder=checkpoint_folder)
        history = vae.fit(data_generator,
                          verbose=1,
                          initial_epoch=epoch,
                          epochs=epoch + 1,
                          use_multiprocessing=True,
                          steps_per_epoch=steps,
                          callbacks=[tensorboard_cb],
                          )

        # Save encoder and decoder models
        epoch_folder = os.path.join(checkpoint_folder, f'epoch-{epoch + 1}')

        vae.encoder.save(filepath=os.path.join(epoch_folder, 'encoder'))
        vae.decoder.save(filepath=os.path.join(epoch_folder, 'decoder'))

        # Each epoch, the script generates a batch-sized set of sample images
        reconstructions_folder = os.path.join(artifact_folder, 'reconstructions')
        os.makedirs(reconstructions_folder, exist_ok=True)

        sample_inputs = data_generator.next()
        reconstructions = vae(sample_inputs)

        for img_idx in range(reconstructions.shape[0]):
            save_img(os.path.join(reconstructions_folder, f'epoch-{epoch}-{img_idx + 1}.png'), reconstructions[img_idx])

    return history
