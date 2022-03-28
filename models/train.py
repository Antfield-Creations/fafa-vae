import os.path
import time
from logging import getLogger
from typing import Optional

from keras_preprocessing.image import save_img
from tensorflow import keras
from tensorflow.python.keras.callbacks import History

from models.callbacks import tensorboard_callback
from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders.config import Config
from models.loaders.data_generator import get_generator
from models.vae import VAE

logger = getLogger('Train')


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

    # Checkpoints, sample reconstructions and metric artifact folders
    run_id = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
    artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)
    checkpoint_folder = os.path.join(artifact_folder, 'checkpoints')
    # Each epoch, the script generates a batch-sized set of sample images
    reconstructions_folder = os.path.join(artifact_folder, 'reconstructions')
    os.makedirs(reconstructions_folder, exist_ok=True)

    epochs = config['models']['vae']['epochs']
    steps = config['models']['vae']['batches_per_epoch']

    tensorboard_cb = tensorboard_callback(artifacts_folder=artifact_folder)
    data_generator = get_generator(config)
    history = None

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1} of {epochs} in {steps} steps of batch size {data_generator.batch_size}:")
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

        sample_inputs = data_generator.next()
        reconstructions = vae(sample_inputs)

        for img_idx in range(reconstructions.shape[0]):
            output_path = os.path.join(reconstructions_folder, f'epoch-{epoch + 1}-{img_idx + 1}.png')
            save_img(output_path, reconstructions[img_idx])

    return history
