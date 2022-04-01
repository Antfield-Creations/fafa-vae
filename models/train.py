import os.path
import shutil
import time
from logging import getLogger
from typing import Optional

from tensorflow import keras
from tensorflow.python.keras.callbacks import History

from models.callbacks import tensorboard_callback
from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders.callbacks import CustomImageSamplerCallback
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
    optimizer = keras.optimizers.Adam(learning_rate=config['models']['vae']['learning_rate'])
    encoder.compile(optimizer=optimizer)
    encoder.summary()

    decoder = get_decoder(config)
    decoder.compile(optimizer=optimizer)
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizer)

    # Checkpoints, sample reconstructions and metric artifact folders
    run_id = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
    artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)
    checkpoint_folder = os.path.join(artifact_folder, 'checkpoints')

    # Copy model modules to artifacts for archiving
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(os.path.join(script_dir, 'models'), artifact_folder)

    epochs = config['models']['vae']['epochs']
    steps = config['models']['vae']['batches_per_epoch']

    data_generator = get_generator(config)
    tensorboard_cb = tensorboard_callback(artifacts_folder=artifact_folder)
    image_sampler = CustomImageSamplerCallback(config, run_id)
    history = None

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch} of {epochs} in {steps} steps of batch size {data_generator.batch_size}:")
        history = vae.fit(data_generator,
                          verbose=1,
                          initial_epoch=epoch - 1,
                          epochs=epoch,
                          use_multiprocessing=True,
                          steps_per_epoch=steps,
                          callbacks=[tensorboard_cb, image_sampler],
                          )

        # Save encoder and decoder models on interval
        if epoch % config['models']['vae']['checkpoints']['save_every_epoch'] == 0:
            epoch_folder = os.path.join(checkpoint_folder, f'epoch-{epoch}')
            vae.encoder.save(filepath=os.path.join(epoch_folder, 'encoder'))
            vae.decoder.save(filepath=os.path.join(epoch_folder, 'decoder'))

    return history
