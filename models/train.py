import os.path
import shutil
import time
from logging import getLogger
from typing import Optional

from tensorflow import keras
from tensorflow.keras.callbacks import History

from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders.callbacks import CustomImageSamplerCallback, CustomModelCheckpointSaver, tensorboard_callback
from models.loaders.config import Config
from models.loaders.data_generator import padding_generator
from models.vae import VAE

logger = getLogger('Train')


def train(config: Config, run_id: str = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')) -> Optional[History]:
    """
    Trains the VAE model on the images

    :param config: a Config object from the load_config function
    :param run_id: The timestamp of the run. Allows setting to a custom value, mostly for testing

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
    artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)
    logs_folder = config['models']['vae']['artifacts']['logs']['folder'].format(run_id=run_id)
    models_folder = os.path.join(artifact_folder, 'models')

    # Copy model modules to artifacts for archiving
    models_src = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(models_src, models_folder)
    shutil.copy(os.path.join(models_src, '..', 'config.yaml'), os.path.join(models_folder, 'config.yaml'))

    epochs = config['models']['vae']['epochs']
    steps = config['models']['vae']['batches_per_epoch']

    data_generator = padding_generator(config)
    tensorboard_cb = tensorboard_callback(artifacts_folder=logs_folder)
    image_sampler = CustomImageSamplerCallback(config, run_id)
    checkpoint_saver = CustomModelCheckpointSaver(config, run_id)

    history = vae.fit(
        data_generator,
        verbose=1,
        epochs=epochs,
        use_multiprocessing=True,
        steps_per_epoch=steps,
        callbacks=[tensorboard_cb, image_sampler, checkpoint_saver],
    )

    return history
