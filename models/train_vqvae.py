from typing import Optional, List

import numpy as np
from numpy import ndarray
from tensorflow import keras
from tensorflow.keras.callbacks import History  # type: ignore

from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders.callbacks import CustomImageSamplerCallback, CustomModelCheckpointSaver, tensorboard_callback
from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.loaders.script_archive import archive_scripts
from models.vqvae import VQVAETrainer


def train(config: Config) -> Optional[History]:
    """
    Trains the VAE model on the images

    :param config: a Config object from the load_config function

    :return: None
    """
    archive_scripts(config)

    # Compile the encoder and decoder separately to get rid of "uncompiled metrics" warnings
    # See also: https://stackoverflow.com/questions/67970389
    optimizer = keras.optimizers.Adam(learning_rate=config['models']['vqvae']['learning_rate'])
    encoder = get_encoder(config)
    encoder.compile(optimizer=optimizer)
    encoder.summary()

    decoder = get_decoder(config)
    decoder.compile(optimizer=optimizer)
    decoder.summary()

    data_generator = PaddingGenerator(config)

    variance_sample: List[ndarray] = []
    while len(variance_sample) < config['models']['vqvae']['data_generator']['fit_samples']:
        variance_sample.extend(next(data_generator))

    config['models']['vqvae']['train_variance'] = np.var(variance_sample)
    vqvae_trainer = VQVAETrainer(config)
    vqvae_trainer.compile(optimizer=optimizer)
    logs_folder = config['models']['vqvae']['artifacts']['logs']['folder']

    epochs = config['models']['vqvae']['epochs']
    steps = config['models']['vqvae']['batches_per_epoch']

    tensorboard_cb = tensorboard_callback(artifacts_folder=logs_folder)
    image_sampler = CustomImageSamplerCallback(config)
    checkpoint_saver = CustomModelCheckpointSaver(config)

    history = vqvae_trainer.fit(
        data_generator,
        verbose=1,
        epochs=epochs,
        use_multiprocessing=True,
        workers=config['models']['vqvae']['data_generator']['num_workers'],
        max_queue_size=config['models']['vqvae']['batch_size'] * 2,
        steps_per_epoch=steps,
        callbacks=[tensorboard_cb, image_sampler, checkpoint_saver],
    )

    return history
