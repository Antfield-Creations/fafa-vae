import logging
import os
import os.path
from typing import Union

import numpy as np
from google.cloud.storage.blob import Blob  # noqa
from google.cloud.storage.bucket import Bucket  # noqa
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard  # noqa

from models.loaders.config import Config
from models.loaders.vae_data_generator import PaddingGenerator
from models.loaders.image_saver import save_reconstructions
from models.pixelcnn import get_pixelcnn_sampler


def tensorboard_callback(artifacts_folder: str, update_freq: Union[int, str] = 'epoch') -> TensorBoard:
    """
    Returns a Tensorboard logging callback instance

    :param artifacts_folder:    main folder to save tensorboard logs to. Saves in a subfolder `tensorboard`
    :param update_freq:         Frequency to write logs

    :return: a Tensorboard callback function
    """
    tb_folder = os.path.join(artifacts_folder, 'tensorboard')
    logging.info(f'You may inspect the logs using\n\n tensorboard --logdir={tb_folder}')

    return TensorBoard(
        log_dir=tb_folder,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq=update_freq,
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )


class CustomImageSamplerCallback(keras.callbacks.Callback):
    """
    Saves image reconstructions sampled from the input dataset at the end of each epoch
    """
    def __init__(self, config: Config) -> None:
        self.epoch_interval = config['models']['vq_vae']['artifacts']['reconstructions']['save_every_epoch']
        self.data_generator = PaddingGenerator(config)
        self.run_id = config['run_id']
        self.artifact_folder = str(config['models']['vq_vae']['artifacts']['folder'])
        self.reconstructions_folder:  str = ''

        if self.artifact_folder.startswith('gs://') or self.artifact_folder.startswith('gcs://'):
            self.reconstructions_folder = self.artifact_folder.removesuffix('/') + '/reconstructions/'
        else:
            self.reconstructions_folder = os.path.join(self.artifact_folder, 'reconstructions')

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.epoch_interval != 0:
            return

        sample_inputs = next(self.data_generator)
        reconstructions = self.model(sample_inputs)
        save_reconstructions(self.reconstructions_folder, reconstructions, epoch)


class CustomModelCheckpointSaver(keras.callbacks.Callback):
    """
    Saves encoder and decoder checkpoints at a given epoch interval
    """

    def __init__(self, config: Config, model_name: str) -> None:
        self.model_name = model_name
        self.epoch_interval = config['models'][model_name]['artifacts']['checkpoints']['save_every_epoch']
        artifact_folder = config['models'][model_name]['artifacts']['folder']
        self.checkpoint_folder = os.path.join(artifact_folder, 'checkpoints')

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Saves (sub)-models on a given epoch interval.

        :param epoch: the current finished epoch number
        :param logs: dictionary of epoch logging

        :return: None
        """
        if (epoch + 1) % self.epoch_interval != 0:
            return

        epoch_folder = os.path.join(self.checkpoint_folder, f'epoch-{epoch + 1}')
        self.model.get_layer(self.model_name).save(filepath=os.path.join(epoch_folder, self.model_name))


class PixelCNNReconstructionSaver(keras.callbacks.Callback):
    def __init__(self, config: Config, decoder: keras.Model):
        self.pxl_conf = config['models']['pixelcnn']
        self.epoch_interval = self.pxl_conf['artifacts']['reconstructions']['save_every_epoch']

        artifact_folder = self.pxl_conf['artifacts']['folder']
        self.reconstructions_folder = os.path.join(artifact_folder, 'reconstructions')

        self.decoder = decoder

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Creates n reconstructions, where n is the batch size.

        :param epoch:   The current batch size
        :param logs:    The epoch details, unused but kept for keeping the superclass method signature

        :return: None
        """
        if (epoch + 1) % self.epoch_interval != 0:
            return

        mini_sampler = get_pixelcnn_sampler(self.model)
        priors = np.zeros(shape=(self.pxl_conf['batch_size'],) + self.model.input_shape[1:])
        batch, rows, cols = priors.shape

        # Iterate over the priors because generation has to be done sequentially pixel by pixel.
        for row in range(rows):
            for col in range(cols):
                # Feed the whole array and retrieving the pixel value probabilities for the next
                # pixel.
                probs = mini_sampler.predict(priors, verbose=0)
                # Use the probabilities to pick pixel values and append the values to the priors.
                priors[:, row, col] = probs[:, row, col]

            logging.info(f'Generated row {row + 1} of {rows}')

        images = self.decoder(priors)
        save_reconstructions(self.reconstructions_folder, images, epoch)
