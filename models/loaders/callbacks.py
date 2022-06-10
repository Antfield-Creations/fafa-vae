import logging
import os
import os.path
from typing import Union, Optional

import tensorflow as tf
from google.cloud.storage.blob import Blob  # noqa
from google.cloud.storage.bucket import Bucket  # noqa
from keras_preprocessing.image import save_img
from tempfile import TemporaryDirectory
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard  # noqa
from urllib.parse import urlparse

from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.loaders.script_archive import get_bucket


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
        self.artifact_folder = config['models']['vq_vae']['artifacts']['folder']
        self.reconstructions_folder = os.path.join(self.artifact_folder, 'reconstructions')
        self.bucket: Optional[Bucket] = None

        if self.reconstructions_folder.startswith('gs://') or self.reconstructions_folder.startswith('gcs://'):
            self.bucket = get_bucket(self.reconstructions_folder)
        else:
            os.makedirs(self.reconstructions_folder, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.epoch_interval == 0:
            sample_inputs = next(self.data_generator)
            reconstructions = self.model(sample_inputs)

            for img_idx in range(reconstructions.shape[0]):
                if self.bucket is not None:
                    self.save_reconstruction_bucket(reconstructions, epoch, img_idx)
                else:
                    self.save_reconstruction_local(reconstructions, epoch, img_idx)

    def save_reconstruction_local(self, reconstructions: tf.Tensor, epoch: int, img_idx: int) -> None:
        output_path = os.path.join(self.reconstructions_folder, f'epoch-{epoch + 1}-{img_idx + 1}.png')
        sample = reconstructions[img_idx]
        save_img(path=output_path, x=sample, scale=True)

    def save_reconstruction_bucket(self, reconstructions: tf.Tensor, epoch: int, img_idx: int) -> None:
        sample = reconstructions[img_idx]
        gs_url = urlparse(self.artifact_folder)
        bucket_subpath = gs_url.path.removeprefix('/')

        with TemporaryDirectory() as tempdir:
            filename = f'epoch-{epoch + 1}-{img_idx + 1}.png'
            temp_filename = os.path.join(tempdir, filename)
            save_img(path=temp_filename, x=sample, scale=True)
            blob = Blob(name=os.path.join(bucket_subpath, filename), bucket=self.bucket)
            blob.upload_from_filename(filename=temp_filename)


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
        Saves encoder and decoder sub-models on a given epoch interval

        :param epoch: the current finished epoch number
        :param logs: dictionary of epoch logging

        :return: None
        """
        if (epoch + 1) % self.epoch_interval == 0:
            epoch_folder = os.path.join(self.checkpoint_folder, f'epoch-{epoch + 1}')
            self.model.get_layer(self.model_name).save(filepath=os.path.join(epoch_folder, self.model_name))
