import logging
import os

from keras_preprocessing.image import save_img
import tensorflow as tf
from tensorflow import keras

from models.loaders.config import Config
from models.loaders.data_generator import get_generator


class CustomImageSamplerCallback(keras.callbacks.Callback):
    """
    Saves image reconstructions sampled from the input dataset at the end of each epoch
    """
    def __init__(self, config: Config, run_id: str) -> None:
        self.epoch_interval = config['models']['vae']['artifacts']['reconstructions']['save_every_epoch']
        self.data_generator = get_generator(config)
        self.run_id = run_id
        self.artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.epoch_interval == 0:
            sample_inputs = next(self.data_generator)
            reconstructions = self.model(sample_inputs)
            reconstructions_folder = os.path.join(self.artifact_folder, 'reconstructions')
            os.makedirs(reconstructions_folder, exist_ok=True)

            for img_idx in range(reconstructions.shape[0]):
                output_path = os.path.join(reconstructions_folder, f'epoch-{epoch + 1}-{img_idx + 1}.png')
                sample = reconstructions[img_idx]
                # Manual re-scale
                sample = tf.minimum(0, tf.maximum(255, sample) * 255)
                save_img(path=output_path, x=sample, scale=False)


class CustomModelCheckpointSaver(keras.callbacks.Callback):
    """
    Saves encoder and decoder checkpoints at a given epoch interval
    """

    def __init__(self, config: Config, run_id: str) -> None:
        self.epoch_interval = config['models']['vae']['artifacts']['checkpoints']['save_every_epoch']
        artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)
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
            self.model.encoder.save(filepath=os.path.join(epoch_folder, 'encoder'))
            self.model.decoder.save(filepath=os.path.join(epoch_folder, 'decoder'))
        else:
            logging.info(f'Skipping model checkpoint save for epoch {epoch + 1}')
