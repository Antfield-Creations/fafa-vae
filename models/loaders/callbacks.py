import os

from keras_preprocessing.image import save_img
from tensorflow import keras
from tensorflow.python.ops.gen_batch_ops import Batch

from models.loaders.config import Config
from models.loaders.data_generator import get_generator


class CustomImageSamplerCallback(keras.callbacks.Callback):
    """
    Saves image reconstructions sampled from the input dataset at the end of each epoch
    """
    def __init__(self, config: Config, run_id: str) -> None:
        self.data_generator = get_generator(config)
        self.run_id = run_id
        self.artifact_folder = os.path.join(config['models']['vae']['artifacts']['folder'], run_id)

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        sample_inputs = self.data_generator.next()
        reconstructions = self.model(sample_inputs)
        reconstructions_folder = os.path.join(self.artifact_folder, 'reconstructions')
        os.makedirs(reconstructions_folder, exist_ok=True)

        for img_idx in range(reconstructions.shape[0]):
            output_path = os.path.join(reconstructions_folder, f'epoch-{epoch + 1}-{img_idx + 1}.png')
            save_img(output_path, reconstructions[img_idx])

    def on_train_batch_end(self, batch: Batch, logs: dict = None) -> None:
        """
        Overrides the default on_train_batch_end method to basically a no-op.
        This fixes the message "Callback method `on_train_batch_end` is slow compared to the batch time"

        :param batch:
        :param logs:
        :return:
        """
        pass
