import logging
import os.path
from typing import Union

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard


def checkpoint_callback(checkpoint_folder: str) -> ModelCheckpoint:
    """
    Returns a ModelCheckpoint instance with some basic configuration

    :param checkpoint_folder: Folder to save checkpoints to

    :return: a ModelCheckpoint callback function
    """
    return ModelCheckpoint(
        filepath=checkpoint_folder,
        monitor='loss',
        save_freq='epoch',
        verbose=1,
    )


def tensorboard_callback(artifacts_folder: str, update_freq: Union[int, str] = 100) -> TensorBoard:
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
