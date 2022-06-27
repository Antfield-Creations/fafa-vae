import logging
import os.path
import random
from typing import Dict, List, Optional

import numpy as np
from PIL.Image import Image
from keras.utils.data_utils import Sequence
from keras_preprocessing.image import img_to_array, load_img
from math import floor, ceil
from numpy import ndarray
from tensorflow import keras

from models.loaders.config import Config
from models.loaders.metadata import load_metadata


class PaddingGenerator(Sequence):
    """
    Conventional Keras loaders use some kind of interpolation method to rescale images to a target size. This one,
    however, adds padding instead of stretching the image. In this way, the network does not need to learn stretched
    representations of the domain, instead it can learn the representations directly.

    Hopefully, this will not only help the network learn, but it also drops the requirement of re-scaling reconstructed
    images to their original orientation.
    """

    def __init__(self, config: Config, model_name: str = 'vq_vae'):
        """
        :param config:      The VAE config
        :param model_name:  Name of the model config to use, for the batch size

        :return: An infinite generator over batches of image tensors
        """

        self.img_folder = config['data']['images']['folder']
        self.img_cfg = config['data']['images']
        self.batch_size = config['models'][model_name]['batch_size']

        self.img_metadata = load_metadata(
            img_folder=self.img_folder,
            orientation=config['data']['images']['filter']['orientation'],
            include_tags=config['data']['images']['filter']['include'],
            exclude_tags=config['data']['images']['filter']['exclude'],
        )

        if len(self.img_metadata) <= 1:
            raise ValueError('Combination of orientation, include and exclude filters resulted in empty list')

        logging.info(f'Set contains {len(self.img_metadata)} images to train on.')
        # Drop last batch to prune to full batch size batches
        self.record_indices = list(range(len(self.img_metadata) - 1))
        random.shuffle(self.record_indices)

    def __len__(self) -> int:
        return floor(len(self.img_metadata)/self.batch_size)

    def __next__(self) -> ndarray:
        return self.__getitem__()

    def __getitem__(self, index: Optional[int] = None) -> ndarray:
        if index is None:
            start_index = random.choice(self.record_indices)
        else:
            start_index = index * self.batch_size

        img_data: List[ndarray] = []

        for idx in self.record_indices[start_index:start_index + self.batch_size]:
            src = dict(self.img_metadata.iloc[idx])
            src['img_folder'] = self.img_folder
            src['img_cfg'] = self.img_cfg
            img_data.append(load_image_data(src))

        batch = np.array(img_data)
        return batch

    def on_epoch_end(self) -> None:
        random.shuffle(self.record_indices)


def load_image_data(src: dict) -> ndarray:
    img = load_img(path=os.path.join(src['img_folder'], src['filename']))
    img_values = pad_image(img, src['img_cfg'])
    img_values = scale(img_values)
    return img_values


def scale(img_values: ndarray) -> ndarray:
    """
    Simplistic feature-wise normalization. Scales pixel int values from [0..255] to floats [0..1]
    This is helpful, because it allows you to activate the last vae decoder layer to sigmoid, and use
    binary cross-entropy as reconstruction loss, as the original VAE implementation did.

    :param img_values: batch of image data

    :return: The same image data, scaled down.

    """
    # Prevents 1. values: hard 1. floats cannot be reached by sigmoid activations
    img_values /= (255 + (255 * 2 * keras.backend.epsilon()))  # type: ignore
    # Prevent 0. values: hard 0. floats cannot be reached by sigmoid activations
    img_values += keras.backend.epsilon()

    return img_values


def pad_image(img: Image, img_cfg: Dict[str, int]) -> ndarray:
    """
    Converts an PIL Image instance to a zero-padded numpy ndarray

    :param img:     The Image instance
    :param img_cfg: A dictionary containing target 'height' and a 'width' keys, with each an integer value

    :return:
    """
    # img_to_array will result in an ndarray of size (height, width, channels)
    img_values = img_to_array(img)

    height_padding = img_cfg['height'] - img.height
    pad_top = floor(height_padding / 2)
    pad_bottom = ceil(height_padding / 2)
    width_padding = img_cfg['width'] - img.width
    pad_left = floor(width_padding / 2)
    pad_right = ceil(width_padding / 2)

    img_values = np.pad(
        array=img_values,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=(0, 0)
    )

    return img_values
