import os.path
import random
from math import floor, ceil
from typing import Generator, Dict, List

from tensorflow import keras
import numpy as np
from PIL.Image import Image
from numpy import ndarray
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

from models.loaders.config import Config
from models.loaders.metadata import load_metadata


class FAFADataGenerator(ImageDataGenerator):
    def __init__(self) -> None:
        super(FAFADataGenerator, self).__init__(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=0,
            validation_split=0.2,
        )


def padding_generator(config: Config) -> Generator:
    """
    Conventional Keras loaders use some kind of interpolation method to rescale images to a target size. This one,
    however, adds padding instead of stretching the image. In this way, the network does not need to learn stretched
    representations of the domain, instead it can learn the representations directly.

    Hopefully, this will not only help the network learn, but it also drops the requirement of re-scaling reconstructed
    images to their original, unstretched size.

    :param config: The VAE config

    :return: An infinite generator over batches of image tensors
    """

    img_folder = config['images']['folder']
    img_cfg = config['images']
    batch_size = config['models']['vae']['batch_size']

    img_metadata = load_metadata(
        img_folder=img_folder,
        orientation=config['images']['filter']['orientation'],
        include_tags=config['images']['filter']['include'],
        exclude_tags=config['images']['filter']['exclude'],
    )

    if len(img_metadata) == 0:
        raise ValueError('Combination of orientation, include and exclude filters resulted in empty list')

    record_indices: List[int] = []
    while True:
        # Keep adding items to the record indices until we have a large enough list to sample a batch
        while len(record_indices) < batch_size:
            record_indices.extend(list(range(len(img_metadata))))

        random.shuffle(record_indices)

        batch_indices = [record_indices.pop() for _ in range(batch_size)]
        batch_meta = img_metadata.iloc[batch_indices]

        img_data = []
        for record in batch_meta.itertuples():
            img = load_img(path=os.path.join(img_folder, record.filename))
            # img_to_array will result in an ndarray of size (height, width, channels)
            img_values = pad_image(img, img_cfg)
            img_values = scale(img_values)
            img_data.append(img_values)

        batch = np.array(img_data)
        yield batch


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
