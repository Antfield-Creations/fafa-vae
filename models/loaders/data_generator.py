import os.path
import random
from math import floor, ceil
from typing import Generator, Dict, List

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


def get_generator(config: Config) -> ImageDataGenerator:
    """
    Constucts a Keras ImageDataGenerator to load data from. You can iterate over the data in the dataset, or get a
    batch using the data_generator.next() method

    :param config: Config dict containing tag filter settings and image folder

    :return: an iterator over numpy batches of the size of the image
    """
    fafa_loader = FAFADataGenerator()
    img_folder = config['images']['folder']

    img_metadata = load_metadata(
        img_folder=img_folder,
        orientation=config['images']['filter']['orientation'],
        include_tags=config['images']['filter']['include'],
        exclude_tags=config['images']['filter']['exclude'],
    )

    batch_size = config['models']['vae']['batch_size']
    data_generator = fafa_loader.flow_from_dataframe(
        dataframe=img_metadata,
        class_mode=None,
        target_size=(config['images']['height'], config['images']['width']),
        directory=img_folder,
        batch_size=batch_size,
    )

    return data_generator


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
            # Sample-wise normalize
            img_values -= np.mean(img_values, keepdims=True)
            img_values /= (np.std(img_values, keepdims=True) + 1e-6)
            img_data.append(img_values)

        batch = np.array(img_data)
        yield batch


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
