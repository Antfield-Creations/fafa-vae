import os

import tensorflow as tf
from google.cloud.storage.blob import Blob  # noqa
from keras_preprocessing.image import save_img
from urllib.parse import urlparse

from tempfile import TemporaryDirectory

from models.loaders.script_archive import get_bucket


def save_reconstructions(
        reconstructions_folder: str,
        reconstructions: tf.Tensor,
        epoch: int) -> None:
    """
    Saves the n-th reconstruction (specified by `img_idx`) as PNG from image batch tensor `reconstructions` to
    `reconstructions_folder` with automatically filled-in file names for `epoch` and `img_idx`

    :param reconstructions_folder:  Folder to save to: either a local path or a bucket path
    :param reconstructions:         Tensor of at least rank 4: (batch, rows, cols, channels)
    :param epoch:                   The epoch number for which the image samples being saved

    :return: None
    """
    for img_idx in range(reconstructions.shape[0]):
        sample = reconstructions[img_idx]

        if reconstructions_folder.startswith('gs://') or reconstructions_folder.startswith('gcs://'):
            # Save to bucket
            bucket = get_bucket(reconstructions_folder)
            bucket_subpath = urlparse(reconstructions_folder).path.removeprefix('/')
            bucket_subpath = bucket_subpath.removesuffix('/')

            with TemporaryDirectory() as tempdir:
                # Save to temporary file first
                filename = f'epoch-{epoch + 1}-{img_idx + 1}.png'
                temp_filename = os.path.join(tempdir, filename)
                save_img(path=temp_filename, x=sample, scale=True)

                # Then upload from temp file
                blob = Blob(name=f'{bucket_subpath}/{filename}', bucket=bucket)
                blob.upload_from_filename(filename=temp_filename)
        else:
            # Save to local folder
            os.makedirs(reconstructions_folder, exist_ok=True)
            output_path = os.path.join(
                reconstructions_folder, f'epoch-{epoch + 1}-{img_idx + 1}.png')
            sample = reconstructions[img_idx]
            save_img(path=output_path, x=sample, scale=True)
