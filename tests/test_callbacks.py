import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import tensorflow as tf
from google.cloud.storage.blob import Blob  # noqa

from models.loaders.callbacks import CustomImageSamplerCallback
from models.loaders.config import load_config


class CallbacksTestCase(unittest.TestCase):
    def test_reconstruction_save(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            config = load_config(run_id='dummy', artifact_folder=tempdir)
            artifacts_folder = config['models']['vq_vae']['artifacts']['folder']
            config['data']['images']['folder'] = img_root_dir
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []
            config['data']['images']['filter']['orientation'] = 'any'

            reconstructor = CustomImageSamplerCallback(config)
            zeroes = np.zeros((1, 640, 640, 3))
            halves = zeroes + 0.5
            reconstructor.save_reconstruction_local(reconstructions=halves, epoch=0, img_idx=0)
            saved_img = Image.open(os.path.join(artifacts_folder, 'reconstructions', 'epoch-1-1.png')).load()

            expected_values = (0, 0, 0)
            with self.subTest(f'It scales 0.5 floats by 255 to {expected_values}'):
                first_pixel = saved_img[0, 0]  # type: ignore
                self.assertEqual(first_pixel, expected_values)  # add assertion here

    def test_bucket_reconstruction_save(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            config = load_config(run_id='dummy', artifact_folder='gs://antfield/test/artifacts')
            config['data']['images']['folder'] = img_root_dir
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []
            config['data']['images']['filter']['orientation'] = 'any'

            reconstructor = CustomImageSamplerCallback(config)
            target = 'test/artifacts/epoch-1-1.png'
            blob = Blob(name=target, bucket=reconstructor.bucket)

            with self.subTest('It initializes the bucket because we passed a bucket url'):
                self.assertIsNotNone(reconstructor.bucket)

            with self.subTest('It does not have the target reconstruction blob yet'):
                self.assertFalse(blob.exists())

            with self.subTest('It saves the image'):
                reconstructor.save_reconstruction_bucket(
                    reconstructions=tf.zeros((1, 640, 640, 3)),
                    epoch=0,
                    img_idx=0)
                self.assertTrue(blob.exists())
                assert reconstructor.bucket is not None
                reconstructor.bucket.delete_blob(target)
