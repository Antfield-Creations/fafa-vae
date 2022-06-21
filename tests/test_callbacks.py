import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import tensorflow as tf
from google.cloud.storage.blob import Blob  # noqa

from models.loaders.callbacks import CustomImageSamplerCallback, PixelCNNReconstructionSaver
from models.loaders.config import load_config
from models.pixelcnn import get_pixelcnn


class CallbacksTestCase(unittest.TestCase):
    def test_vq_vae_reconstruction_save(self) -> None:
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

    def test_vq_vae_bucket_reconstruction_save(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            config = load_config(run_id='dummy', artifact_folder='gs://antfield/test/artifacts')
            config['data']['images']['folder'] = img_root_dir
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []
            config['data']['images']['filter']['orientation'] = 'any'

            reconstructor = CustomImageSamplerCallback(config)
            target = 'test/artifacts/reconstructions/epoch-1-1.png'
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

    def test_pixelcnn_reconstruction_sampler(self) -> None:
        with TemporaryDirectory() as tempdir:
            config = load_config(run_id='dummy', artifact_folder=tempdir)
            pxl_conf = config['models']['pixelcnn']
            pxl_conf['input_vq_vae'] = 'gs://antfield/test/small_output_vqvae/vq_vae'
            pxl_conf['artifacts']['reconstructions']['save_every_epoch'] = 1

            pixelcnn = get_pixelcnn(config)
            reconstructor = PixelCNNReconstructionSaver(config)
            # Normally, the model is passed to the callback object using history.fit()
            # but we'll inject it here directly to skip having to train
            setattr(reconstructor, 'model', pixelcnn)
            reconstructor.on_epoch_end(epoch=0, logs={})
