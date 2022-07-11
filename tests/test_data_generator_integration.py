import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

import keras.backend
import numpy as np

from models.loaders.config import load_config
from models.loaders.pixelcnn_data_generator import CodebookGenerator
from models.loaders.vae_data_generator import PaddingGenerator, scale
from models.loaders.metadata import export_metadata
from models.pixelcnn import get_pixelcnn
from models.vq_vae import get_vq_vae


class DataGeneratorTestCase(unittest.TestCase):
    def test_padding_generator(self) -> None:
        config = load_config()

        with TemporaryDirectory() as tempdir:
            img_dir = os.path.join(tempdir, 'set-1')
            shutil.copytree('tests/data', img_dir)
            export_metadata(tempdir)

            # Force-include everything
            config['data']['images']['folder'] = tempdir
            config['data']['images']['filter']['orientation'] = 'any'
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []

            data_generator = PaddingGenerator(config=config)

            batch = next(data_generator)
            img_cfg = config['data']['images']
            self.assertEqual(batch[0].shape, (img_cfg['height'], img_cfg['width'], img_cfg['channels']))

    def test_padding_generator_scaling(self) -> None:
        test_data = np.zeros((1, 24, 24, 3))

        with self.subTest('It scales zeros to epsilons'):
            scaled = scale(test_data)
            self.assertEqual(scaled[0, 0, 0, 1], keras.backend.epsilon())

        with self.subTest('It scales 255 to 1 - epsilon'):
            test_data += 255
            scaled = scale(test_data)
            almost_1 = 1 - keras.backend.epsilon()
            self.assertAlmostEqual(scaled[0, 0, 0, 1], almost_1, places=8)

    def test_pixelcnn_data_generator(self) -> None:
        with TemporaryDirectory() as tempdir:
            # Use temporary directory to prevent polluting the bucket with test run data
            config = load_config(artifact_folder=tempdir)
            config['models']['vq_vae']['artifacts']['logs']['folder'] = tempdir
            img_dir = os.path.join(tempdir, 'set-1')
            shutil.copytree('tests/data', img_dir)
            export_metadata(tempdir)

            # Force-include all images
            config['data']['images']['folder'] = tempdir
            config['data']['images']['filter']['orientation'] = 'any'
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []

            pxl_conf = config['models']['pixelcnn']
            pxl_conf['batch_size'] = pxl_conf['batches_per_epoch'] = pxl_conf['epochs'] = 1

            vq_vae = get_vq_vae(config)
            encoder = vq_vae.get_layer('encoder')
            quantizer = vq_vae.get_layer('vector_quantizer')

            with self.subTest('It generates input data of shape (batch, encoder_cols, encoder_rows)'):
                data_generator = CodebookGenerator(config, encoder, quantizer)
                inputs, targets = data_generator[0]
                expected_batch_shape = (pxl_conf['batch_size'], encoder.output.shape[1], encoder.output.shape[2], 1)
                self.assertEqual(inputs.shape, expected_batch_shape)

            with self.subTest('It generates PixelCNN input data for differing encoder output and embedding shapes'):
                vq_vae_conf = config['models']['vq_vae']
                last_conv_layer = vq_vae_conf['conv2d'][-1]
                last_conv_layer['filters'] = vq_vae_conf['latent_size'] * 2

                vq_vae = get_vq_vae(config)
                encoder = vq_vae.get_layer('encoder')
                quantizer = vq_vae.get_layer('vector_quantizer')
                data_generator = CodebookGenerator(config, encoder, quantizer)
                inputs, targets = data_generator[0]

                expected_batch_shape = (pxl_conf['batch_size'], encoder.output.shape[1], encoder.output.shape[2], 2)
                self.assertEqual(inputs.shape, expected_batch_shape)
