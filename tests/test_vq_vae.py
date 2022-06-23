import unittest
from tempfile import TemporaryDirectory

import tensorflow as tf
from tensorflow import keras

from models.loaders.config import load_config
from models.loaders.data_provision import provision
from models.loaders.pixelcnn_data_generator import CodebookGenerator
from models.loaders.vae_data_generator import PaddingGenerator
from models.vq_vae import get_vq_vae, get_code_indices


class VQVAETestCase(unittest.TestCase):
    def test_fresh_model(self) -> None:
        config = load_config()
        config['models']['vq_vae']['artifacts']['resume_model'] = None
        vq_vae = get_vq_vae(config)
        self.assertIsInstance(vq_vae, keras.Model)

    def test_model_resume_loading(self) -> None:
        config = load_config()
        config['models']['vq_vae']['artifacts']['resume_model'] = 'gs://antfield/test/artifacts/checkpoints/vq_vae/'
        vq_vae = get_vq_vae(config)
        self.assertIsInstance(vq_vae, keras.Model)

    def test_get_code_indices(self) -> None:
        config = load_config()
        vq_vae = get_vq_vae(config)

        with TemporaryDirectory() as tempdir:
            config['models']['vq_vae']['artifacts']['folder'] = tempdir
            pxl_conf = config['models']['pixelcnn']
            img_conf = config['data']['images']

            # Load sample data
            img_conf['folder'] = tempdir
            img_conf['cloud_storage_folder'] = pxl_conf['image_test_folder']
            provision(config)

            # use the model for training session
            img_conf['filter']['exclude'] = []
            pxl_conf['artifacts']['reconstructions']['enabled'] = False
            pxl_conf['batch_size'] = 8

            data_generator = PaddingGenerator(config, model_name='pixelcnn')

            encoder = vq_vae.get_layer('encoder')
            quantizer = vq_vae.get_layer('vector_quantizer')
            batch = data_generator[0]

            with self.subTest('It fetches best-fit codes from the code book'):
                encoded = encoder(batch)
                flattened = tf.reshape(encoded, shape=(-1, encoded.shape[-1]))
                codes = get_code_indices(vector_quantizer=quantizer, flattened_inputs=flattened)
                self.assertEqual(len(codes), pxl_conf['batch_size'] * encoded.shape[1] * encoded.shape[2])

            with self.subTest('Which is fully shape-compatible with the codebook data generator'):
                codebook_generator = CodebookGenerator(config, encoder, quantizer)
                pixelcnn_codes = codebook_generator[0]
                batch_shaped = codes.numpy().reshape(encoded.shape[:-1])
                self.assertEqual(batch_shaped.shape, pixelcnn_codes.shape)
