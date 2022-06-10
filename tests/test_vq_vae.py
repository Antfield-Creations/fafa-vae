import unittest

from tensorflow import keras

from models.loaders.config import load_config
from models.vq_vae import get_vq_vae


class VQVAETestCase(unittest.TestCase):
    def test_model_resume_loading(self) -> None:
        config = load_config()
        config['models']['vq_vae']['artifacts']['resume_model'] = 'gs://antfield/test/artifacts/checkpoints/vq_vae/'
        vq_vae = get_vq_vae(config)
        self.assertIsInstance(vq_vae, keras.Model)
