import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

import keras.backend
import numpy as np

from models.loaders.config import load_config
from models.loaders.data_generator import padding_generator, scale
from models.loaders.metadata import export_metadata


class DataGeneratorTestCase(unittest.TestCase):
    def test_output_size(self) -> None:
        config = load_config()

        with TemporaryDirectory() as tempdir:
            img_dir = os.path.join(tempdir, 'set-1')
            shutil.copytree('tests/data', img_dir)
            export_metadata(tempdir)

            # Force-include everything
            config['images']['folder'] = tempdir
            config['images']['filter']['orientation'] = 'any'
            config['images']['filter']['include'] = None
            config['images']['filter']['exclude'] = []

            data_generator = padding_generator(config=config)

            batch = next(data_generator)
            img_cfg = config['images']
            self.assertEqual(batch[0].shape, (img_cfg['height'], img_cfg['width'], img_cfg['channels']))

    def test_scaling(self) -> None:
        test_data = np.zeros((1, 24, 24, 3))

        with self.subTest('It scales zeros to epsilons'):
            scaled = scale(test_data)
            self.assertEqual(scaled[0, 0, 0, 1], keras.backend.epsilon())

        with self.subTest('It scales 255 to 1 - epsilon'):
            test_data += 255
            scaled = scale(test_data)
            almost_1 = 1 - keras.backend.epsilon()
            self.assertAlmostEqual(scaled[0, 0, 0, 1], almost_1, places=8)
