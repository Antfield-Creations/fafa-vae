import os.path
import shutil
import unittest
from os import listdir
from tempfile import TemporaryDirectory

from models.loaders import load_config
from models.train import train


class VAEModelTestCase(unittest.TestCase):
    def test_training(self) -> None:
        with TemporaryDirectory() as tempdir:
            shutil.copytree('tests/data', tempdir + '/img')

            config = load_config()
            config['images']['folder'] = os.path.join(tempdir, 'img')
            config['checkpoints']['folder'] = os.path.join(tempdir, 'checkpoints')

            with self.subTest('It generates a checkpoint for a single epoch'):
                train(config)
                checkpoints = listdir(str(config['checkpoints']['folder']))
                self.assertEqual(len(checkpoints), 1)
