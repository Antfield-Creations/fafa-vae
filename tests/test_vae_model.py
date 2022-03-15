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
            config['models']['vae']['checkpoints']['folder'] = tempdir
            num_epochs = 2
            config['models']['vae']['epochs'] = num_epochs

            with self.subTest('It generates a checkpoint for a single epoch'):
                train(config)
                checkpoints = listdir(str(config['models']['vae']['checkpoints']['folder']))
                self.assertEqual(len(checkpoints), num_epochs)
