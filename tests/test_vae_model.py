import os.path
import unittest
from os import listdir
from tempfile import TemporaryDirectory

from tqdm import tqdm

from models.loaders import load_config
from models.train import train
from scraper.scraper import harvest_set


class VAEModelTestCase(unittest.TestCase):
    def test_training(self) -> None:
        with TemporaryDirectory() as tempdir:
            config = load_config()
            config['images']['folder'] = os.path.join(tempdir, 'img')
            config['checkpoints']['folder'] = os.path.join(tempdir, 'checkpoints')

            pbar = tqdm()
            harvest_set(pbar, 5159, str(config['images']['folder']))

            with self.subTest('It generates a checkpoint for a single epoch'):
                train(config)
                checkpoints = listdir(str(config['checkpoints']['folder']))
                self.assertEqual(len(checkpoints), 1)
