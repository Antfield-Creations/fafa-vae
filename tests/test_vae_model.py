import logging
import os.path
import unittest
from os import listdir
from tempfile import TemporaryDirectory

import numpy as np

from models.loaders import load_config
from models.train import train
from scraper import scraper

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class VAEModelTestCase(unittest.TestCase):
    def test_training(self) -> None:
        with TemporaryDirectory() as tempdir:
            config = load_config()

            config['images']['folder'] = os.path.join(tempdir, 'img')
            set_no = 5159
            config['images']['scraper']['first_set'] = set_no
            config['images']['scraper']['last_set'] = set_no

            with self.subTest(f'It harvests set number {set_no}'):
                scraper.scrape(config)

            config['models']['vae']['checkpoints']['folder'] = os.path.join(tempdir, 'checkpoints')

            # Simplify training
            config['models']['vae']['data_generator']['fit_samples'] = 10

            num_epochs = 2
            config['models']['vae']['epochs'] = num_epochs

            batches_per_epoch = 16
            config['models']['vae']['batches_per_epoch'] = batches_per_epoch

            batch_size = 2
            config['models']['vae']['batch_size'] = batch_size

            # Dummy-train
            history = train(config)

            checkpoints_folder = str(config['models']['vae']['checkpoints']['folder'])
            epoch_1_folder = os.path.join(checkpoints_folder, 'epoch-1')

            with self.subTest('The loss is a valid float'):
                self.assertFalse(np.isnan(history.history.get('loss')))

            with self.subTest('It generates a checkpoint each epoch'):
                checkpoints = listdir(checkpoints_folder)
                num_checkpoints = num_epochs
                self.assertEqual(len(checkpoints), num_checkpoints)

            with self.subTest('It generates a folder for the decoder and encoder'):
                contents = listdir(epoch_1_folder)
                self.assertIn('encoder', contents)
                self.assertIn('decoder', contents)

            epoch_1_samples = listdir(os.path.join(epoch_1_folder, 'reconstructions'))
            for img_idx in range(batch_size):
                with self.subTest(f"It generates the image {img_idx + 1} sample"):
                    self.assertIn(f'{img_idx + 1}.png', epoch_1_samples)
