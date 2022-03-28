import itertools
import logging
import os.path
import unittest
from os import listdir
from tempfile import TemporaryDirectory

import numpy as np

from models.loaders.config import load_config
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

            config['models']['vae']['artifacts']['folder'] = os.path.join(tempdir, 'checkpoints')

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

            artifacts_folder = str(config['models']['vae']['artifacts']['folder'])
            runs = listdir(artifacts_folder)
            artifacts_folder = os.path.join(artifacts_folder, runs[0])
            epoch_1_folder = os.path.join(artifacts_folder, 'checkpoints', 'epoch-1')

            with self.subTest('The loss is a valid float'):
                assert history is not None
                self.assertFalse(np.isnan(history.history.get('loss')))

            with self.subTest('It generates a set of artifact directories'):
                artifacts = listdir(artifacts_folder)
                self.assertSetEqual(set(artifacts), {'checkpoints', 'reconstructions', 'tensorboard'},
                                    f"Got: {artifacts} from {artifacts_folder}")

            with self.subTest('It generates a checkpoint dir for each epoch'):
                epochs = listdir(os.path.join(artifacts_folder, 'checkpoints'))
                self.assertSetEqual(set(epochs), {'epoch-1', 'epoch-2'})

            with self.subTest('It generates a folder for the decoder and encoder'):
                contents = listdir(epoch_1_folder)
                self.assertIn('encoder', contents)
                self.assertIn('decoder', contents)

            samples = listdir(os.path.join(artifacts_folder, 'reconstructions'))
            for epoch, img_idx in itertools.product(range(num_epochs), range(batch_size)):
                with self.subTest(f"It generates the image {img_idx + 1} sample"):
                    self.assertIn(f'epoch-{epoch + 1}-{img_idx + 1}.png', samples)
