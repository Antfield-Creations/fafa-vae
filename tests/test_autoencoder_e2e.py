import logging
import os.path
import unittest
import time
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
            config = load_config(run_id='dummy')

            # Override image location and filter settings
            config['images']['folder'] = os.path.join(tempdir, 'img')
            config['images']['filter']['include'] = None
            config['images']['filter']['exclude'] = []
            config['images']['filter']['orientation'] = 'any'

            set_no = 5159
            config['images']['scraper']['first_set'] = set_no
            config['images']['scraper']['last_set'] = set_no

            with self.subTest(f'It harvests set number {set_no}'):
                scraper.scrape(config)

            artifact_dir = os.path.join(tempdir, 'artifacts', config['run_id'])
            config['models']['vqvae']['artifacts']['folder'] = artifact_dir
            config['models']['vqvae']['artifacts']['logs']['folder'] = os.path.join(artifact_dir, 'tensorboard')

            # Simplify training
            config['models']['vqvae']['data_generator']['fit_samples'] = 10

            num_epochs = 2
            config['models']['vqvae']['epochs'] = num_epochs

            checkpoint_interval = num_epochs
            config['models']['vqvae']['artifacts']['checkpoints']['save_every_epoch'] = checkpoint_interval
            config['models']['vqvae']['artifacts']['reconstructions']['save_every_epoch'] = checkpoint_interval

            batches_per_epoch = 16
            config['models']['vqvae']['batches_per_epoch'] = batches_per_epoch

            batch_size = 2
            config['models']['vqvae']['batch_size'] = batch_size

            # Dummy-train
            history = train(config)

            epoch_2_folder = os.path.join(artifact_dir, 'checkpoints', 'epoch-2')

            with self.subTest('The loss is a valid float'):
                assert history is not None
                last_epoch_loss = history.history.get('loss')[-1]
                self.assertFalse(np.isnan(last_epoch_loss))

            with self.subTest('It generates a set of artifact directories'):
                artifacts = listdir(artifact_dir)
                self.assertSetEqual(
                    set(artifacts),
                    {'checkpoints', 'models', 'reconstructions', 'tensorboard'},
                    f"Got: {artifacts} from {artifact_dir}")

            with self.subTest(f'It generates a checkpoint dir for epoch intervals of {checkpoint_interval}'):
                epochs = listdir(os.path.join(artifact_dir, 'checkpoints'))
                self.assertSetEqual(set(epochs), {'epoch-2'})

            with self.subTest('It generates a folder for the decoder and encoder'):
                contents = listdir(epoch_2_folder)
                self.assertIn('vq_vae', contents)

            samples = listdir(os.path.join(artifact_dir, 'reconstructions'))
            for img_idx in range(batch_size):
                with self.subTest(f"It generates the image {img_idx + 1} sample"):
                    self.assertIn(f'epoch-{num_epochs}-{img_idx + 1}.png', samples)
