import logging
import os.path
import shutil
import unittest
from os import listdir
from tempfile import TemporaryDirectory

from models.loaders import load_config
from models.train import train

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class VAEModelTestCase(unittest.TestCase):
    def test_training(self) -> None:
        with TemporaryDirectory() as tempdir:
            shutil.copytree('tests/data', tempdir + '/img')

            config = load_config()
            config['images']['folder'] = os.path.join(tempdir, 'img')
            config['models']['vae']['checkpoints']['folder'] = os.path.join(tempdir, 'checkpoints')
            num_epochs = 2
            config['models']['vae']['epochs'] = num_epochs
            num_reconstructions_per_epoch = 4
            config['models']['vae']['predict_samples'] = num_reconstructions_per_epoch
            config['models']['vae']['data_generator']['fit_samples'] = 10

            train(config)

            checkpoints_folder = str(config['models']['vae']['checkpoints']['folder'])
            with self.subTest('It generates a checkpoint for the decoder and an encoder for each epoch'):
                checkpoints = listdir(checkpoints_folder)
                num_checkpoints = num_epochs
                self.assertEqual(len(checkpoints), num_checkpoints)

            with self.subTest(f'It generates {num_reconstructions_per_epoch} image samples per epoch'):
                epoch_1_samples = listdir(os.path.join(checkpoints_folder, 'epoch-1', 'reconstructions'))
                self.assertSetEqual(set(epoch_1_samples), {'1.png', '2.png', '3.png', '4.png'})
