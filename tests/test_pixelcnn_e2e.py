import os.path
import unittest

import numpy as np
from tempfile import TemporaryDirectory

from models import train_pixelcnn
from models.loaders.config import load_config
from models.loaders.data_provision import provision


class PixelCNNTestCase(unittest.TestCase):
    def test_model(self) -> None:
        with TemporaryDirectory() as tempdir:
            config = load_config(run_id='dummy', artifact_folder=tempdir)
            config['models']['vq_vae']['artifacts']['logs']['folder'] = tempdir
            img_conf = config['data']['images']

            pxl_conf = config['models']['pixelcnn']
            pxl_conf['batch_size'] = pxl_conf['batches_per_epoch'] = pxl_conf['epochs'] = 1

            pxl_conf['artifacts']['folder'] = tempdir

            # Load sample data
            img_conf['folder'] = tempdir
            img_conf['cloud_storage_folder'] = pxl_conf['image_test_folder']
            provision(config)

            # use the model for training session
            img_conf['filter']['exclude'] = []
            pxl_conf['artifacts']['reconstructions']['enabled'] = False
            pxl_conf['artifacts']['folder'] = tempdir
            pxl_conf['artifacts']['checkpoints']['save_every_epoch'] = 1
            history = train_pixelcnn.train(config)

            with self.subTest('The loss is a valid float'):
                assert history is not None
                last_epoch_loss = history.history.get('loss')[-1]
                self.assertFalse(np.isnan(last_epoch_loss))

            with self.subTest('It stores a saved model'):
                checkpoint_location = os.path.join(tempdir, 'checkpoints', 'epoch-1', 'pixelcnn')
                self.assertTrue(
                    os.path.isdir(checkpoint_location)
                )

            with self.subTest('It is able to resume training using a saved model'):
                pxl_conf['artifacts']['resume_model'] = checkpoint_location
                train_pixelcnn.train(config)
