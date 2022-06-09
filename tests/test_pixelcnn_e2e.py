import unittest

import numpy as np
from tempfile import TemporaryDirectory

from models import train_pixelcnn
from models.loaders.config import load_config
from models.loaders.data_provision import provision


class PixelCNNTestCase(unittest.TestCase):
    def test_model(self) -> None:
        config = load_config()

        pxl_conf = config['models']['pixelcnn']
        img_conf = config['data']['images']

        pxl_conf['batch_size'] = 1
        pxl_conf['batches_per_epoch'] = 1
        pxl_conf['epochs'] = 1

        with TemporaryDirectory() as tempdir:
            # Load sample data
            img_conf['folder'] = tempdir
            img_conf['cloud_storage_folder'] = pxl_conf['image_test_folder']
            provision(config)

            # use the model for training session
            img_conf['filter']['exclude'] = []
            history = train_pixelcnn.train(config)

            with self.subTest('The loss is a valid float'):
                assert history is not None
                last_epoch_loss = history.history.get('loss')[-1]
                self.assertFalse(np.isnan(last_epoch_loss))
