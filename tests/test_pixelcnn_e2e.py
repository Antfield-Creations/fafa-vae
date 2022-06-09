import unittest

import numpy as np

from models import train_pixelcnn
from models.loaders.config import load_config


class PixelCNNTestCase(unittest.TestCase):
    def test_model(self) -> None:
        config = load_config()
        pxl_conf = config['models']['pixelcnn']
        pxl_conf['batch_size'] = 1
        pxl_conf['batches_per_epoch'] = 1
        pxl_conf['epochs'] = 1

        history = train_pixelcnn.train(config)

        with self.subTest('The loss is a valid float'):
            assert history is not None
            last_epoch_loss = history.history.get('loss')[-1]
            self.assertFalse(np.isnan(last_epoch_loss))
