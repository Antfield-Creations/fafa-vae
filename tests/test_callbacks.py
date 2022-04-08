import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from models.loaders.callbacks import CustomImageSamplerCallback
from models.loaders.config import load_config


class CallbacksTestCase(unittest.TestCase):
    def test_reconstruction_save(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            config = load_config()
            config['models']['vae']['artifacts']['folder'] = tempdir
            config['images']['folder'] = img_root_dir

            reconstructor = CustomImageSamplerCallback(config, run_id='dummy')
            zeroes = np.zeros((1, 640, 640, 3))
            halves = zeroes + 0.5
            reconstructor.save_reconstruction(reconstructions=halves, epoch=0, img_idx=0)
            saved_img = Image.open(os.path.join(tempdir, 'dummy', 'reconstructions', 'epoch-1-1.png')).load()

            with self.subTest('It scales 0.5 floats by 255 to 128'):
                first_pixel = saved_img[0, 0]
                self.assertEqual(first_pixel, (127, 127, 127))  # add assertion here
