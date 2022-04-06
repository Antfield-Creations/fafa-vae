import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

from models.loaders.config import load_config
from models.loaders.data_generator import padding_generator
from models.loaders.metadata import export_metadata


class DataGeneratorTestCase(unittest.TestCase):
    def test_output_size(self) -> None:
        config = load_config()

        with TemporaryDirectory() as tempdir:
            img_dir = os.path.join(tempdir, 'img')
            shutil.copytree('tests/data', img_dir)
            export_metadata(tempdir)
            config['images']['folder'] = img_dir

            data_generator = padding_generator(config=config)

            batch = next(data_generator)
            img_cfg = config['images']
            self.assertEqual(batch[0].shape, (img_cfg['height'], img_cfg['width'], img_cfg['channels']))