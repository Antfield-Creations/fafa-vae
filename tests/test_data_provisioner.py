import os
import unittest
from tempfile import TemporaryDirectory

from models.loaders.config import load_config
from models.loaders.data_provision import provision


class DataProvisionerTestCase(unittest.TestCase):
    def test_download(self) -> None:
        config = load_config()
        config['data']['images']['cloud_storage_folder'] = 'gs://antfield/test'

        with TemporaryDirectory() as tempdir:
            same_extension_as_storage_folder = os.path.join(tempdir, 'test')
            config['data']['images']['folder'] = same_extension_as_storage_folder
            provision(config)

            self.assertTrue(os.path.isfile(os.path.join(same_extension_as_storage_folder, 'test.txt')))
