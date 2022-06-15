import os.path
import unittest
from tempfile import TemporaryDirectory

from models.loaders.config import load_config


class ConfigLoaderTestCase(unittest.TestCase):
    def test_config_loader(self) -> None:
        with self.subTest('It makes a local artifact subfolder on a local artifact path'):
            with TemporaryDirectory() as tempdir:
                local_artifact_folder = os.path.join(tempdir, 'artifacts')
                load_config(artifact_folder=local_artifact_folder)
                self.assertTrue(os.path.isdir(local_artifact_folder))

        with self.subTest('It skips making a local artifact subfolder on a bucket artifact path'):
            with TemporaryDirectory() as tempdir:
                remote_artifact_folder = 'gs://something/something'
                load_config(artifact_folder=remote_artifact_folder)
                self.assertFalse(os.path.isdir('gs:'))

        with self.subTest('It loads a pre-defined run id'):
            config = load_config(run_id='test')
            self.assertEqual(config['run_id'], 'test')

        with self.subTest('It fails loading a config file with homedir tildes still in it'):
            with self.assertRaises(ValueError):
                load_config(run_id='~')
