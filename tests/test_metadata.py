import json
import os.path
import shutil
import unittest
from os.path import isfile
from tempfile import TemporaryDirectory

from pandas import DataFrame

from models.loaders import export_metadata, load_metadata


class MetadataTestCase(unittest.TestCase):
    def test_metadata_exporter(self) -> None:
        with TemporaryDirectory() as tempdir:
            shutil.copytree('tests/data', tempdir + '/data')
            export_metadata(tempdir)

            with self.subTest('It writes the sample files to a metadata json file'):
                expected_output_path = os.path.join(tempdir, 'metadata.json')
                self.assertTrue(isfile(expected_output_path))

            with self.subTest('It holds the metadata for the sample files'):
                with open(expected_output_path, 'rt') as f:
                    metadata = json.loads(f.read())

                self.assertEqual(len(metadata), 2)
                self.assertIn('filename', metadata[0].keys())
                self.assertIn('tags', metadata[0].keys())

    def test_metadata_loader(self) -> None:
        with TemporaryDirectory() as tempdir:
            shutil.copytree('tests/data', tempdir + '/data')

            with self.subTest('It loads the metadata from a fresh folder without metadata.json'):
                df = load_metadata(tempdir)
                self.assertEqual(type(df), DataFrame)
                self.assertEqual(len(df), 2)
