import json
import os.path
import shutil
import unittest
from os.path import isfile
from tempfile import TemporaryDirectory

from pandas import DataFrame

from models.loaders.metadata import export_metadata, load_metadata


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
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            with self.subTest('It loads the metadata from a fresh folder without metadata.json'):
                df = load_metadata(img_root_dir)
                self.assertEqual(type(df), DataFrame)
                self.assertEqual(len(df), 2)

            with self.subTest('It filters the metadata on inclusive tags'):
                df = load_metadata(img_root_dir, include_tags=['portrait'])
                self.assertEqual(len(df), 1)
                self.assertEqual(df.iloc[0].filename, 'set-1/blank-portrait.jpg')

            with self.subTest('The portrait-oriented img is 427 pix wide'):
                self.assertEqual(df.iloc[0].width, 427)
                self.assertEqual(df.iloc[0].height, 640)

            with self.subTest('It raises a ValueError when the same tag is both included and excluded'):
                with self.assertRaises(ValueError):
                    load_metadata(img_root_dir, include_tags=['portrait'], exclude_tags=['portrait'])

            with self.subTest('It excludes the portrait picture'):
                df = load_metadata(img_root_dir, exclude_tags=['portrait'])
                self.assertEqual(len(df), 1)
                self.assertEqual(df.iloc[0].filename, 'set-1/blank-landscape_640v640.jpg')
                self.assertSetEqual(df.iloc[0].tags, {'blank', 'landscape', 'set_1'})

            with self.subTest('It includes the folder as set tag'):
                df = load_metadata(img_root_dir)
                self.assertIn('set_1', df.iloc[0].tags)
