import os.path
import unittest
from os.path import isdir
from tempfile import TemporaryDirectory

from models.loaders import load_config
from scraper import scraper


class ScraperTestCase(unittest.TestCase):
    def test_scrape_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            set_no = 1337

            config = load_config()
            config['images']['folder'] = temp_dir
            config['images']['scraper']['first_set'] = set_no
            config['images']['scraper']['last_set'] = set_no

            with self.subTest(f'It harvests set number {set_no}'):

                # Execute the "main" function
                scraper.scrape(config)

                test_set_dir = os.path.join(temp_dir, f'set-{set_no}')
                self.assertTrue(isdir(test_set_dir))

                dir_contents = os.listdir(test_set_dir)
                self.assertEqual(len(dir_contents), 24)
