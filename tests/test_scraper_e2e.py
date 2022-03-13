import os.path
import unittest
from argparse import Namespace
from os.path import isdir
from tempfile import TemporaryDirectory

from scraper import main


class ScraperTestCase(unittest.TestCase):
    def test_scrape_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            set_no = 1337

            with self.subTest(f'It harvests set number {set_no}'):
                args = Namespace(
                    first_set=set_no,
                    last_set=set_no,
                    directory=temp_dir
                )

                # Execute the "main" function
                main.main(args=args)

                test_set_dir = os.path.join(temp_dir, f'set-{set_no}')
                self.assertTrue(isdir(test_set_dir))

                dir_contents = os.listdir(test_set_dir)
                self.assertEqual(len(dir_contents), 24)


if __name__ == '__main__':
    unittest.main()
