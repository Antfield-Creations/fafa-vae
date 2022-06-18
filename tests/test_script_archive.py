import unittest

from models.loaders.script_archive import get_bucket


class ScriptArchiveTestCase(unittest.TestCase):
    def test_get_bucket(self) -> None:
        checkpoint_url = 'gs://antfield/FAFA/artifacts/2022-06-15_09h09m47s/checkpoints/epoch-128/vq_vae'
        bucket = get_bucket(checkpoint_url)
        self.assertEqual(bucket.name, 'antfield')
