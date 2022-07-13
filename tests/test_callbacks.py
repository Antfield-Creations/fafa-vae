import os.path
import shutil
import unittest
from tempfile import TemporaryDirectory

from PIL import Image
from google.cloud.storage.blob import Blob  # noqa

from models.loaders.callbacks import CustomImageSamplerCallback, PixelCNNReconstructionSaver
from models.loaders.config import load_config
from models.loaders.script_archive import get_bucket
from models.pixelcnn import get_pixelcnn
from models.vq_vae import get_vq_vae


class CallbacksTestCase(unittest.TestCase):
    def test_vq_vae_local_reconstruction(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            config = load_config(run_id='dummy', artifact_folder=tempdir)
            artifacts_folder = config['models']['vq_vae']['artifacts']['folder']
            config['models']['vq_vae']['artifacts']['reconstructions']['save_every_epoch'] = 1
            config['data']['images']['folder'] = img_root_dir
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []
            config['data']['images']['filter']['orientation'] = 'any'

            reconstructor = CustomImageSamplerCallback(config)
            # Normally, the model is passed to the callback object using history.fit()
            # but we'll inject it here directly to skip having to train
            setattr(reconstructor, 'model', get_vq_vae(config))
            reconstructor.on_epoch_end(epoch=0)
            expected_output_path = os.path.join(artifacts_folder, 'reconstructions', 'epoch-1-1.png')
            saved_img = Image.open(expected_output_path)

            with self.subTest('It creates a three-channel image'):
                self.assertEqual(saved_img.getbands(), ('R', 'G', 'B'))

    def test_vq_vae_bucket_reconstruction_save(self) -> None:
        with TemporaryDirectory() as tempdir:
            img_root_dir = tempdir + '/img'
            shutil.copytree('tests/data', img_root_dir + '/set-1/')

            bucket_artifacts_folder = 'gs://antfield/test/artifacts'
            config = load_config(run_id='dummy', artifact_folder=bucket_artifacts_folder)
            config['models']['vq_vae']['artifacts']['reconstructions']['save_every_epoch'] = 1
            config['data']['images']['folder'] = img_root_dir
            config['data']['images']['filter']['include'] = None
            config['data']['images']['filter']['exclude'] = []
            config['data']['images']['filter']['orientation'] = 'any'

            reconstructor = CustomImageSamplerCallback(config)
            # Normally, the model is passed to the callback object using history.fit()
            # but we'll inject it here directly to skip having to train
            setattr(reconstructor, 'model', get_vq_vae(config))
            target = 'test/artifacts/reconstructions/epoch-1-1.png'
            blob = Blob(name=target, bucket=get_bucket(bucket_artifacts_folder))

            with self.subTest('It does not have the target reconstruction blob yet'):
                self.assertFalse(blob.exists())

            with self.subTest('It saves the image'):
                reconstructor.on_epoch_end(0)
                self.assertTrue(blob.exists())
                blob.delete()

    def test_pixelcnn_local_reconstruction_sampler(self) -> None:
        with TemporaryDirectory() as tempdir:
            config = load_config(run_id='dummy', artifact_folder=tempdir)

            vq_vae_conf = config['models']['vq_vae']
            vq_vae_conf['latent_size'] = 8
            # reduce from 640x640 to 5x5
            vq_vae_conf['conv2d'] = [
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
                {'filters': 16, 'kernel_size': 3, 'strides': 2},
            ]

            # Save an untrained model
            vq_vae = get_vq_vae(config)
            save_path = os.path.join(tempdir, 'vq_vae')
            vq_vae.save(save_path)

            pxl_conf = config['models']['pixelcnn']
            pxl_conf['input_vq_vae'] = save_path
            pxl_conf['artifacts']['reconstructions']['save_every_epoch'] = 1
            pxl_conf['artifacts']['folder'] = tempdir

            pixelcnn = get_pixelcnn(config)

            reconstructor = PixelCNNReconstructionSaver(config, vq_vae.get_layer('decoder'))
            # Normally, the model is passed to the callback object using history.fit()
            # but we'll inject it here directly to skip having to train
            setattr(reconstructor, 'model', pixelcnn)
            reconstructor.on_epoch_end(epoch=0, logs={})

            with self.subTest(f'It generates all images of the batch size {pxl_conf["batch_size"]}'):
                self.assertTrue(os.path.isfile(os.path.join(tempdir, 'reconstructions', 'epoch-1-1.png')))
