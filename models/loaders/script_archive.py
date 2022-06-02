import glob
import os
import shutil
from urllib.parse import urlparse

from google.cloud.storage.blob import Blob
from google.cloud.storage.client import Client  # noqa

from models.loaders.config import Config


def archive_scripts(config: Config) -> None:
    """
    # Archive all config and modules to artifacts folder

    :param config: models.loaders.load_config returned instance

    :return: None
    """
    artifact_folder = str(config['models']['vqvae']['artifacts']['folder'])
    models_src = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(models_src, '..', '..')

    if artifact_folder.startswith('gs://') or artifact_folder.startswith('gcs://'):
        files = glob.glob(pathname=f'{root_dir}/**', recursive=True)
        # The artifact location is a bucket url so we need to extract the subpath from it
        gs_url = urlparse(artifact_folder)
        bucket_subpath = gs_url.path.removeprefix('/')
        storage_client = Client()
        bucket = storage_client.bucket(gs_url.netloc)

        for file in files:
            if os.path.isfile(file):
                relative_path = file.removeprefix(root_dir).removeprefix('/')
                bucket_target = f'{bucket_subpath}/{relative_path}'
                blob = Blob(name=bucket_target, bucket=bucket)
                blob.upload_from_filename(filename=file)
    else:
        shutil.copytree(root_dir, artifact_folder)
