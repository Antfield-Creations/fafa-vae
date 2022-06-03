import glob
import os
import shutil
from urllib.parse import urlparse

from google.cloud.storage.blob import Blob  # noqa
from google.cloud.storage.bucket import Bucket  # noqa
from google.cloud.storage.client import Client  # noqa

from models.loaders.config import Config


def get_bucket(blob_url: str) -> Bucket:
    """
    Extracts a bucket name from a blob url and returns the bucket as a google storage object

    :param blob_url: URL of a blob in a bucket, or just a gs://{bucket_name} or gcs://{bucket_name} url string

    :return: a Bucket instance
    """
    gs_url = urlparse(blob_url)
    storage_client = Client()
    bucket = storage_client.bucket(gs_url.netloc)

    return bucket


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
        bucket = get_bucket(artifact_folder)

        for file in files:
            if os.path.isfile(file):
                relative_path = file.removeprefix(root_dir).removeprefix('/')
                bucket_target = f'{bucket_subpath}/{relative_path}'
                blob = Blob(name=bucket_target, bucket=bucket)
                blob.upload_from_filename(filename=file)
    else:
        shutil.copytree(root_dir, artifact_folder)
