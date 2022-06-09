import logging
import os.path
from urllib.parse import urlparse

from google.cloud import storage  # noqa
from tqdm import tqdm

from models.loaders.config import Config


def provision(config: Config) -> None:
    storage_client = storage.Client()

    source_location = urlparse(str(config['data']['images']['cloud_storage_folder']))
    if source_location.scheme != 'gs' and source_location.scheme != 'gcs':
        logging.warning('Source location is not a google storage type, exiting.')
        return

    target_folder = str(config['data']['images']['folder'])
    bucket = storage_client.get_bucket(source_location.netloc)
    source_path = source_location.path.lstrip('/')

    for blob in tqdm(bucket.list_blobs(prefix=source_path), total=config['data']['images']['file_count']):
        # Skip if the blob is a "directory"
        if blob.name.endswith('/'):
            continue

        # Strip matching parts of source from target
        if blob.name.startswith(source_path):
            target_subpath = blob.name.removeprefix(source_path)
        else:
            target_subpath = blob.name

        # Drop any leading slashes from subpath, otherwise it is not a relative path
        target_subpath = target_subpath.removeprefix('/')

        # Assemble the target download path
        download_path = os.path.join(target_folder, target_subpath)
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        blob.download_to_filename(download_path)

    return
