import logging
import os.path
from urllib.parse import urlparse

from models.loaders.config import Config, load_config

from google.cloud import storage  # noqa


def provision(config: Config) -> int:
    storage_client = storage.Client()

    source_location = urlparse(str(config['data']['images']['cloud_storage_folder']))
    if source_location.scheme != 'gs' and source_location.scheme != 'gcs':
        logging.warning('Source location is not a google storage type, exiting.')
        return 1

    target_folder = str(config['data']['images']['folder'])
    # if not os.path.isdir(target_folder):
    #     raise NotADirectoryError(f'Target download directory {target_folder} does not exist')

    bucket = storage_client.get_bucket(source_location.netloc)
    source_path = source_location.path.lstrip('/')

    for blob in bucket.list_blobs(prefix=source_path):
        # Strip matching parts of source from target
        if blob.name.endswith(source_path):
            target_subpath = blob.name.removeprefix(source_path)
        else:
            target_subpath = blob.name

        # Drop any leading slashes from subpath, otherwise it is not a relative path
        target_subpath = target_subpath.removeprefix('/')

        # Assemble the target download path
        download_path = os.path.join(target_folder, target_subpath)
        blob.download_to_filename(download_path)

    return 0


if __name__ == '__main__':
    config = load_config()
    raise SystemExit(provision(config))
