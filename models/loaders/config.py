import logging
import os
import time
from typing import Optional

from ruamel.yaml import YAML

# Type alias for config type
Config = dict


# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def load_config(
        path: str = 'config.yaml',
        run_id: Optional[str] = None,
        artifact_folder: Optional[str] = None) -> Config:
    """
    Loads 'config.yaml' from the current working directory, or somewhere else if specified

    :param path:    Path to the config yaml file
    :param run_id:  Optional manual Id override for the run, mainly for testing
    :param artifact_folder: Optional manual artifact folder override, mainly for testing

    :return: A Config object: a nested dictionary
    """
    yaml = YAML(typ='safe')
    with open(path) as f:
        config = yaml.load(f)

    if run_id is not None:
        config['run_id'] = run_id

    # run_id can be passed either from the function arguments or from config.yaml
    if config['run_id'] is None:
        config['run_id'] = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')

    if artifact_folder is not None:
        config['models']['vqvae']['artifacts']['folder'] = artifact_folder
    else:
        artifact_folder = config['models']['vqvae']['artifacts']['folder']

    # You can use common home-folder tildes '~' in folder specs
    artifact_folder = os.path.expanduser(artifact_folder)

    # Replace run id template with actual run id value
    assert config['run_id'] is not None
    artifact_folder = artifact_folder.replace('{run_id}', config['run_id'])
    config['models']['vqvae']['artifacts']['logs']['folder'].replace('{run_id}', config['run_id'])
    os.makedirs(artifact_folder, exist_ok=True)

    config['images']['folder'] = os.path.expanduser(config['images']['folder'])

    # Configure logging
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        filename=os.path.join(artifact_folder, 'logfile.txt'),
        mode='a',
    )
    logger.addHandler(file_handler)

    return config
