import logging
import os
import time
from typing import Optional

from ruamel.yaml import YAML

# Type alias for config type
Config = dict


def load_config(path: str = 'config.yaml', run_id: Optional[str] = None) -> Config:
    """
    Loads 'config.yaml' from the current working directory, or somewhere else if specified

    :param path:    Path to the config yaml file
    :param run_id:  Optional manually set Id for the run, mainly for testing

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

    # You can use common home-folder tildes '~' in folder specs
    config['models']['vqvae']['artifacts']['folder'] = \
        os.path.expanduser(config['models']['vqvae']['artifacts']['folder']).format(run_id=config['run_id'])

    config['images']['folder'] = os.path.expanduser(config['images']['folder'])

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(config['models']['vqvae']['artifacts']['folder'], 'logfile.txt'),
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )

    return config
