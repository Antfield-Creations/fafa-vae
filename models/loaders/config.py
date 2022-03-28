import os

from ruamel.yaml import YAML

# Type alias for config type
Config = dict


def load_config(path: str = 'config.yaml') -> Config:
    """
    Loads 'config.yaml' from the current working directory, or somewhere else if specified

    :param path: Path to the config yaml file

    :return: A Config object: a nested dictionary
    """
    yaml = YAML(typ='safe')
    with open(path) as f:
        config = yaml.load(f)

    # You can use common home-folder tildes '~' in folder specs
    config['models']['vae']['artifacts']['folder'] = \
        os.path.expanduser(config['models']['vae']['artifacts']['folder'])

    config['images']['folder'] = \
        os.path.expanduser(config['images']['folder'])

    return config
