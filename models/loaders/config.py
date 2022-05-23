import logging
import os
import time
from io import StringIO
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
        config = yaml.load(f)['spec']

    if run_id is not None:
        config['run_id'] = run_id

    # run_id can be passed either from the function arguments or from config.yaml
    if config['run_id'] is None:
        config['run_id'] = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')

    if artifact_folder is not None:
        config['models']['vqvae']['artifacts']['folder'] = artifact_folder

    # Replace run id template with actual run id value
    assert config['run_id'] is not None
    config['models']['vqvae']['artifacts']['folder'] = \
        config['models']['vqvae']['artifacts']['folder'].replace('{run_id}', config['run_id'])

    # You can use common home-folder tildes '~' in folder specs
    config['models']['vqvae']['artifacts']['folder'] = \
        os.path.expanduser(config['models']['vqvae']['artifacts']['folder'])

    config['models']['vqvae']['artifacts']['logs']['folder'] = \
        config['models']['vqvae']['artifacts']['logs']['folder'].replace('{run_id}', config['run_id'])
    config['models']['vqvae']['artifacts']['logs']['folder'] = \
        os.path.expanduser(config['models']['vqvae']['artifacts']['logs']['folder'])

    os.makedirs(config['models']['vqvae']['artifacts']['folder'], exist_ok=True)

    config['data']['images']['folder'] = os.path.expanduser(config['data']['images']['folder'])
    config['data']['images']['folder'] = config['data']['images']['folder'].replace('{run_id}', config['run_id'])

    # Configure logging
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        filename=os.path.join(config['models']['vqvae']['artifacts']['folder'], 'logfile.txt'),
        mode='a',
    )
    logger.addHandler(file_handler)
    logging.info(f"Session has run id {config['run_id']}")

    # Validate the config: convert to text and check for template markers "~" and "{"
    string_stream = StringIO()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.dump(config, string_stream)
    config_as_text = string_stream.getvalue()

    if '~' in config_as_text:
        raise ValueError(f'Not all home-dir tildes "~" were substituted: \n{config_as_text}')
    if '{' in config_as_text:
        raise ValueError(f'Not all template values with "{{" were substituted: \n{config_as_text}')

    return config
