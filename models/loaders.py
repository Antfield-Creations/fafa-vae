from typing import Dict, Union

import yaml


Config = Dict[str, Union[str, int]]


def load_config(path: str = 'config.yaml') -> Config:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config
