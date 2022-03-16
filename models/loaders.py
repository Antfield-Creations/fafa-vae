import json
import os.path
from os.path import isfile
from typing import List, Optional

import pandas
import yaml

from keras_preprocessing.image import ImageDataGenerator

from pandas import DataFrame

# Type alias for config type
Config = dict


def load_config(path: str = 'config.yaml') -> Config:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


class FAFADataGenerator(ImageDataGenerator):
    def __init__(self) -> None:
        super(FAFADataGenerator, self).__init__(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            validation_split=0.2,
        )


def load_metadata(
        img_folder: str,
        exclude_tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None) -> DataFrame:

    metadata_path = os.path.join(img_folder, 'metadata.json')

    if not isfile(metadata_path):
        export_metadata(img_folder)

    df = pandas.read_json(metadata_path)

    if include_tags is not None:
        mask = df.tags.apply(lambda tags: set(tags).intersection(set(include_tags)) != set())
        df = df[mask]
    if exclude_tags is None:
        exclude_tags = []

    return df


def export_metadata(img_folder: str) -> None:
    metadata = []
    file_filter = '.jpg'
    export_path = os.path.join(img_folder, 'metadata.json')

    for root, dirs, files in os.walk(img_folder):
        relative_path = os.path.relpath(root, img_folder)

        for file in files:
            if file_filter not in file:
                continue

            file_base = os.path.splitext(file)[0]
            tags = file_base.split('-')
            metadata.append({
                'filename': os.path.join(relative_path, file),
                'tags': tags
            })

    with open(export_path, 'wt') as f:
        f.write(json.dumps(metadata, indent=2))
