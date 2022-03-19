import json
import logging
import os.path
from functools import lru_cache
from os.path import isfile
from typing import List, Optional

import pandas
import yaml
from PIL import Image

from keras_preprocessing.image import ImageDataGenerator

from pandas import DataFrame, Series

# Type alias for config type
from tqdm import tqdm

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

    img_folder = os.path.expanduser(img_folder)
    metadata_path = os.path.join(img_folder, 'metadata.json')

    if exclude_tags is None:
        exclude_tags_set = set()
    else:
        exclude_tags_set = set(exclude_tags)

    if not isfile(metadata_path):
        export_metadata(img_folder)

    df = pandas.read_json(metadata_path)
    # Convert tags to immutabel in order to be able to hash it. The filter application requires this
    df['tags'] = df['tags'].apply(lambda tags: frozenset(tags))

    # Validate that the exclusion an inclusion tags do not overlap
    # If no include tags are given, anything is included and any exclusion tag may be applied
    if include_tags is not None:
        # The mask returns true for each record where the picture tags have any overlap with the inclusion tags
        include_tags_set = set(include_tags)
        both = include_tags_set.intersection(exclude_tags_set)
        if len(both) > 0:
            raise ValueError(f'Tags "{both}" found in both included and excluded tags.')

        @lru_cache
        def tag_includer(tags) -> bool:
            has_tags = tags.intersection(include_tags_set) != set()
            return has_tags

        mask = df['tags'].apply(tag_includer)
        df = df[mask]

    if exclude_tags_set != set():
        @lru_cache
        def tag_excluder(tags) -> bool:
            no_matches = tags.intersection(exclude_tags_set) == set()
            return no_matches

        # The mask returns true for each record where the picture tags have no overlap with the exclusion tags
        mask = df['tags'].apply(tag_excluder)
        df = df[mask]

    return df


def export_metadata(img_folder: str) -> None:
    metadata = []
    file_type = '.jpg'
    img_folder = os.path.expanduser(img_folder)
    export_path = os.path.join(img_folder, 'metadata.json')

    # Force dir recursion into a list so that we can show a progressbar
    logging.info(f'Creating image index for {img_folder}...')
    dirlist = list(os.walk(img_folder))

    for root, dirs, files in tqdm(dirlist):
        relative_path = os.path.relpath(root, img_folder)

        for file in files:
            if file_type not in file:
                continue

            file_base = os.path.splitext(file)[0]
            tags = file_base.split('-')
            # Drop the size designator from the last tag
            tags[-1].strip('_640v640')
            # Add the set id as tag
            tags.append(relative_path.replace('-', '_'))

            # Image dimensions
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path)
            except Exception as e:
                logging.error(f"Couldn't load image at {img_path}: {e}, skipping")
                continue

            metadata.append({
                'filename': os.path.join(relative_path, file),
                'tags': tags,
                'width': img.width,
                'height': img.height,
            })

    with open(export_path, 'wt') as f:
        f.write(json.dumps(metadata, indent=2))

    logging.info(f'Finished writing metadata to {export_path}')
