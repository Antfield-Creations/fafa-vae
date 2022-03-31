import json
import os
from functools import lru_cache
from logging import getLogger
from os.path import isfile
from typing import Optional, List

import pandas
from PIL import Image
from pandas import DataFrame
from tqdm import tqdm

logger = getLogger('Metadata')


def load_metadata(
        img_folder: str,
        orientation: str = 'any',
        exclude_tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None) -> DataFrame:

    metadata_path = os.path.join(img_folder, 'metadata.json')

    if exclude_tags is None:
        exclude_tags_set = set()
    else:
        exclude_tags_set = set(exclude_tags)

    if not isfile(metadata_path):
        export_metadata(img_folder)

    df = pandas.read_json(metadata_path)

    # For now: only use portrait-oriented images
    if orientation == 'portrait':
        df = df[df['height'] > df['width']]
    elif orientation == 'landscape':
        df = df[df['width'] > df['height']]

    # Convert tags to immutable in order to be able to hash it. The filter application requires this
    df['tags'] = df['tags'].apply(lambda tags: frozenset(tags))

    # Validate that the exclusion an inclusion tags do not overlap
    # If no include tags are given, anything is included and any exclusion tag may be applied
    if include_tags is not None:
        logger.info("Filtering include tags...")
        include_tags_set = set(include_tags)
        both = include_tags_set.intersection(exclude_tags_set)
        if len(both) > 0:
            raise ValueError(f'Tags "{both}" found in both included and excluded tags.')

        @lru_cache
        def tag_includer(tags: frozenset) -> bool:
            # The mask returns true for each record where the picture tags have any overlap with the inclusion tags
            has_overlap = not tags.isdisjoint(include_tags_set)
            return has_overlap

        mask = df['tags'].apply(tag_includer)
        df = df[mask]

    if exclude_tags_set != set():
        logger.info("Filtering exclude tags...")

        @lru_cache
        def tag_excluder(tags: frozenset) -> bool:
            no_overlap = tags.isdisjoint(exclude_tags_set)
            return no_overlap

        # The mask returns true for each record where the picture tags have no overlap with the exclusion tags
        mask = df['tags'].apply(tag_excluder)
        df = df[mask]

    return df


def export_metadata(img_folder: str) -> None:
    file_type = '.jpg'
    export_path = os.path.join(img_folder, 'metadata.json')

    # Force dir recursion into a list so that we can show a progressbar
    logger.info(f'Creating image index for {img_folder}...')
    dirlist = list(os.walk(img_folder))

    metadata = []
    for root, dirs, files in tqdm(dirlist):
        relative_path = os.path.relpath(root, img_folder)

        for file in files:
            if file_type not in file:
                continue

            file_base = os.path.splitext(file)[0]
            tags = file_base.split('-')
            # Drop the size designator from the last tag
            tags[-1] = tags[-1].strip('_640v640')
            # Add the set id as tag
            tags.append(relative_path.replace('-', '_'))

            # Image dimensions
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path)
                # Try loading the image to validate it is not truncated.
                # Truncated images lead to hard errors in machine learning
                img.load()
            except Exception as e:
                logger.error(f"Couldn't load image at {img_path}: {e}, skipping")
                continue

            metadata.append({
                'filename': os.path.join(relative_path, file),
                'tags': tags,
                'width': img.width,
                'height': img.height,
            })
            img.close()

    with open(export_path, 'wt') as f:
        f.write(json.dumps(metadata, indent=2))

    logger.info(f'Finished writing metadata to {export_path}')