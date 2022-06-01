"""
This is the 'entrypoint' so to say for running or creating ML models.
You will need to run the scraper script first if you want to use the FAFA images.
"""
import logging
from argparse import ArgumentParser

from models.loaders.config import load_config, Config
from models.loaders.data_provision import provision
from models.loaders.metadata import export_metadata, load_metadata
from models.train import train
from scraper.scraper import scrape


def main(config: Config, mode: str) -> int:
    if mode == 'scrape':
        scrape(config)
    elif mode == 'provision':
        provision(config)
    elif mode == 'index':
        export_metadata(img_folder=config['data']['images']['folder'])
        metadata = load_metadata(
            img_folder=config['data']['images']['folder'],
            exclude_tags=config['data']['images']['filter']['exclude'],
            include_tags=config['data']['images']['filter']['include'],
        )
        logging.info(f'{len(metadata)} images to train on')
    elif mode == 'train':
        train(config)

    print("Done!")
    return 0


if __name__ == '__main__':
    known_modes = ['scrape', 'provision', 'index', 'train', 'encode', 'decode']

    argparser = ArgumentParser(description='Training script for FAFA variational autoencoder')
    argparser.add_argument('-c', '--config', help='Path to the config.yaml file', default='config.yaml')
    argparser.add_argument('-m', '--mode', help=f'Mode, one of "{known_modes}"', required=True)
    args = argparser.parse_args()

    if args.mode not in known_modes:
        raise NotImplementedError("This mode isn't implemented (yet)")

    config = load_config(args.config)

    raise SystemExit(main(config, args.mode))
