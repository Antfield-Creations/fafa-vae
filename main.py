"""
This is the 'entrypoint' so to say for running or creating ML models.
You will need to run the scraper script first if you want to use the FAFA images.
"""
import logging
from argparse import ArgumentParser

from models.loaders import load_config, load_metadata, export_metadata
from models.train import train
from scraper.scraper import scrape

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    known_modes = ['scrape', 'index', 'train', 'encode', 'decode']

    argparser = ArgumentParser(description='Training script for FAFA variational autoencoder')
    argparser.add_argument('-c', '--config', help='Path to the config.yaml file', default='config.yaml')
    argparser.add_argument('-m', '--mode', help=f'Mode, one of "{known_modes}"', required=True)
    args = argparser.parse_args()

    config = load_config()

    if args.mode == 'scrape':
        scrape(config)
    elif args.mode == 'index':
        export_metadata(img_folder=config['images']['folder'])
        metadata = load_metadata(
            img_folder=config['images']['folder'],
            exclude_tags=config['images']['filter']['exclude'],
            include_tags=config['images']['filter']['include'],
        )
        logging.info(f'{len(metadata)} images to train on')
    elif args.mode == 'train':
        train(config)
    else:
        raise NotImplementedError("This mode isn't implemented (yet)")
