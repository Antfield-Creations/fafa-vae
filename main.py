"""
This is the 'entrypoint' so to say for running or creating ML models.
You will need to run the scraper script first if you want to use the FAFA images.
"""
from argparse import ArgumentParser
from typing import Dict, Callable

from models import train_vq_vae, train_pixelcnn
from models.loaders.config import load_config, Config
from models.loaders.data_provision import provision
from models.loaders.metadata import index
from scraper.scraper import scrape

MODES: Dict[str, Callable[[Config], None]] = {
    'scrape': scrape,
    'provision': provision,
    'index': index,
    'train-vq-vae': train_vq_vae.train,
    'train-pixelcnn': train_pixelcnn.train
}


def main(config: Config, mode: str) -> int:
    MODES[mode](config)
    print("Done!")
    return 0


if __name__ == '__main__':
    argparser = ArgumentParser(description='Training script for FAFA variational autoencoder')
    argparser.add_argument('-c', '--config', help='Path to the config.yaml file', default='config.yaml')
    argparser.add_argument('-m', '--mode', help=f'Mode, one of "{MODES}"', required=True)
    args = argparser.parse_args()

    if args.mode not in MODES.keys():
        raise NotImplementedError("This mode isn't implemented (yet)")

    config = load_config(args.config)

    raise SystemExit(main(config, args.mode))
