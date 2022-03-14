"""
This is the 'entrypoint' so to say for running or creating ML models.
You will need to run the scraper script first if you want to use the FAFA images.
"""

from argparse import ArgumentParser

from models.loaders import load_config
from models.train import train

if __name__ == '__main__':
    argparser = ArgumentParser(description='Training script for FAFA variational autoencoder')
    argparser.add_argument('-c' '--config', help='Path to the config.yaml file', default='config.yaml')
    args = argparser.parse_args()

    config = load_config()
    train(config)
