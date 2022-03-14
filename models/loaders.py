import yaml

from keras_preprocessing.image import ImageDataGenerator

# Type alias for config type
Config = dict


def load_config(path: str = 'config.yaml') -> Config:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


class FAFADataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super(FAFADataGenerator, self).__init__(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            validation_split=0.2,
            **kwargs
        )

