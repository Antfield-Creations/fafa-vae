from keras_preprocessing.image import ImageDataGenerator

from models.loaders.config import Config
from models.loaders.metadata import load_metadata


class FAFADataGenerator(ImageDataGenerator):
    def __init__(self) -> None:
        super(FAFADataGenerator, self).__init__(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=0,
            validation_split=0.2,
        )


def get_generator(config: Config) -> ImageDataGenerator:
    """
    Constucts a Keras ImageDataGenerator to load data from. You can iterate over the data in the dataset, or get a
    batch using the data_generator.next() method

    :param config: Config dict containing tag filter settings and image folder

    :return: an iterator over numpy batches of the size of the image
    """
    fafa_loader = FAFADataGenerator()
    img_folder = config['images']['folder']

    img_metadata = load_metadata(
        img_folder=img_folder,
        include_tags=config['images']['filter']['include'],
        exclude_tags=config['images']['filter']['exclude'],
    )

    batch_size = config['models']['vae']['batch_size']
    data_generator = fafa_loader.flow_from_dataframe(
        dataframe=img_metadata,
        class_mode=None,
        target_size=(config['images']['height'], config['images']['width']),
        directory=img_folder,
        batch_size=batch_size,
    )

    return data_generator
