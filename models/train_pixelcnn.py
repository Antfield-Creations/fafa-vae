from tensorflow import keras
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa

from models.loaders.callbacks import tensorboard_callback, CustomModelCheckpointSaver, PixelCNNReconstructionSaver
from models.loaders.config import Config
from models.loaders.pixelcnn_data_generator import CodebookGenerator
from models.loaders.script_archive import archive_scripts
from models.pixelcnn import get_pixelcnn


def train(config: Config) -> History:
    # Load "submodels" from saved model
    pxl_conf = config['models']['pixelcnn']
    vq_vae = keras.models.load_model(pxl_conf['input_vq_vae'])
    encoder = vq_vae.get_layer('encoder')
    quantizer = vq_vae.get_layer("vector_quantizer")

    # Codebook indices batch generator
    data_generator = CodebookGenerator(config, encoder, quantizer)

    pixel_cnn = get_pixelcnn(config)
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pxl_conf['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    logs_folder = config['models']['vq_vae']['artifacts']['logs']['folder']
    tensorboard_cb = tensorboard_callback(artifacts_folder=logs_folder)
    checkpoint_saver = CustomModelCheckpointSaver(config, 'pixelcnn')

    callbacks = [tensorboard_cb, checkpoint_saver]
    # Since generating pixelcnn embedding reconstructions is time-consuming, we only generate if requested
    if pxl_conf['artifacts']['reconstructions']['enabled']:
        decoder = vq_vae.get_layer('decoder')
        callbacks.append(PixelCNNReconstructionSaver(config, decoder))

    history = pixel_cnn.fit(
        data_generator,
        verbose=1,
        batch_size=pxl_conf['batch_size'],
        steps_per_epoch=pxl_conf['batches_per_epoch'],
        epochs=pxl_conf['epochs'],
        callbacks=callbacks
    )

    # Archive current scripts and config used for the session
    archive_scripts(config)

    return history
