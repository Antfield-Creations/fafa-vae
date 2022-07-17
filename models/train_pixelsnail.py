import logging

from tensorflow import keras
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa

from models.loaders.callbacks import tensorboard_callback, CustomModelCheckpointSaver, PixelSNAILReconstructionSaver
from models.loaders.config import Config
from models.loaders.pixelcnn_data_generator import CodebookGenerator
from models.loaders.script_archive import archive_scripts
from models.pixelsnail.layers import get_pixelsnail
from models.pixelsnail.losses import discretized_mix_logistic_loss


def train(config: Config) -> History:
    # Load "submodels" from saved model
    pxlsnail_conf = config['models']['pixelsnail']
    vq_vae = keras.models.load_model(pxlsnail_conf['input_vq_vae'])
    encoder = vq_vae.get_layer('encoder')
    quantizer = vq_vae.get_layer("vector_quantizer")

    # Codebook indices batch generator
    data_generator = CodebookGenerator(config, encoder, quantizer)

    pixelsnail = get_pixelsnail(config, quantizer)
    pixelsnail.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pxlsnail_conf['learning_rate']),
        loss=discretized_mix_logistic_loss,
        metrics=["accuracy"],
    )
    logging.info('Compiled pixelcnn model')

    logs_folder = pxlsnail_conf['artifacts']['logs']['folder']
    tensorboard_cb = tensorboard_callback(artifacts_folder=logs_folder)
    checkpoint_saver = CustomModelCheckpointSaver(config, 'pixelsnail')

    callbacks = [tensorboard_cb, checkpoint_saver]

    # Since generating pixelcnn embedding reconstructions is time-consuming, we only generate if requested
    if pxlsnail_conf['artifacts']['reconstructions']['enabled']:
        callbacks.append(PixelSNAILReconstructionSaver(config, vq_vae))

    history = pixelsnail.fit(
        x=data_generator,
        verbose=1,
        batch_size=pxlsnail_conf['batch_size'],
        steps_per_epoch=pxlsnail_conf['batches_per_epoch'],
        epochs=pxlsnail_conf['epochs'],
        callbacks=callbacks
    )

    # Archive current scripts and config used for the session
    archive_scripts(config, 'pixelsnail')

    return history
