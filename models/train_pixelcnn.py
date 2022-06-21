import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa

from models.loaders.callbacks import tensorboard_callback, CustomModelCheckpointSaver, PixelCNNReconstructionSaver
from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.loaders.script_archive import archive_scripts
from models.pixelcnn import get_pixelcnn
from models.vq_vae import get_code_indices


def train(config: Config) -> History:
    # Load "submodels" from saved model
    pxl_conf = config['models']['pixelcnn']
    vq_vae = keras.models.load_model(pxl_conf['input_vq_vae'])
    encoder = vq_vae.get_layer('encoder')
    quantizer = vq_vae.get_layer("vector_quantizer")

    # Generate the codebook indices
    data_generator = PaddingGenerator(config, model_name='pixelcnn')
    codebook_indices = np.array([generate_codes(batch, encoder, quantizer) for batch in data_generator])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

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
        x=codebook_indices,
        y=codebook_indices,
        verbose=1,
        batch_size=pxl_conf['batch_size'],
        steps_per_epoch=pxl_conf['batches_per_epoch'],
        epochs=pxl_conf['epochs'],
        validation_split=0.1,
        callbacks=callbacks
    )

    # Archive current scripts and config used for the session
    archive_scripts(config)

    return history


def generate_codes(batch: tf.Tensor, encoder: keras.Model, quantizer: keras.Model) -> tf.Tensor:
    """
    Fetches codes as indices from the code book for a particular batch of input images.
    These codebook codes act as both the input and reconstruction target for the pixelcnn.

    :param batch:       A Tensor of size returned by the data generator
    :param encoder:     The trained VQ-VAE encoder model
    :param quantizer:   The trained VQ-VAE quantizer model

    :return:            A tensor of size (batch, codebook_rows, codebook_cols)
    """
    encoded_outputs = encoder.predict(batch)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = get_code_indices(quantizer, flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    return codebook_indices
