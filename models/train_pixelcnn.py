import logging

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras, float32
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa

from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.loaders.script_archive import archive_scripts
from models.pixelcnn import get_pixelcnn
from models.vqvae import get_code_indices


def train(config: Config) -> History:
    # Load from saved model
    pxl_conf = config['models']['pixelcnn']
    vqvae = keras.models.load_model(pxl_conf['input_vqvae'])

    data_generator = PaddingGenerator(config)

    # Generate the codebook indices.
    encoder = vqvae.get_layer('encoder')
    encoded_outputs = encoder.predict(next(data_generator))
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    quantizer = vqvae.get_layer("vector_quantizer")
    codebook_indices = get_code_indices(quantizer, flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixel_cnn = get_pixelcnn(config)

    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pxl_conf['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        verbose=1,
        batch_size=pxl_conf['batch_size'],
        steps_per_epoch=pxl_conf['batches_per_epoch'],
        epochs=pxl_conf['epochs'],
        validation_split=0.1,
    )

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    x = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = tf.py_function(func=dist.sample, inp=[x], Tout=float32)
    sampler = keras.Model(inputs, sampled)
    # TODO: do something useful with the sampler
    logging.info(sampler)

    # Archive current scripts and config used for the session
    archive_scripts(config)

    return history
