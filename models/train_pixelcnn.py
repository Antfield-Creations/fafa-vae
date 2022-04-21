import logging

import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa

from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.pixelcnn import get_pixelcnn


def train(config: Config) -> History:
    # Load from saved model
    vqvae = keras.models.load_model(config['models']['pixelcnn']['input_vqvae']).vqvae

    encoder = vqvae.vqvae.get_layer('encoder')
    quantizer = vqvae.vqvae.get_layer("vector_quantizer")
    data_generator = PaddingGenerator(config)

    # Generate the codebook indices.
    encoded_outputs = encoder.predict(next(data_generator))
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixel_cnn = get_pixelcnn(config)

    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
    )

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    x = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = dist.sample()
    sampler = keras.Model(inputs, sampled)
    # TODO: do something useful with the sampler
    logging.info(sampler)
