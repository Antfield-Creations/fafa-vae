import logging

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
from models.loaders.config import Config


def get_pixelcnn(config: Config) -> keras.Model:
    pxl_conf = config['models']['pixelcnn']
    saved_model = pxl_conf['artifacts']['resume_model']
    if saved_model is not None:
        logging.info(f"Loading saved model {saved_model}")
        return keras.models.load_model(saved_model)

    vq_vae = keras.models.load_model(pxl_conf['input_vq_vae'])
    encoder = vq_vae.get_layer('encoder')
    quantizer = vq_vae.get_layer('vector_quantizer')

    embedding_size = quantizer.embeddings.shape[0]
    num_embeddings = quantizer.embeddings.shape[1]
    assert encoder.output_shape[-1] % embedding_size == 0, \
        f"Encoder output dimension must be a multiple of the embedding size, got {encoder.output_shape[-1]} " \
        f"vs {embedding_size}"

    embedding_stack_size = encoder.output_shape[-1] // embedding_size
    pixelcnn_input_shape = encoder.output_shape[1:-1] + (embedding_stack_size,)

    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.float32)
    dist = tfd.PixelCNN(
        image_shape=pixelcnn_input_shape,
        num_resnet=pxl_conf['num_residual_blocks'],
        low=0,
        high=num_embeddings,
        dtype=tf.float32,
        name='pixelcnn'
    )
    log_prob = dist.log_prob(pixelcnn_inputs)
    pixel_cnn = keras.Model(inputs=pixelcnn_inputs, outputs=log_prob)
    pixel_cnn.add_loss(-tf.reduce_mean(log_prob))

    return pixel_cnn


def get_pixelcnn_sampler(pixelcnn: keras.Model) -> keras.Model:
    """
    Creates a mini sampler model. This samples from a categorical distribution given a sample set of sample embedding
    imputs. From the sample embedding inputs, a single "block" of "candidate" embedding reconstructions is returned as
    a categorical distribution from which the next autoregressive part of the embedding reconstructions can be
    generated.

    :param pixelcnn: A (partially) trained pixelCNN keras model

    :return: the sampler model
    """
    inputs = layers.Input(shape=pixelcnn.input_shape[1:])
    outputs = pixelcnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    return sampler
