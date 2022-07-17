import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras, Tensor
from tensorflow.keras import layers  # type: ignore

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
from models.loaders.config import Config


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type: str, **kwargs: dict) -> None:
        """
        Initializes a masked Conv2 layer

        :param mask_type:   The masking type:
                            -   The "A" mask type, for only the initial convolution layer in a PixelCNN,
                                zeroing the central pixel in the mask;
                            -   "B" for subsequent convolution layers
        :param kwargs:
        """
        super(PixelConvLayer, self).__init__()
        if mask_type not in {"A", "B"}:
            raise ValueError(f'Unknown mask type {mask_type}, choose either "A" or "B"')
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)
        self.mask: Optional[Tensor] = None

    @classmethod
    def generate_mask(cls, kernel_shape: Tuple[int, int, int], mask_type: str) -> tf.Tensor:
        """
        Uses an initialized kernel to create the mask. This class method is useful for re-creating the mask on a saved
        model as the mask tensor itself cannot be saved.

        :param kernel_shape:    Shape of the Conv2D kernel
        :param mask_type:       "A" or "B", see class initializer function

        :return:                The mask as a tensor
        """

        row_center = kernel_shape[0] // 2
        col_center = kernel_shape[1] // 2

        mask = np.ones(shape=kernel_shape, dtype=np.float32)

        # Set the center "pixel" and every pixel to the right in the receptive field to "disabled"
        mask[row_center, col_center:, ...] = 0.

        # Set the rows below to "disabled"
        mask[row_center + 1:, ...] = 0.

        if mask_type == "B":
            mask[row_center, col_center, ...] = 1.

        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
        return mask_tensor

    @classmethod
    def from_config(cls, config: dict) -> 'PixelConvLayer':
        instance = cls(**config)
        instance.mask = cls.generate_mask(instance.kernel_shape, instance.mask_type)
        return instance

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update({
            'mask_type': self.mask_type,
            'conv': self.conv,
        })

        return config

    def build(self, input_shape: tuple) -> None:
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.shape
        self.mask = PixelConvLayer.generate_mask(kernel_shape, self.mask_type)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters: int, **kwargs: dict):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,   # type: ignore
            kernel_size=3,          # type: ignore
            activation="relu",      # type: ignore
            padding="same",         # type: ignore
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update({
            'conv1': self.conv1,
            'pixel_conv': self.pixel_conv,
            'conv2': self.conv2,
        })
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.pixel_conv(x)  # noqa
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


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
    assert encoder.output_shape[-1] % embedding_size == 0, \
        f"Encoder output dimension must be a multiple of the embedding size, got {encoder.output_shape[-1]} " \
        f"vs {embedding_size}"

    embedding_stack = encoder.output_shape[-1] // embedding_size
    pixelcnn_input_shape = encoder.output_shape[1:-1] + (embedding_stack,)

    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, config['models']['vq_vae']['num_embeddings'])

    outputs = PixelConvLayer(  # noqa
        mask_type="A",
        filters=pxl_conf['pixel_conv_residual_filters'],
        kernel_size=7,      # type: ignore
        activation="relu",  # type: ignore
        padding="same"      # type: ignore
    )(ohe)

    for _ in range(config['models']['pixelcnn']['num_residual_blocks']):
        outputs = ResidualBlock(filters=pxl_conf['pixel_conv_residual_filters'])(outputs)  # noqa

    for _ in range(config['models']['pixelcnn']['num_pixelcnn_layers']):
        outputs = PixelConvLayer(     # noqa
            mask_type="B",
            filters=pxl_conf['pixel_conv_1x1_filters'],
            kernel_size=3,      # type: ignore
            strides=1,          # type: ignore
            activation="relu",  # type: ignore
            padding="same"     # type: ignore
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=config['models']['vq_vae']['num_embeddings'],
        kernel_size=1,
        strides=1,
        padding="same"
    )(outputs)

    pixel_cnn = keras.Model(pixelcnn_inputs, outputs, name="pixel_cnn")
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
