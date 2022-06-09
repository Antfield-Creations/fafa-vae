import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore


# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
from models.loaders.config import Config
from models.vqvae import get_vqvae


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type: str, **kwargs: dict) -> None:
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)
        self.kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=self.kernel_shape)

    def build(self, input_shape: tuple) -> None:
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)

        # Use the initialized kernel to create the mask
        self.mask[: self.kernel_shape[0] // 2, ...] = 1.0
        self.mask[self.kernel_shape[0] // 2, : self.kernel_shape[1] // 2, ...] = 1.0

        if self.mask_type == "B":
            self.mask[self.kernel_shape[0] // 2, self.kernel_shape[1] // 2, ...] = 1.0

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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


def get_pixelcnn(config: Config) -> keras.Model:
    vqvae_trainer = get_vqvae(config)
    encoder = vqvae_trainer.get_layer('encoder')
    pixelcnn_input_shape = encoder.output_shape[1:-1]

    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"  # type: ignore
    )(ohe)

    for _ in range(config['models']['pixelcnn']['num_residual_blocks']):
        x = ResidualBlock(filters=128)(x)

    for _ in range(config['models']['pixelcnn']['num_pixelcnn_layers']):
        x = PixelConvLayer(
            mask_type="B",      # type: ignore
            filters=128,        # type: ignore
            kernel_size=1,      # type: ignore
            strides=1,          # type: ignore
            activation="relu",  # type: ignore
            padding="valid",    # type: ignore
        )(x)

    out = keras.layers.Conv2D(
        filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    return pixel_cnn
