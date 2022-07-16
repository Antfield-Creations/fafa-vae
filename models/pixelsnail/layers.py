from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.pixelsnail.utils import get_causal_mask
from models.pixelsnail.wn import WeightNormalization


# ----------------------------------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------------------------------

class Shift(keras.layers.Layer):
    """
    A layer to shift a tensor
    """

    def __init__(self, direction: str, size: int = 1, **kwargs: dict):
        self.size = size
        self.direction = direction
        super(Shift, self).__init__(**kwargs)

        if self.direction == "down":
            self.pad = keras.layers.ZeroPadding2D(padding=((self.size, 0), (0, 0)), data_format="channels_last")
            self.crop = keras.layers.Cropping2D(((0, self.size), (0, 0)))
        elif self.direction == "right":
            self.pad = keras.layers.ZeroPadding2D(padding=((0, 0), (self.size, 0)), data_format="channels_last")
            self.crop = keras.layers.Cropping2D(((0, 0), (0, self.size)))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        super(Shift, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.crop(self.pad(inputs))

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> dict:
        config = super(Shift, self).get_config()
        config.update({
            'direction': self.direction,
            'size': self.size
        })
        return config


class CausalConv2D(keras.layers.Layer):
    """
    Basic causal convolution layer; implementing causality and weight normalization.
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 weight_norm: bool = True,
                 shift: Optional[str] = None,
                 strides: int = 1,
                 activation: str = "relu",
                 name: str = '',
                 **kwargs: dict):
        self.output_dim = filters
        super(CausalConv2D, self).__init__(name=name, **kwargs)

        pad_h = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
        pad_v = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
        if shift == "down":
            pad_h = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            pad_v = (kernel_size[0] - 1, 0)
        elif shift == "right":
            pad_h = (kernel_size[1] - 1, 0)
            pad_v = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
        elif shift == "downright":
            pad_h = (kernel_size[1] - 1, 0)
            pad_v = (kernel_size[0] - 1, 0)

        self.padding = (pad_v, pad_h)

        self.pad = keras.layers.ZeroPadding2D(
            padding=self.padding,
            data_format="channels_last"
        )
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="VALID",
            strides=strides,
            activation=activation
        )

        if weight_norm:
            self.conv = WeightNormalization(self.conv, data_init=True)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        super(CausalConv2D, self).build(input_shape)  # Be sure to call this before training

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.conv(self.pad(inputs))  # noqa

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> tf.TensorShape:
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> dict:
        config = super(CausalConv2D, self).get_config()
        config.update({
            'padding': self.padding,
            'output_dim': self.output_dim
        })

        return config


class NetworkInNetwork(keras.layers.Layer):
    """
    Basic causal convolution layer; implementing causality and weight normalization.

    """

    def __init__(self, filters: int, activation: str, weight_norm: bool = True, **kwargs: dict):
        super(NetworkInNetwork, self).__init__(**kwargs)
        self.filters = filters
        self.activation_function = keras.layers.Activation(activation)

        if weight_norm:
            self.dense = WeightNormalization(
                keras.layers.Dense(self.filters)
            )
        else:
            self.dense = keras.layers.Dense(self.filters)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        super(NetworkInNetwork, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.dense(inputs)
        outputs = self.activation_function(inputs)
        return outputs

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (self.filters,)

    def get_config(self) -> dict:
        config = super(NetworkInNetwork, self).get_config()
        return config


class CausalAttention(keras.layers.Layer):
    """
    """

    def __init__(self, **kwargs: dict):
        super(CausalAttention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        super(CausalAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x: tf.Tensor) -> tf.Tensor:
        key, query, value = x

        nr_chns = key.shape[-1]
        mixin_chns = value.shape[-1]

        canvas_size = int(np.prod(key.shape[1:-1]))
        canvas_size_q = int(np.prod(query.shape[1:-1]))
        causal_mask = get_causal_mask(canvas_size_q)

        q_m = keras.layers.Reshape((canvas_size_q, nr_chns))(tf.debugging.check_numerics(query, "badQ"))
        k_m = keras.layers.Reshape((canvas_size, nr_chns))(tf.debugging.check_numerics(key, "badK"))
        v_m = keras.layers.Reshape((canvas_size, mixin_chns))(tf.debugging.check_numerics(value, "badV"))

        dot = tf.matmul(q_m, k_m, transpose_b=True)
        dk = tf.cast(nr_chns, tf.float32)
        causal_probs = tf.nn.softmax(dot / tf.math.sqrt(dk) - 1e9 * causal_mask, axis=-1) * causal_mask
        # causal_probs = tf.nn.softmax(dot, axis=-1) * causal_mask
        mixed = tf.matmul(causal_probs, v_m)

        # mixed = tf.keras.layers.Attention(causal=True)([q_m, v_m, k_m])
        out = keras.layers.Reshape(query.shape[1:-1] + [mixin_chns])(mixed)
        out = tf.debugging.check_numerics(out, "bad mixed")

        return out

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> tf.TensorShape:
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> dict:
        config = super(CausalAttention, self).get_config()
        return config


# ----------------------------------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------------------------------


def GatedResidualBlock(inputs: tf.Tensor,
                       aux: Optional[tf.Tensor],
                       nonlinearity: str,
                       dropout: float,
                       conv1: keras.layers.Layer,
                       conv2: keras.layers.Layer) -> keras.layers.Layer:
    """
    inputs, aux are both logits; logits are also returned from the function
    """
    filters = inputs.shape[-1]
    activation = keras.layers.Activation(nonlinearity)

    # should this not have activation??
    c1 = conv1(activation(inputs))

    if aux is not None:
        # add short-cut connection if auxiliary input 'a' is given
        # using NIN (network-in-network)
        c1 += NetworkInNetwork(filters, activation='linear')(activation(aux))  # noqa

    # c1 is passed through a non-linearity step here; not sure if it is needed??
    c1 = activation(c1)

    if dropout > 0.0:
        c1 = keras.layers.Dropout(dropout)(c1)

    c2 = conv2(c1)

    # Gating ; split into two pieces along teh channels
    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)

    # skip connection to input
    x_out = inputs + c3

    return x_out


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------

def pixelSNAIL(
        attention: bool = True,
        out_channels: Optional[int] = None,
        num_pixel_blocks: int = 1,
        num_grb_per_pixel_block: int = 1,
        dropout: float = 0.0,
        nr_filters: int = 128
) -> keras.Model:
    """
    Defines a PixelSNAIL model and returns it, still requires compilation through .compile()

    :param attention:               Whether to use attention
    :param out_channels:            The number of channels to include in the output
    :param num_pixel_blocks:        The number of masked convolution layers to use
    :param num_grb_per_pixel_block: The number of gated residual blocks to use
    :param dropout:                 Dropout proportion to apply, needs to be within [0.,1.]
    :param nr_filters:              The number of filters to apply in the convolution layers

    :return: The keras
    """

    nr_logistic_mix = 10
    kernel_size = 3

    x_in = keras.Input(shape=(32, 32, 3))

    k_d = (kernel_size - 1, kernel_size)
    k_dr = (kernel_size - 1, kernel_size - 1)

    u = Shift("down")(CausalConv2D(nr_filters, (kernel_size - 1, kernel_size), shift="down")(x_in))  # noqa
    ul = Shift("down")(CausalConv2D(nr_filters, (1, kernel_size), shift="down")(x_in))               # noqa
    ul += Shift("right")(CausalConv2D(nr_filters, (kernel_size - 1, 1), shift="downright")(x_in))    # noqa

    for pixel_block in range(num_pixel_blocks):
        for gated_block in range(num_grb_per_pixel_block):
            conv1u = CausalConv2D(
                filters=nr_filters,
                kernel_size=k_d,
                shift="down",
                activation="elu",
                name=f"causalconv_u_1_{pixel_block}_{gated_block}"
            )

            conv2u = CausalConv2D(
                filters=2 * nr_filters,
                kernel_size=k_d,
                shift="down",
                activation="elu",
                name=f"causalconv_u_2_{pixel_block}_{gated_block}"
            )

            u = GatedResidualBlock(
                inputs=u,
                aux=None,
                nonlinearity="elu",
                dropout=dropout,
                conv1=conv1u,
                conv2=conv2u
            )

            conv1ul = CausalConv2D(
                filters=nr_filters,
                kernel_size=k_dr,
                shift="downright",
                activation="elu",
                name=f"causalconv_ul_1_{pixel_block}_{gated_block}"
            )
            conv2ul = CausalConv2D(
                filters=2 * nr_filters,
                kernel_size=k_dr,
                shift="downright",
                activation="elu",
                name=f"causalconv_ul_2_{pixel_block}_{gated_block}"
            )
            ul = GatedResidualBlock(
                inputs=ul,
                aux=u,
                nonlinearity="elu",
                dropout=dropout,
                conv1=conv1ul,
                conv2=conv2ul
            )

        if attention:
            content = keras.layers.Concatenate(axis=3)([x_in, ul])

            content = tf.debugging.check_numerics(content, "bad conent")
            channels = content.shape[-1]
            kv = GatedResidualBlock(
                inputs=content,
                aux=None,
                nonlinearity="elu",
                dropout=dropout,
                conv1=NetworkInNetwork(filters=channels, activation='linear'),
                conv2=NetworkInNetwork(filters=2 * channels, activation='linear')
            )
            kv = NetworkInNetwork(filters=2 * nr_filters, activation='linear')(kv)  # noqa
            key, value = tf.split(kv, 2, axis=3)

            query = GatedResidualBlock(
                inputs=ul,
                aux=None,
                nonlinearity="elu",
                dropout=dropout,
                conv1=NetworkInNetwork(filters=nr_filters, activation='linear'),
                conv2=NetworkInNetwork(filters=2 * nr_filters, activation='linear')
            )

            query = NetworkInNetwork(filters=nr_filters, activation='linear')(query)  # noqa
            a = CausalAttention()([key, query, value])  # noqa
            a = tf.debugging.check_numerics(a, "bad a!!")
        else:
            a = None

        ul = GatedResidualBlock(
            inputs=ul,
            aux=a,
            nonlinearity="elu",
            dropout=dropout,
            conv1=NetworkInNetwork(filters=nr_filters, activation='linear'),
            conv2=NetworkInNetwork(filters=2 * nr_filters, activation='linear')
        )

    ul = keras.layers.Activation("elu")(ul)

    if out_channels is not None:
        filters = out_channels
        x_out = NetworkInNetwork(filters=filters, activation='linear')(ul)  # noqa
    else:
        filters = 10 * nr_logistic_mix
        x_out = NetworkInNetwork(filters=filters, activation='linear')(ul)  # noqa

    model = tf.keras.Model(inputs=x_in, outputs=x_out)

    return model
