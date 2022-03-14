"""
A Sampling layer module, adapted from https://keras.io/examples/generative/vae/
"""
from typing import Tuple

from keras import layers
from keras import backend as K
from tensorflow import Tensor


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + K.exp(0.5 * z_log_var) * epsilon
