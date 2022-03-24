"""
A Sampling layer module, adapted from https://keras.io/examples/generative/vae/
"""
from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras import layers
from tensorflow.python.keras.backend import random_normal


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
