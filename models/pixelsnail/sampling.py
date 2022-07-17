from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import keras


def sample_from_discretized_mix_logistic(logits: tf.Tensor, nr_mix: int) -> tf.Tensor:
    """
    Sampling function for the pixelCNN family of algorithms which will generate an RGB image.

    :param logits:       Tensor (B,H,W,N). The output from a pixelCNN network, where N is (nr_mix * (1 + 3 + 3 + 3))
                    corresponding to the pi_i (mixture indicator), mu_i, s_i and c_i.
    :param nr_mix:  The number of logistic distributions included in the network output. Usually 5 or 10

    :return: Tensor, (B,H,W,3) : The RGB values of the sampled pixels


    """
    ls = list(logits.shape)
    xs = ls[:-1] + [3]

    # split the network output into its pieces
    split = [nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix]
    logit_probs, means, log_s, coeff = tf.split(logits, num_or_size_splits=split, axis=-1)
    means = tf.reshape(means, shape=xs + [nr_mix])
    scale = tf.exp(tf.reshape(log_s, shape=xs + [nr_mix]))
    coeff = tf.reshape(tf.nn.tanh(coeff), shape=xs + [nr_mix])

    # the probabilities for each "mixture indicator"
    logit_probs = tf.nn.log_softmax(logit_probs - tf.reduce_max(logit_probs, -1, keepdims=True))

    # sample "mixture indicator" from softmax using Gumbel-max trick
    rand_sample = -tf.math.log(tf.random.uniform(list(logit_probs.shape), minval=1e-5, maxval=1. - 1e-5))
    sel = tf.argmax(logit_probs - tf.math.log(rand_sample), 3)
    sel = tf.one_hot(sel, depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])

    # select logistic parameters from the sampled mixture indicator
    means = tf.reduce_sum(means * sel, 4)
    scale = tf.maximum(tf.reduce_sum(scale * sel, 4), -7.)
    coeff = tf.reduce_sum(coeff * sel, 4)

    # sample the RGB values (before adding linear dependence)
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    sample_mu = means + scale * (tf.math.log(u) - tf.math.log(1. - u))
    mu_hat_r, mu_hat_g, mu_hat_b = tf.split(sample_mu, num_or_size_splits=3, axis=-1)

    # include the linear dependence of r->g and r,g->b
    c0, c1, c2 = tf.split(coeff, num_or_size_splits=3, axis=-1)
    x_r = tf.clip_by_value(mu_hat_r, -1.0, 1.0)
    x_g = tf.clip_by_value(mu_hat_g + c0 * x_r, -1.0, 1.0)
    x_b = tf.clip_by_value(mu_hat_b + c1 * x_r + c2 * x_g, -1.0, 1.0)

    return tf.concat([tf.reshape(x_r, xs[:-1] + [1]),
                      tf.reshape(x_g, xs[:-1] + [1]),
                      tf.reshape(x_b, xs[:-1] + [1])], 3)


def sample_from_model(model: keras.Model, shape: Tuple[int, int, int], batch_size: int) -> ndarray:
    """
    Given a Keras model

    :param model:       A trained Keras pixelCNN model
    :param shape:       Tuple (H,W,C); The shape of a single image input to the model.
    :param batch_size:  The number of samples to geenrate in parallel.

    :return: ndarray of the sample, (batch_size, shape)
    """

    x_gen = np.zeros((batch_size,) + shape, dtype=np.float32)
    print("\n")
    for yi in range(shape[0]):
        print("Sampling batch of images : {:.1f} %".format(100 * yi / shape[0]), end="\r")
        for xi in range(shape[1]):
            new_x_gen = model.predict(x_gen)
            new_x_gen = sample_from_discretized_mix_logistic(new_x_gen, 10)
            x_gen[:, yi, xi, :] = new_x_gen[:, yi, xi, :]
    print("\n")

    return x_gen
