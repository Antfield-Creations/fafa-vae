import numpy as np
import tensorflow as tf


def logsumexp(inputs: tf.Tensor) -> tf.Tensor:
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(inputs.get_shape()) - 1
    m = tf.reduce_max(inputs, axis)
    m2 = tf.reduce_max(inputs, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(inputs - m2), axis))


def discretized_mix_logistic_loss(inputs: tf.Tensor, logits: tf.Tensor) -> tf.float32:
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    ls = logits.shape  # predicted distribution, e.g. (B,32,32,100)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs, means, log_scales, coeffs = tf.split(logits,
                                                      num_or_size_splits=[nr_mix, 3 * nr_mix, 3 * nr_mix, 3 * nr_mix],
                                                      axis=-1)

    log_scales = tf.maximum(log_scales, -7.)
    log_scales = tf.concat(tf.split(tf.expand_dims(log_scales, -2), 3, -1), -2)
    coeffs = tf.nn.tanh(coeffs)

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    m1, m2, m3 = tf.split(means, num_or_size_splits=3, axis=-1)
    c1, c2, c3 = tf.split(coeffs, num_or_size_splits=3, axis=-1)
    x1, x2, x3 = tf.split(inputs, num_or_size_splits=3, axis=-1)

    m2 += c1 * x1
    m3 += c2 * x1 + c3 * x2

    means = tf.concat([tf.expand_dims(m1, axis=-2),
                       tf.expand_dims(m2, axis=-2),
                       tf.expand_dims(m3, axis=-2)], axis=-2)

    inputs = tf.expand_dims(inputs, -1)
    x_c = tf.subtract(inputs, means)

    inv_stdv = tf.exp(-log_scales)

    plus_in = inv_stdv * (x_c + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)

    min_in = inv_stdv * (x_c - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case o
    # f 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * x_c
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(
        inputs < -0.999, log_cdf_plus, tf.where(
            inputs > 0.999, log_one_minus_cdf_min, tf.where(
                cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))
        )
    )

    # sum log probs ==> multiply the probs
    log_probs = tf.reduce_sum(log_probs, 3)
    log_probs += tf.nn.log_softmax(logit_probs - tf.reduce_max(logit_probs, -1, keepdims=True))
    loss = -tf.reduce_sum(logsumexp(log_probs))

    n = tf.cast(tf.size(inputs), tf.float32)
    return tf.cast(loss, tf.float32) / (n * np.log(2))
