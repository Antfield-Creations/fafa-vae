from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras


class VAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> Tuple[keras.metrics.Mean, keras.metrics.Mean, keras.metrics.Mean]:
        return (
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        )

    def __call__(self, inputs: tf.Tensor, **kwargs: dict) -> tf.Tensor:
        """
        Custom call method, allows you to directly sample reconstructions from the complete VAE model

        :param inputs: a tensor containing a batch of inputs

        :return: a batch of the reconstructions as a tensor
        """

        # Note that z_mean and z_log_var are unused here. We only use the encoder outputs to feed back
        # into the decoder in order to get the reconstruction
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        return reconstruction

    def train_step(self, data: Tensor) -> dict:
        """
        Trains a single mini-batch of tensors

        :param data:

        :return: a dictionary of typical variational auto-encoder metrics:
                    - a total loss, summed from:
                    - the reconstruction loss - how much the result diverges from the input
                    - the Kullback-Leibler loss - how much the encoder outputs diverge from a normal distribution
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
