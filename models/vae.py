from typing import Tuple, Dict

import keras
import tensorflow as tf
from keras.metrics import Mean, Reduce
from tensorflow import Tensor


class VAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> Tuple[Mean, Mean, Mean]:
        return (
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        )

    def train_step(self, data: Tensor) -> Dict[str, Reduce]:
        """
        Trains a single mini-batch of tensors

        :param data:
        :return: a dictionary of typical variational auto-encoder metrics:
                    - a run-of-the-mill total loss
                    - the reconstruction loss - how much the result diverges from the input
                    - the Kullback-Leibler loss - how much the encoder outputs diverge from a normal distribution
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
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