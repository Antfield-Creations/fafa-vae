"""
Vector-quantized layer, adapted only for stricter typing from https://keras.io/examples/generative/vq_vae/
"""
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from models.decoder import get_decoder
from models.encoder import get_encoder
from models.loaders.config import Config


def get_vq_vae(config: Config) -> keras.Model:
    vq_vae_conf = config['models']['vq_vae']
    if vq_vae_conf['artifacts']['resume_model'] is not None:
        return keras.models.load_model(vq_vae_conf['artifacts']['resume_model'])

    vq_layer = VectorQuantizer(
        num_embeddings=vq_vae_conf['num_embeddings'],
        embedding_dim=vq_vae_conf['latent_size'])
    encoder = get_encoder(config)
    decoder = get_decoder(config)

    img_cfg = config['data']['images']
    inputs = keras.Input(shape=(img_cfg['height'], img_cfg['width'], img_cfg['channels']))
    encoder_outputs = encoder(inputs)

    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)

    return keras.Model(inputs, reconstructions, name="vq_vae")


class VectorQuantizer(layers.Layer):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 name: str = 'vector_quantizer',
                 **kwargs: dict) -> None:
        super().__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # This parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vq_vae",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs: tf.Tensor) -> tf.Tensor:
        return get_code_indices(self, flattened_inputs)


def get_code_indices(vector_quantizer: VectorQuantizer, flattened_inputs: tf.Tensor) -> tf.Tensor:
    """
    This basically extracts the `get_code_indices` method on the VectorQuantizer class itself, so that you can
    use this in a much more simple way from a tf.keras.load_model(path) and not have to deal with all the "custom
    layers" issues that arise from this.

    :param vector_quantizer: A trained VectorQuantizer layer
    :param flattened_inputs: A sample set of images, flattened into shape (batch, pixels, channels)

    :return: A vector of indices from the code book
    """

    # Calculate L2-normalized distance between the inputs and the codes.
    similarity = tf.matmul(flattened_inputs, vector_quantizer.embeddings)
    distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(vector_quantizer.embeddings ** 2, axis=0)
            - 2 * similarity
    )

    # Derive the indices for minimum distances.
    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices


class VQVAETrainer(keras.models.Model):
    def __init__(self, config: Config, **kwargs: dict) -> None:
        super(VQVAETrainer, self).__init__(**kwargs)
        if 'train_variance' not in config['models']['vq_vae']:
            raise ValueError('Vector-quantized variational auto-encoders require a variance setting.\n'
                             'Please pass a config dict having `models.vq_vae.train_variance.')
        self.train_variance = config['models']['vq_vae']['train_variance']
        self.latent_dim = config['models']['vq_vae']['latent_size']
        self.num_embeddings = config['models']['vq_vae']['num_embeddings']

        self.vq_vae = get_vq_vae(config)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self) -> Tuple[keras.metrics.Mean, keras.metrics.Mean, keras.metrics.Mean]:
        return (
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        )

    def train_step(self, x: tf.Tensor) -> dict:
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vq_vae(x)

            # Calculate the losses.
            reconstruction_loss = (
                    tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vq_vae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vq_vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vq_vae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vq_vae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_vae_loss": self.vq_loss_tracker.result(),
        }

    def __call__(self, inputs: tf.Tensor, **kwargs: dict) -> tf.Tensor:
        """
        Custom call method, allows you to directly sample reconstructions from the complete VQ-VAE model

        :param inputs: a tensor containing a batch of inputs

        :return: a batch of the reconstructions as a tensor
        """

        return self.vq_vae(inputs)
