import numpy as np
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers  # noqa
from tensorflow.keras.callbacks import History  # noqa
from tqdm import tqdm

from models.loaders.callbacks import tensorboard_callback, CustomModelCheckpointSaver
from models.loaders.config import Config
from models.loaders.data_generator import PaddingGenerator
from models.loaders.script_archive import archive_scripts
from models.pixelcnn import get_pixelcnn
from models.vq_vae import get_code_indices


def train(config: Config) -> History:
    # Load from saved model
    pxl_conf = config['models']['pixelcnn']
    vq_vae = keras.models.load_model(pxl_conf['input_vq_vae'])

    data_generator = PaddingGenerator(config)

    # Generate the codebook indices.
    encoder = vq_vae.get_layer('encoder')
    encoded_outputs = encoder.predict(next(data_generator))
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    quantizer = vq_vae.get_layer("vector_quantizer")
    codebook_indices = get_code_indices(quantizer, flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    pixel_cnn = get_pixelcnn(config)

    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pxl_conf['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    logs_folder = config['models']['vq_vae']['artifacts']['logs']['folder']
    tensorboard_cb = tensorboard_callback(artifacts_folder=logs_folder)
    checkpoint_saver = CustomModelCheckpointSaver(config, 'pixelcnn')

    history = pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        verbose=1,
        batch_size=pxl_conf['batch_size'],
        steps_per_epoch=pxl_conf['batches_per_epoch'],
        epochs=pxl_conf['epochs'],
        validation_split=0.1,
        callbacks=[tensorboard_cb, checkpoint_saver]
    )

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    x = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = dist.sample()
    sampler = keras.Model(inputs, sampled)

    priors = np.zeros(shape=(pxl_conf['batch_size'],) + pixel_cnn.input_shape[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in tqdm(range(rows)):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors, verbose=0)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    # Archive current scripts and config used for the session
    archive_scripts(config)

    return history
