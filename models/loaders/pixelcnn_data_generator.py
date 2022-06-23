from typing import Optional

from numpy import ndarray
from tensorflow import keras

from models.loaders.config import Config
from models.loaders.vae_data_generator import PaddingGenerator
from models.vq_vae import get_code_indices


class CodebookGenerator(PaddingGenerator):
    def __init__(self, config: Config, encoder: keras.Model, quantizer: keras.Model):
        super().__init__(config, model_name='pixelcnn')
        self.encoder = encoder
        self.quantizer = quantizer

    def __getitem__(self, index: Optional[int] = None) -> ndarray:
        """
        Fetches codes as indices from the code book for a particular batch of input images.
        These codebook codes act as both the input and reconstruction target for the pixelcnn.

        :param index:       An optional specified item to fetch from the list of available batches

        :return:            A numpy array of size (batch, encoded_rows, encoded_cols)
        """

        batch = super(CodebookGenerator, self).__getitem__(index)
        encoded_outputs = self.encoder.predict(batch)
        flattened = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
        codebook_indices = get_code_indices(self.quantizer, flattened)
        codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

        return codebook_indices
