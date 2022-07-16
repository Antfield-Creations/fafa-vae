import unittest

import numpy as np

from models.pixelcnn import PixelConvLayer


class PixelCNNTestCase(unittest.TestCase):
    def test_mask_type_A(self) -> None:
        kernel_shape = (5, 5, 3)
        mask = PixelConvLayer.generate_mask(kernel_shape, "A")

        with self.subTest(f'From a {kernel_shape=}, all the row values up to the middle row are ones'):
            self.assertListEqual(mask.numpy()[:2, ...].tolist(), np.ones(shape=[2, 5, 3]).tolist())  # type: ignore

        with self.subTest(f'From a {kernel_shape=}, all the row values from the middle row + 1 onwards are zeros'):
            self.assertListEqual(mask.numpy()[3:, ...].tolist(), np.zeros(shape=[2, 5, 3]).tolist())  # type: ignore

        with self.subTest(f'From a {kernel_shape=}, all the middle row values up to the center are ones'):
            self.assertListEqual(mask.numpy()[2, :2, ...].tolist(), np.ones(shape=[2, 3]).tolist())  # type: ignore
        with self.subTest(f'From a {kernel_shape=}, all the middle row values up from the center onwards are zeros'):
            self.assertListEqual(mask.numpy()[2, 2:, ...].tolist(), np.zeros(shape=[3, 3]).tolist())  # type: ignore

    def test_mask_type_B(self) -> None:
        kernel_shape = (5, 5, 3)
        mask = PixelConvLayer.generate_mask(kernel_shape, "B")

        with self.subTest(f'From a {kernel_shape=}, the center pixel is a one'):
            self.assertListEqual(mask.numpy()[2, 2, :].tolist(), np.ones(shape=(3,)).tolist())  # type: ignore

        kernel_shape = (1, 1, 3)
        mask = PixelConvLayer.generate_mask(kernel_shape, "B")

        with self.subTest(f'From a {kernel_shape=}, the center pixel is a one'):
            self.assertListEqual(mask.numpy()[0, 0, :].tolist(), np.ones(shape=(3,)).tolist())  # type: ignore
