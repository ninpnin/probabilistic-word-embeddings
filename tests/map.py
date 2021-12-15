import unittest

import math
import os
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.embeddings import NormalEmbedding
from probabilistic_word_embeddings.models import CBOW
import tensorflow as tf
import numpy as np

class Test(unittest.TestCase):

    # Test MAP estimation with random data
    def test_map(self):
        V = 17
        D = 25
        data_len = 10000
        batch_size = 250

        prior = NormalEmbedding(V * 2, dim=D)
        model = CBOW(prior, batch_size=batch_size)
        model.data_size = data_len

        data = np.random.randint(V, size=data_len)
        data = [data]

        theta = map_estimate(model, data, epochs=1)

        self.assertEqual(type(theta), np.ndarray)

        theta_shape = theta.shape
        valid_shape = (V * 2, D)
        self.assertEqual(theta_shape, valid_shape)

    # Test MAP estimation with random data
    def test_map_history(self):
        V = 17
        D = 25
        data_len = 10000
        batch_size = 250

        prior = NormalEmbedding(V * 2, dim=D)
        model = CBOW(prior, batch_size=batch_size)
        model.data_size = data_len

        data = np.random.randint(V, size=data_len)
        data = [data]

        theta_history = map_estimate(model, data, epochs=1, history=True)

        self.assertEqual(type(theta_history), list)

        theta = theta_history[0]
        theta_shape = theta.shape
        valid_shape = (V * 2, D)
        self.assertEqual(theta_shape, valid_shape)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
