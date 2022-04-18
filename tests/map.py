import unittest

from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
import tensorflow as tf
import numpy as np

class Test(unittest.TestCase):

    # Test MAP estimation with random data
    def test_map(self):
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        e = map_estimate(e, text, evaluate=False, epochs=1)
        theta = e.theta.numpy()

        self.assertEqual(type(theta), np.ndarray)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
