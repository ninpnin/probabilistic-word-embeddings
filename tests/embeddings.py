import unittest

import math
import tensorflow as tf
import probabilistic_word_embeddings

class EmbeddingTest(unittest.TestCase):

    # SGNS log probability. Batch size == 2
    def test_normal_embedding(self):
        vocab_size, dim = 7, 11

        embedding = NormalEmbedding(vocab_size, dim=dim)
        sample = embedding.sample()

        test_shape = sample.shape
        valid_shape = (7,11)
        self.assertEqual(test_shape, valid_shape)

        init = embedding.init()
        log_prob = embedding.log_prob(init)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
