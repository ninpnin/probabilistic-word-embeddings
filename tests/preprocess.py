import unittest

import math
import os
from probabilistic_word_embeddings import preprocess
import numpy as np

class Test(unittest.TestCase):

    # SGNS log probability. Batch size == 2
    def test_preprocessing(self):
        # Reads all files from datapath.
        datapath = "tests/data/"
        data, vocab, vocab_size = preprocess(datapath)
        data = data[0]

        # Smallest word index should be 0 both in the data and in the vocab
        self.assertEqual(0, min(vocab.values()))
        self.assertEqual(0, np.min(data))

        # Correct vocab size
        self.assertEqual(vocab_size, len(vocab.values()))
        self.assertEqual(vocab_size - 1, np.max(data))

        # Vocab is a bijection
        self.assertEqual(len(vocab.keys()), len(vocab.values()))

    # SGNS log probability. Batch size == 2
    def test_dynamic_preprocessing(self):
        # Reads all files from datapath.
        datapath = "tests/data/"
        data, vocab, vocab_size = preprocess(datapath, data_type="dynamic")
        data = data[0]

        self.assertEqual(0, min(vocab.values()))
        self.assertEqual(0, np.min(data))

        self.assertEqual(vocab_size, len(vocab.values()))
        self.assertEqual(vocab_size - 1, np.max(data))

        # Vocab is a bijection
        self.assertEqual(len(vocab.keys()), len(vocab.values()))



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
