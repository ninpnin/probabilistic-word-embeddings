import unittest

from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
import tensorflow as tf
import numpy as np
from pathlib import Path

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

    def test_dynamic_map(self):
        paths = sorted(list(Path("tests/data/").glob("*.txt")))
        names, texts = [], []
        for p in paths:
            names.append(p.stem)
            with p.open() as f:
                t = f.read().lower().split()
                texts.append(t)

        texts, vocabulary = preprocess_standard(texts)

        def append_suffix(text, suffix):
            return [wd + "_" + suffix for wd in text]
        texts = [append_suffix(text) for text in texts]

        print(texts[0][:10])
        print(texts[1][:10])

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        #e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        #e = map_estimate(e, text, evaluate=False, epochs=1)
        #theta = e.theta.numpy()
        #self.assertEqual(type(theta), np.ndarray)

        #theta_shape = theta.shape
        #valid_shape = (vocab_size * 2 * len(texts), dim)
        #self.assertEqual(theta_shape, valid_shape)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
