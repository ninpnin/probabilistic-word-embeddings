import unittest

from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
import tensorflow as tf
import numpy as np
from pathlib import Path
import random
import networkx as nx

class Test(unittest.TestCase):

    def test_map(self):
        """
        Test MAP estimation with example dataset
        """
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

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_map_laplacian(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        g = nx.Graph()
        g.add_edge("dog", "dogs")
        g.add_edge("cat", "cats")
        g.add_edge("cat", "dog")

        e = LaplacianEmbedding(vocabulary=vocabulary, dimensionality=dim, graph=g)
        e = map_estimate(e, text, evaluate=False, epochs=1)
        theta = e.theta.numpy()

        self.assertEqual(type(theta), np.ndarray)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_dynamic_map(self):
        paths = sorted(list(Path("tests/data/").glob("*.txt")))
        names, texts = [], []
        for p in paths:
            names.append(p.stem)
            with p.open() as f:
                t = f.read().lower().split()
                texts.append(t)

        texts, vocabulary = preprocess_partitioned(texts, names)
        text = []
        for t in texts:
            text = text + t

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        e = map_estimate(e, text, evaluate=False, epochs=1)
        theta = e.theta.numpy()
        self.assertEqual(type(theta), np.ndarray)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
