import unittest

from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity, evaluate_analogy
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import random
import networkx as nx

class EmbeddingTest(unittest.TestCase):

    def test_embedding_saving_1(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        saved_model_path = "test.pkl"
        e.save(saved_model_path)
        e2 = Embedding(saved_model_path=saved_model_path)

        theta1 = e.theta.numpy()
        theta2 = e2.theta.numpy()
        self.assertAlmostEqual(np.max(np.abs(theta1-theta2)), 0.0)
        self.assertEqual(e.lambda0, e2.lambda0)

    def test_embedding_saving_2(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        graph = nx.Graph()
        graph.add_edge("this", "that")
        e = LaplacianEmbedding(vocabulary=vocabulary, dimensionality=dim, graph=graph)
        saved_model_path = "test.pkl"
        e.save(saved_model_path)
        e2 = LaplacianEmbedding(saved_model_path=saved_model_path)

        theta1 = e.theta.numpy()
        theta2 = e2.theta.numpy()
        self.assertAlmostEqual(np.max(np.abs(theta1-theta2)), 0.0)
        self.assertEqual(e.lambda0, e2.lambda0)
        self.assertEqual(e.lambda1, e2.lambda1)

    def test_embedding_setting(self):
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
        words = list(vocabulary)[:10]
        new_embs = np.random.rand(10, dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        words = list(vocabulary)[-1]
        new_embs = np.random.rand(dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

    def test_evaluation(self):
        """
        Test word analogy evaluation
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        words = list(vocabulary)[:12]

        d = {"w1": words[:3], "w2": words[3:6], "w3": words[6:9], "w4": words[9:]}
        df = pd.DataFrame(d)
        d2 = pd.DataFrame([[words[9], words[9], words[9], words[9]]], columns=["w1", "w2", "w3", "w4"])
        df = pd.concat([df, d2])

        df = evaluate_analogy(e, df)
        print(df)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
