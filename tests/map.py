import unittest

from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate, mean_field_vi
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set, bli
import tensorflow as tf
import tensorflow_probability as tfp
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
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, model="sgns", evaluate=False, batch_size=batch_size, epochs=1)
        theta = e.theta.numpy()

        self.assertEqual(type(theta), np.ndarray)
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_map_training_loss(self):
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
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, evaluate=False, batch_size=batch_size, epochs=5, training_loss=True)
        theta = e.theta.numpy()

        self.assertEqual(type(theta), np.ndarray)
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_map_cbow(self):
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
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, model="cbow", evaluate=False, batch_size=batch_size, epochs=1)
        theta = e.theta.numpy()

        self.assertEqual(type(theta), np.ndarray)
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_map_with_generator(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        N = len(text)
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        def data_generator():
            while True:
                i = np.random.choice(text, batch_size)
                j = np.random.choice(text, batch_size)
                x = np.random.binomial(1, 0.25, size=batch_size)

                i = tf.constant(i)
                j = tf.constant(j)
                x = tf.constant(x, dtype=tf.float64)
                yield (i,j,x)

        theta_orig = e.theta.numpy()
        e = map_estimate(e, data_generator=data_generator(), N=N, evaluate=False, model="sgns", epochs=10, batch_size=batch_size)
        theta = e.theta.numpy()
        self.assertNotEqual(theta_orig[0,0], theta[0,0])

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
        g.add_edge("this", "that")

        e = LaplacianEmbedding(vocabulary=vocabulary, dimensionality=dim, graph=g)
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, evaluate=False, batch_size=batch_size, epochs=1)
        theta = e.theta.numpy()
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)

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
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, evaluate=False, batch_size=batch_size, epochs=1)
        theta = e.theta.numpy()
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)
        self.assertEqual(type(theta), np.ndarray)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))

    def test_crosslingual_map(self):
        paths = sorted(list(Path("tests/data/crosslingual/").glob("*.txt")))
        names, texts = [], []
        for p in paths:
            names.append(p.stem)
            with p.open(errors="ignore") as f:
                t = f.read().lower().split()
                texts.append(t)

        texts, vocabulary = preprocess_partitioned(texts, names)
        text = []
        for t in texts:
            text += t

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        
        e = Embedding(vocabulary=vocabulary, dimensionality=dim, shared_context_vectors=False)
        theta_before = e.theta.numpy()
        e = map_estimate(e, text, evaluate=False, batch_size=batch_size, epochs=1)
        theta = e.theta.numpy()
        self.assertNotEqual(tf.reduce_sum(theta-theta_before), 0.0)
        self.assertEqual(type(theta), np.ndarray)

        K = 23
        for i in range(len(texts)):
            words = random.choices(texts[i], k=K)
            embeddings = e[words]
            self.assertEqual(embeddings.shape, (K, dim))

        pairs = [("cat_en", "gatto_it"), ("man_en", "uomo_it"), ("day_en", "giorno_it")]
        bli(pairs, e)

    def test_holdout(self):
        """
        Test evaluation of an embedding on a holdout set
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        dim = 25
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        test_ll = evaluate_on_holdout_set(e, text, model="cbow", ws=5, batch_size=len(text))
        self.assertLess(test_ll, 0.0)
        test_ll = evaluate_on_holdout_set(e, text, model="sgns", ws=5, batch_size=len(text))
        self.assertLess(test_ll, 0.0)

    def test_holdout_accuracy(self):
        """
        Test evaluation of an embedding on a holdout set
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        dim = 25
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        test_acc = evaluate_on_holdout_set(e, text, model="cbow", ws=5, batch_size=len(text), metric="accuracy")
        print(f"Test accuracy: {test_acc}")
        self.assertLess(0.0, test_acc)
        self.assertLess(test_acc, 1.0)

    def test_keep_words(self):
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        _, vocabulary1 = preprocess_standard(text, downsample=False)
        _, vocabulary2 = preprocess_standard(text, downsample=False, keep_words={"hasty"})

        self.assertNotIn("hasty", vocabulary1)
        self.assertIn("hasty", vocabulary2)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
