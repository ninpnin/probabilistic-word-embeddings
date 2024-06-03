import unittest

from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate, mean_field_vi
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
import random
import os
import copy
os.environ['PROGRESSBAR_MINIMUM_UPDATE_INTERVAL'] = "30"

class Test(unittest.TestCase):

    def test_vi_generator(self):
        """
        Test VI estimation with data_generator
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
        q_mu, q_std_e, elbo_history = mean_field_vi(e, text, data_generator=data_generator(), N=N, model="sgns", evaluate=False, batch_size=batch_size, init_mean=False, epochs=1, elbo_history=True)
        theta = q_mu.theta.numpy()
        self.assertNotEqual(theta_orig[0,0], theta[0,0])

        self.assertEqual(type(theta), np.ndarray)

        theta_shape = theta.shape
        valid_shape = (vocab_size * 2, dim)
        self.assertEqual(theta_shape, valid_shape)

        K = 23
        words = random.choices(text, k=K)
        embeddings = e[words]
        self.assertEqual(embeddings.shape, (K, dim))


    def test_vi(self):
        """
        Test mean-field variational inference with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 25
        ws = 3
        dim = 2

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        init_mean = False
        init_std = 0.2
        q_mu, q_std_e = mean_field_vi(e, text, model="cbow", evaluate=False, ws=ws, batch_size=batch_size, init_mean=init_mean, init_std=init_std, epochs=5)

    def test_vi_convergence(self):
        """
        Test mean-field variational inference with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 25
        ws = 3
        dim = 2

        text_full = []
        for i in range(100):
            text_full += text

        e0 = Embedding(vocabulary=vocabulary, dimensionality=dim)
        init_mean = False
        init_std = 0.2
        q_mu0, q_std_e0 = mean_field_vi(e, text, model="sgns", evaluate=False, ws=ws, batch_size=batch_size, init_mean=init_mean, init_std=init_std, epochs=5)

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        q_mu, q_std_e = mean_field_vi(e, text_full, model="sgns", evaluate=False, ws=ws, batch_size=batch_size, init_mean=init_mean, init_std=init_std, epochs=5)
        self.assertGreater(np.mean(q_std_e.theta), np.mean(q_std_e0.theta)) 


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
