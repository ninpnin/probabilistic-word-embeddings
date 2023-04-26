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

    def test_map_with_freqs(self):
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
        e = map_estimate(e, text, evaluate=False, epochs=1, batch_size=batch_size, vocab_freqs=vocabulary)
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

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        
        e = Embedding(vocabulary=vocabulary, dimensionality=dim, shared_context_vectors=False)
        theta_before = e.theta.numpy()
        e = map_estimate(e, texts, evaluate=False, batch_size=batch_size, epochs=1)
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

    def test_keep_words(self):
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        _, vocabulary1 = preprocess_standard(text, downsample=False)
        _, vocabulary2 = preprocess_standard(text, downsample=False, keep_words={"hasty"})

        self.assertNotIn("hasty", vocabulary1)
        self.assertIn("hasty", vocabulary2)

    def test_vi(self):
        """
        Test mean-field variational inference with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 25
        ws = 5
        dim = 2

        estimated_elbos = []
        true_elbos = []
        error_tolerances = []

        for epochs in [5, 15, 25, 50, 50, 50]:
            e = Embedding(vocabulary=vocabulary, dimensionality=dim)
            q_mu, q_std_log, elbo_history = mean_field_vi(e, text, model="cbow", evaluate=False, ws=ws, batch_size=batch_size, epochs=epochs, elbo_history=True)

            q_mean = q_mu.theta.numpy()
            q_std = tf.exp(q_std_log).numpy()

            rounds = len(vocabulary) // batch_size

            # Calculate true expected value of the posterior
            posteriors = []
            for _ in range(rounds):
                epsilon = tf.random.normal(q_std.shape, dtype=tf.float64)
                z = q_mean + tf.multiply(q_std, epsilon)
                e.theta.assign(z)
                ll = evaluate_on_holdout_set(q_mu, text, model="cbow", ws=ws, batch_size=len(text), reduce_mean=False)
                posterior = tf.reduce_sum(ll).numpy() + e.log_prob(1, 1)
                posteriors.append(posterior)
            expected_posterior = tf.reduce_mean(posteriors)

            # Calculate true entropy
            q_flat_std = tf.reshape(q_std,[-1])
            dist = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(q_flat_std.shape, dtype=q_flat_std.dtype), scale_diag=q_flat_std)
            entropy = dist.entropy()

            # Calculate true and estimated ELBO
            estimated_elbo = elbo_history[-1]
            elbo = expected_posterior + entropy
            estimated_elbos.append(estimated_elbo)
            true_elbos.append(elbo)

            # Error due to ELBO improvement during the epoch
            # error = avg. slope during past 4 epochs
            optimization_deviation = (elbo_history[-1] - elbo_history[-4]) / 3.0
            if optimization_deviation < 0.0:
                optimization_deviation = 0.0

            # Error due to ELBO improvement during the epoch
            # error = stdev of hat ELBO * 2
            stochastic_vi_deviation = np.std(elbo_history[-4:]) * 3.0
            stochastic_vi_deviation_prime = np.std(elbo_history[-4:] - np.array(range(4)) * optimization_deviation ) * 2.0
            print("Corrected v original", stochastic_vi_deviation_prime, stochastic_vi_deviation)
            if stochastic_vi_deviation_prime < stochastic_vi_deviation:
                stochastic_vi_deviation = stochastic_vi_deviation_prime
                print("Use linear correction for stochastic_vi_deviation")

            # Error due to stochasticity in the calculation of the true posterior
            # error = standard error of the mean of the posteriors
            posterior_estimate_deviation = np.std(posteriors) / np.sqrt(rounds) * 3.0
            print("optimization_deviation", optimization_deviation)
            print("stochastic_vi_deviation", stochastic_vi_deviation)
            print("posterior_estimate_deviation", posterior_estimate_deviation)
            error_tolerances.append(optimization_deviation + stochastic_vi_deviation + posterior_estimate_deviation)

        # Correct for constant offsets
        constant_offset = tf.reduce_mean(estimated_elbos[-3:]) - tf.reduce_mean(true_elbos[-3:])
        estimated_elbos = [elbo - constant_offset for elbo in estimated_elbos]

        for elbo, hat_elbo, error_tolerance in zip(true_elbos, estimated_elbos, error_tolerances):
            print("ELBO vs hat ELBO", elbo, hat_elbo, f"(+-{error_tolerance})")
            self.assertAlmostEqual(elbo, hat_elbo, delta=error_tolerance)



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
