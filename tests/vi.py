import unittest

from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import mean_field_vi
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
import random
import os
os.environ['PROGRESSBAR_MINIMUM_UPDATE_INTERVAL'] = "30"

class Test(unittest.TestCase):

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

        for epochs in [5, 15, 25, 50, 50, 50, 50]:
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
            error_msg = f"ELBO vs hat ELBO: {elbo}, {hat_elbo} (+-{error_tolerance})"
            self.assertAlmostEqual(elbo, hat_elbo, delta=error_tolerance, msg=error_msg)



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
