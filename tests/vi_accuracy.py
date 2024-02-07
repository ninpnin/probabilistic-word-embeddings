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

    def test_vi_accuracy(self):
        """
        Test that the results of mean-field variational inference are coherent
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 25
        ws = 3
        dim = 2
        estimated_elbos = []
        true_elbos = []
        error_tolerances = []

        e_map = Embedding(vocabulary=vocabulary, dimensionality=dim)
        e_map = map_estimate(e_map, text, model="cbow", evaluate=False, ws=ws, batch_size=batch_size, epochs=20, training_loss=True)

        for epochs in [5, 15, 25, 25, 25, 25]:
            e = copy.deepcopy(e_map)
            init_mean = False
            init_std = 0.2
            q_mu, q_std_e, elbo_history = mean_field_vi(e, text, model="cbow", evaluate=False, ws=ws, batch_size=batch_size, init_mean=init_mean, init_std=init_std, epochs=epochs, elbo_history=True)

            q_mean = q_mu.theta.numpy()
            q_std = q_std_e.theta.numpy()

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
            print("E[log p(x| Q)]", expected_posterior.numpy(), "E[log(Q)]", entropy.numpy())
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
            stochastic_vi_deviation = np.std(elbo_history[-4:])
            stochastic_vi_deviation_prime = np.std(elbo_history[-4:] - np.array(range(4)) * optimization_deviation )
            print("Corrected v original", stochastic_vi_deviation_prime, stochastic_vi_deviation)
            if stochastic_vi_deviation_prime < stochastic_vi_deviation:
                stochastic_vi_deviation = stochastic_vi_deviation_prime
                print("Use linear correction for stochastic_vi_deviation")

            # Error due to stochasticity in the calculation of the true posterior
            # error = standard error of the mean of the posteriors
            posterior_estimate_deviation = np.std(posteriors) / np.sqrt(rounds)
            print("optimization_deviation", optimization_deviation)
            print("stochastic_vi_deviation", stochastic_vi_deviation)
            print("posterior_estimate_deviation", posterior_estimate_deviation)
            error_tolerance = optimization_deviation + stochastic_vi_deviation * 3.0 + posterior_estimate_deviation * 3.0
            print("error_tolerance", error_tolerance)
            error_tolerances.append(error_tolerance)

        # Correct for constant offsets
        constant_offset = np.array(estimated_elbos)[-3:] - np.array(true_elbos)[-3:]

        # Standard error for the normalizing constant
        constant_error = np.std(constant_offset) / np.sqrt(3) * 3.0
        constant_offset = tf.reduce_mean(constant_offset)
        print("constant_error", constant_error)
        estimated_elbos = [elbo - constant_offset for elbo in estimated_elbos]

        for elbo, hat_elbo, error_tolerance in zip(true_elbos, estimated_elbos, error_tolerances):
            error_tolerance += constant_error
            print(elbo, "vs", hat_elbo, "diff", elbo - hat_elbo)
            error_msg = f"ELBO vs hat ELBO: {elbo}, {hat_elbo} (+-{error_tolerance})"
            self.assertAlmostEqual(elbo, hat_elbo, delta=error_tolerance, msg=error_msg)



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
