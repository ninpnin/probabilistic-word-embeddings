import unittest
import math
import tensorflow as tf

def sigmoid(x):
    denominator = 1.0 + math.exp(-x)
    return 1.0 / denominator

class Test(unittest.TestCase):

    # SGNS log probability. Batch size == 2
    def test_sgns_1(self):
        # Set hyperparameters
            lambda0 = 1.0   # Prior stdev of the embedding vectors
            V = 10          # Vocabulary size, number of word types
            D = 25          # Dimensionality of the embedding vectors
            batch_size = 1  # Number of data points

            # Set variable values
            tf.random.set_seed(103)
            rho =   tf.random.normal((V, D), mean=0.0)
            alpha = tf.random.normal((V, D), mean=0.0)
            i = [1]
            j = [3]
            x = [0]

            # Probability of the indices i and j, categorical
            log_ij = - math.log(V) * 2 * batch_size

            # Prior probability of rho, alpha
            rho_sum = tf.reduce_sum(rho * rho)
            alpha_sum = tf.reduce_sum(alpha * alpha)
            log_prior = - 0.5 * lambda0 * (rho_sum + alpha_sum) - 0.5 * D * math.log(math.pi * 2) * V * 2

            # Likelihood of x==0 given rho,alpha,i,j
            eta_1 = - tf.reduce_sum(tf.multiply(rho[i[0]], alpha[j[0]]))
            log_likelihood = math.log(sigmoid(eta_1))

            # Total validation log probability
            valid_log_prob = log_prior + log_likelihood + log_ij

            # Model log probability
            embedding = NormalEmbedding(vocab_size=2*V, lambda0=lambda0, dim=D)
            sgns = SGNS(embedding, batch_size=1, ws=1, ns=0)
            theta = tf.cast(tf.concat([rho,alpha],axis=0), dtype=tf.float32)
            
            y = dict(
                theta=theta,
                i=i,
                j=j,
                x=x
            )
            
            model_prior = sgns.prior.log_prob(theta).numpy()
            model_ll = sgns.likelihood.log_prob(y).numpy()

            model_log_prob = model_ll + model_prior
            
            # Check that the two are equal. Tf default precision is low, thus delta=0.01
            self.assertAlmostEqual(valid_log_prob, model_log_prob, delta=0.01)

    def test_cbow_1(self):
        # Set hyperparameters
        lambda0 = 1.0   # Prior stdev of the embedding vectors
        V = 10          # Vocabulary size, number of word types
        D = 25          # Dimensionality of the embedding vectors
        batch_size = 1  # Number of data points
        ws = 1          # Window size
        ns = 0          # Number of negative samples

        # Set variable values
        tf.random.set_seed(103)
        rho =   tf.random.normal((V, D), mean=0.0)
        alpha = tf.random.normal((V, D), mean=0.0)
        i = tf.constant([1])
        j = tf.constant([[3, 1]])
        x = tf.constant([0])

        # Probability of the indices i and j, categorical
        log_ij = - math.log(V) * (1 + 2 * ws) * batch_size

        # Prior probability of rho, alpha
        rho_sum = tf.reduce_sum(rho * rho)
        alpha_sum = tf.reduce_sum(alpha * alpha)
        log_prior = - 0.5 * lambda0 * (rho_sum + alpha_sum) - 0.5 * D * math.log(math.pi * 2) * V * 2

        # Likelihood of x==0 given rho,alpha,i,j
        alpha_1 = tf.gather(alpha, j[0])
        alpha_1 = tf.reduce_sum(alpha_1, axis=0)
        rho_1 = tf.gather(rho, i[0])
        eta_1 = - tf.reduce_sum(tf.multiply(rho_1, alpha_1))
        log_likelihood = math.log(sigmoid(eta_1))# + math.log(sigmoid(eta_2))

        # Total validation log probability
        valid_log_prob = log_prior + log_likelihood + log_ij

        # Model log probability
        embedding = NormalEmbedding(vocab_size=2*V, lambda0=lambda0, dim=D)
        cbow = CBOW(embedding, ws=ws, ns=0, batch_size=batch_size)

        theta = tf.cast(tf.concat([rho,alpha],axis=0), dtype=tf.float32)
        y = dict(
            theta=theta,
            i=i,
            j=j,
            x=x
        )

        x = cbow.likelihood.sample()
        model_prior = cbow.prior.log_prob(theta).numpy()
        model_ll = cbow.likelihood.log_prob(y).numpy()

        model_log_prob = model_ll + model_prior

        # Check that the two are equal. Tf default precision is low, thus delta=0.01
        self.assertAlmostEqual(valid_log_prob, model_log_prob, delta=0.01)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
