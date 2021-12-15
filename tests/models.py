import unittest

import math
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probabilistic_word_embeddings.models import CBOW, SGNS, DynamicCBOW, DynamicSGNS
from probabilistic_word_embeddings.embeddings import NormalEmbedding, LaplacianEmbedding, DynamicNormalEmbedding
from probabilistic_word_embeddings.embeddings import concat_embeddings

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

    def test_dynamic_sgns_1(self):
        

        #model = init_dynamic(5, 2)
        #sample = model.sample()
        #sample_shape = sample.shape
        #print("Sample shape", sample_shape)

        #valid_log_prob = -123.4
        #model_log_prob = model.log_prob(sample).numpy()
        pass
        #self.assertAlmostEqual(valid_log_prob, model_log_prob, delta=0.01)


    # Dynamic CBOW with 1 timestep should be equal to static CBOW
    def test_dynamic_cbow_1(self):     
        T, V, D = 1, 5, 3
        batch_size = 11
        ws, ns = 2, 2
        lambda0 = 1.0

        dynamic_rho = DynamicNormalEmbedding(V, T, scale=lambda0, dim=D)
        dynamic_alpha = DynamicNormalEmbedding(V, 1, scale=lambda0, dim=D)
        dynamic_embedding = concat_embeddings([dynamic_rho, dynamic_alpha], axis=0)
        model1 = DynamicCBOW(dynamic_embedding, ws=ws, ns=ns, batch_size=batch_size)

        embedding = NormalEmbedding(vocab_size=2*V, lambda0=lambda0, dim=D)
        default = CBOW(embedding, ws=ws, ns=ns, batch_size=batch_size)

        def convert(sample):
            rhos = sample["theta"][0]
            alphas = sample["theta"][1]
            theta = tf.concat([rhos, alphas], axis=0)

            sample["theta"] = theta
            sample.pop("t")
            sample["j"] = tf.transpose(sample["j"])
            return sample

        def posterior(model, sample):
            samplex = model.likelihood.sample()
            ll = model.likelihood.log_prob(sample)
            prior = model.prior.log_prob(sample["theta"])
            return ll + prior

        sample1 = model1.likelihood.sample()

        model_log_prob1 = posterior(model1, sample1)
        sample1c = convert(sample1)
        valid_log_prob1 = posterior(default, sample1c)

        sample2 = model1.likelihood.sample()
        model_log_prob2 = posterior(model1, sample2)
        valid_log_prob2 = posterior(default, convert(sample2))
        
        # Use the difference between two points to account for differences in constants
        model_diff = model_log_prob1 - model_log_prob2
        valid_diff = valid_log_prob1 - valid_log_prob2

        self.assertAlmostEqual(model_diff, valid_diff, delta=0.5)    
    
    # Laplacian CBOW with no edges should be equal to static CBOW
    def test_laplacian_cbow_1(self):
        V, lambda0, D = 10, 1.0, 7

        embedding = NormalEmbedding(vocab_size=2*V, lambda0=lambda0, dim=D)
        default_model = CBOW(embedding)

        laplacian_embedding = LaplacianEmbedding(vocab_size=2*V, lambda0=lambda0, dim=D)
        laplacian_model = CBOW(laplacian_embedding)

        sample1 = default_model.likelihood.sample()
        sample2 = default_model.likelihood.sample()

        def posterior(model, sample):
            ll = model.likelihood.log_prob(sample)
            prior = model.prior.log_prob(sample["theta"])
            return ll + prior

        # Use the difference between two points to account for differences in constants
        model_log_prob = posterior(laplacian_model, sample1) - posterior(laplacian_model, sample2)
        valid_log_prob = posterior(default_model, sample1)   - posterior(default_model, sample2)
        
        self.assertAlmostEqual(valid_log_prob, model_log_prob, delta=0.5)
        
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
