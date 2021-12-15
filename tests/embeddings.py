import unittest

import math
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probabilistic_word_embeddings
from probabilistic_word_embeddings.embeddings import NormalEmbedding, LaplacianEmbedding, DynamicNormalEmbedding, DynamicInformativeEmbedding, concat_embeddings

class EmbeddingTest(unittest.TestCase):

    # SGNS log probability. Batch size == 2
    def test_normal_embedding(self):
        vocab_size, dim = 7, 11

        embedding = NormalEmbedding(vocab_size, dim=dim)
        sample = embedding.sample()

        test_shape = sample.shape
        valid_shape = (7,11)
        self.assertEqual(test_shape, valid_shape)

        init = embedding.init()
        log_prob = embedding.log_prob(init)

    def test_laplacian_embedding(self):
        vocab_size, dim = 7, 11
        
        laplacian = tf.sparse.SparseTensor(indices=[[1,0], [0,1], [0,0], [1,1]], values=[-1., -1., 1., 1.], dense_shape=[vocab_size, vocab_size])

        embedding = LaplacianEmbedding(vocab_size, laplacian, dim=dim)
        sample = embedding.sample()

        test_shape = sample.shape
        valid_shape = (7,11)
        self.assertEqual(test_shape, valid_shape)

        init = embedding.init()
        log_prob = embedding.log_prob(init)

    def test_dynamic_normal_embedding(self):
        vocab_size, dim, timesteps = 7, 11, 2
        
        embedding = DynamicNormalEmbedding(vocab_size, timesteps, dim=dim)
        sample = embedding.sample()
        
        test_shape = sample.shape
        valid_shape = (2, 7, 11)
        self.assertEqual(test_shape, valid_shape)

        init = embedding.init()
        log_prob = embedding.log_prob(init)

    def test_dynamic_informative_embedding(self):
        vocab_size, dim, timesteps = 7, 11, 2

        si = {2: -1.0, 3: 1.0} # Side-information
        embedding = DynamicInformativeEmbedding(vocab_size, timesteps, si=si, dim=dim)
        sample = embedding.sample()
        
        test_shape = sample.shape
        valid_shape = (timesteps, vocab_size, dim)
        self.assertEqual(test_shape, valid_shape)

        sample = embedding.sample()
        sample_val1 = sample[0, 2, -1]
        sample_val2 = sample[0, 3, -1]
        self.assertAlmostEqual(sample_val1, -1.0, delta=0.1)
        self.assertAlmostEqual(sample_val2, 1.0, delta=0.1)

        init = embedding.init()
        init_val1 = init[0, 2, -1]
        init_val2 = init[0, 3, -1]
        self.assertAlmostEqual(init_val1, -1.0, delta=0.1)
        #self.assertAlmostEqual(init_val2, 1.0, delta=0.1)
    
    def test_concatenation_1(self):
        vocab_size, dim = 5, 1
        mean1, mean2 = 0.0, 1.0
        scale = 0.01
        
        embedding_1 = NormalEmbedding(vocab_size, dim=dim, lambda0=scale, mean=mean1)
        embedding_2 = NormalEmbedding(vocab_size, dim=dim, lambda0=scale, mean=mean2)

        joint = concat_embeddings([embedding_1, embedding_2], axis=1)
        
        sample = joint.sample()
        
        test_shape = sample.shape
        valid_shape = (5, 2)
        self.assertEqual(test_shape, valid_shape)
    
    def test_concatenation_2(self):
        vocab_size, dim = 5, 3
        mean1, mean2 = 0.0, 1.0
        scale = 0.01
        
        rhos = NormalEmbedding(vocab_size, dim=dim, mean=mean1)
        alphas = NormalEmbedding(vocab_size, dim=dim, mean=mean2)
        
        theta = concat_embeddings([rhos, alphas], axis=0)
        sample = theta.sample()
        
        test_shape = sample.shape
        valid_shape = (10, 3)
        self.assertEqual(test_shape, valid_shape)
        
    def test_concatenation_3(self):
        timesteps, vocab_size, dim = 2, 5, 3

        alphas = DynamicNormalEmbedding(vocab_size, timesteps, dim=dim)
        rhos = DynamicNormalEmbedding(vocab_size, 1, dim=dim)

        theta = concat_embeddings([rhos, alphas], axis=0)
        
        sample = theta.sample()
        
        test_shape = sample.shape
        valid_shape = (3, 5, 3)
        self.assertEqual(test_shape, valid_shape)


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
