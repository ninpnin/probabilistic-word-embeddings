import unittest

import math
import tensorflow as tf

import probabilistic_word_embeddings.distributions as ctd

class Test(unittest.TestCase):

    def test_uniform_categorical(self):
        # Test that the default and the custom categorical distributions yield
        # the same log probabilities for a range of number of categories
        for categories in [5, 15, 250, 1000]:
            probs   = tf.ones(categories) / categories
            default = tfd.Categorical(probs=probs)
            custom  = ctd.UniformCategorical(categories)

            prob_default = default.log_prob(2)
            prob_custom  = custom.log_prob(2)

            self.assertAlmostEqual(prob_default, prob_custom, delta=0.01)

    def test_mvn_precision(self):
        # Test that the default and the custom normal distributions yield
        # the same log probabilities for a range of randomly sampled vectors

        indices = [[0,0],[1,1],[2,2],[0,2],[2,0]]
        values  = [2., 1., 2., -1., -1.]
        precision = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[3,3])
        covariance = tf.linalg.inv(tf.sparse.to_dense(tf.sparse.reorder(precision)))

        custom = ctd.MultivariateNormalPrecision(precision = precision, normalize_log_prob=False)
        default = tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(covariance))
        
        for i in range(5):
            x1 = default.sample()
            x2 = default.sample()

            # Use differences so that constants don't affect the results
            diff_custom = custom.log_prob(x1) - custom.log_prob(x2)
            diff_default = default.log_prob(x1) - default.log_prob(x2)
            self.assertAlmostEqual(diff_default, diff_custom, delta=0.01)

    def test_noninformative_1(self):
        shape = (5,)
        custom = ctd.Noninformative(shape=shape)
        x = tf.random.normal(shape)
        y = custom.log_prob(x)

        # Check that log_prob is zero
        self.assertAlmostEqual(0, y, delta=0.01)

    def test_noninformative_2(self):
        shape = (5,3)
        custom = ctd.Noninformative(shape=shape)
        x = tf.random.normal(shape)
        y = custom.log_prob(x)

        # Check shape
        self.assertEqual( (3,), y.shape)

        # Check that log_prob is zero
        y = tf.reduce_sum(y)
        self.assertAlmostEqual(0, y, delta=0.01)

    def test_noninformative_3(self):
        shape = (5,3,2)
        custom = ctd.Noninformative(shape=shape)
        normal = tfd.MultivariateNormalDiag(loc=tf.zeros(shape))
        x = tf.random.normal(shape)
        y = custom.log_prob(x)

        # Check event shape
        self.assertEqual(normal.event_shape, custom.event_shape)

        # Check log-prob shape
        self.assertEqual( (3,2), y.shape)

        # Check sample shape
        self.assertEqual( (5,3,2), x.shape)

        # Check that log-prob is zero
        y = tf.reduce_sum(y)
        self.assertAlmostEqual(0, y, delta=0.01)



if __name__ == '__main__':
    # Run the test(s)
    unittest.main()