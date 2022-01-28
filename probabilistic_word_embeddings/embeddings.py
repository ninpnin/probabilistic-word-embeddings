"""
Definitions of the prior distributions of the embeddings.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
import functools
from collections import Counter

import numpy as np
import tensorflow as tf
import random
import progressbar

class Embedding:
    def __init__(self, vocabulary, dimensionality, lambda0=0.0, shared_context_vectors=True):
        assert isinstance(vocabulary, set)
        keys = list(vocabulary)
        if shared_context_vectors:
            keys = keys + list(set([key + "_c" for key in keys]))
        else:
            keys = keys + list(set([key.split("_")[0] + "_c" for key in keys]))
        keys = sorted(keys)
        values = tf.range(len(keys))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.vocabulary = tf.lookup.StaticHashTable(init, default_value=-1)
        self.theta = tf.Variable(np.random.rand(len(keys), dimensionality) - 0.5, dtype=tf.float64)
        self.lambda0 = lambda0

    @tf.function
    def __getitem__(self, item):
        if type(item) == str or isinstance(item, list):
            item = tf.constant(item)
        #print("Item", item)
        ix = self.vocabulary.lookup(item)
        #assert tf.math.reduce_min(ix) >= 0, "Requested word not found in embedding"

        #print("Ix", ix)
        return tf.gather(self.theta, ix, axis=0)

    def __len__(self):
        return len(self.theta)

    def log_prob(self):
        return - tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0

class LaplacianEmbedding(Embedding):
    """
    Embedding of shape (vocab_size, dim). Each embedding dimension \\( \\theta_{:j} \\) is IID distributed

    $$ \\theta_{:j} \\sim \\mathcal{N}( \\mathbf{0}, \\lambda_0 \\mathbf{I} + \\lambda_1 \\mathbf{L} ) $$

    i.e. the entries in one dimension are normally distributed with an augmented Laplacian as the precision matrix.

    Args:
        vocab_size: Vocabulary size of the embedding.
        dim: Dimensionality of each word vector.
        laplacian: The Laplacian matrix that is used to create the precision matrix. Requires the tf.SparseMatrix format.

            If no Laplacian is provided, it falls back to a zero matrix.
        lambda0: The diagonal weighting of the precision matrix. Corresponds to standard deviation if the Laplacian is a zero matrix.
        lambda1: The off-diagonal weighting of the precision matrix.
    """