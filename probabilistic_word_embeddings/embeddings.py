"""
Definitions of the prior distributions of the embeddings.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
import functools
import probabilistic_word_embeddings.distributions as ctd
from collections import Counter

import numpy as np
import tensorflow as tf
import random
import progressbar

class Embedding:
    """Custom list that returns None instead of IndexError"""
    def __init__(self, vocabulary, dimensionality):
        assert isinstance(vocabulary, set)
        keys = sorted(list(vocabulary))
        values = tf.range(len(keys))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.vocabulary = tf.lookup.StaticHashTable(init, default_value=-1)
        self.theta = np.random.rand(len(keys), dimensionality) - 0.5

    def __getitem__(self, item):
        if type(item) == str or isinstance(item, list):
            item = tf.constant(item)
        ix = self.vocabulary.lookup(item)
        print(ix)
        return tf.gather(self.theta, ix, axis=0)


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
    def __init__(self, vocab_size, laplacian=None, lambda0=1.0, lambda1=1.0, dim=25):
        precision = tf.sparse.eye(vocab_size) / (lambda0 * lambda0)
        if laplacian != None:
            precision = tf.sparse.add(precision, laplacian / (lambda1 * lambda1) )
        
        model = tfd.Sample(ctd.MultivariateNormalPrecision(precision=precision), dim)
        embedding = tfd.TransformedDistribution(
            distribution=model,
            bijector=tfp.bijectors.Transpose(perm=[1, 0])
        )
        super().__init__(embedding)