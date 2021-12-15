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

class Embedding:
    def __init__(self, dist, sample_init=False):
        self.dist = dist
        self.sample_init = sample_init

    def __getattr__(self, attr):
        return getattr(self.dist, attr)

    def init(self):
        init_matrix = None
        if self.sample_init:
            init_matrix = self.dist.sample()
        else:
            es = self.dist.event_shape
            dim = es[-1]
            init_matrix = (tf.random.uniform(es) - 0.5) / dim

        # Warn the user if the number of the parameters is going to push the limits of a regular PC
        parameters = tf.reduce_prod(init_matrix.shape).numpy()
        if parameters > 100000000:
            print("Whoa! That's", "{:,}".format(parameters), "total parameters you've got there.")
            print("If you are running out of memory, try reducing the dimensionality or dropping out rarer words during preprocessing.")
            print("For dynamic embeddings, using less timesteps helps too.")

        return init_matrix




class NormalEmbedding(Embedding):
    """
    Embedding of shape (vocab_size, dim). Each embedding vector \\( \\theta_{i} \\) is IID distributed

    $$ \\theta_{j} \\sim \\mathcal{N}( \\mathbf{0}, \\lambda_0 \\mathbf{I} ) $$

    All entries are IID normally distributed.

    Args:
        vocab_size: Vocabulary size of the embedding.
        dim: Dimensionality of each word vector.
        mean: The mean of the normal distributions.
        lambda0: The standard deviation of the normal distributions.
    """
    def __init__(self, vocab_size, dim=25, mean=0.0, lambda0=1.0):
        location = tf.ones(dim) * mean
        scale_vec = tf.ones(dim) * lambda0
        embedding = tfd.Sample(tfd.MultivariateNormalDiag(loc=location, scale_diag=scale_vec), vocab_size)
        super().__init__(embedding)

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

def _timeseries(location, scale_v, timesteps, t_ratio=0.01):
    dim = location.shape[0]
    def define_model(T):
        vector_0 = yield Root(tfd.MultivariateNormalDiag(loc=location, scale_diag=scale_v))
        vector_prev = vector_0

        for t in range(T-1):
            vector_t = yield tfd.MultivariateNormalDiag(loc=vector_prev, scale_diag=scale_v * t_ratio)
            vector_prev = vector_t
    
    timeseries_model = functools.partial(define_model, T=timesteps)
    timeseries_model = tfd.JointDistributionCoroutine(timeseries_model)
    flattened_model  = tfp.distributions.Blockwise(timeseries_model)
    
    return flattened_model

class DynamicNormalEmbedding(Embedding):
    """
    Embedding of shape (timesteps, vocab_size, dim). The word vectors at timestep 0 \\( \\rho_{0i} \\) is IID distributed

    $$ \\rho_{0, i} \\sim \\mathcal{N}( \\mathbf{0}, \\lambda_0 \\mathbf{I} ) $$

    and the following is a Gaussian random process

    $$ \\rho_{t,i} \\sim \\mathcal{N}( \\rho_{t-1, i}, \\lambda_0 \\mathbf{I} ) $$

    The context vectors are static over time and are distributed

    $$ \\alpha_{i} \\sim \\mathcal{N}( \\mathbf{0}, \\lambda_0 \\mathbf{I} ) $$

    Args:
        vocab_size: Vocabulary size of the embedding.
        timesteps: number of timsteps to be modeled.
        dim: Dimensionality of each word vector.
        lambda0: The standard deviation of the normal distributions.
    """
    def __init__(self, vocab_size, timesteps, scale=1.0, dim=25):
        shape = vocab_size * dim

        location = tf.zeros(shape)
        scale_v = tf.ones(shape) * scale
        model = _timeseries(location, scale_v, timesteps)
        
        embedding = tfd.TransformedDistribution(
            distribution=model,
            bijector=tfb.Reshape(event_shape_out=[timesteps, vocab_size, dim]
        ))
        super().__init__(embedding)

def _informative_dimension(vocab_size, timesteps, si=dict(), scale=1.0):
    all_embeddings = []
    infs = len(list(si.keys()))
    inf_values = list(set(si.values()))

    si_values = [v for (k,v) in list(si.items())]
    si_values = dict(Counter(si_values))
    si_values = dict(sorted(si_values.items(), key=lambda x: -x[0]))

    embedding = DynamicNormalEmbedding(vocab_size - infs, timesteps, scale=scale, dim=1)
    all_embeddings.append(embedding)

    for inf_value, count in si_values.items():
        offset, scale_e = tf.cast(inf_value, tf.float32), tf.cast(scale * 0.01, tf.float32)
        offset, scale_e = tf.ones(timesteps*count) * offset, tf.ones(timesteps*count) * scale_e
        inf_embedding = tfd.MultivariateNormalDiag(loc=offset, scale_diag=scale_e)
        bijector = tfp.bijectors.Reshape(event_shape_out=[timesteps, count, 1])
        inf_embedding = tfp.distributions.TransformedDistribution(
            inf_embedding, bijector
        )
        inf_embedding = Embedding(inf_embedding, sample_init=True)
        all_embeddings.append(inf_embedding)

    # Permutate
    perm = [None] * vocab_size
    direct = set(range(0, vocab_size - infs)) - set(si.keys())
    indirect = set(range(vocab_size - infs, vocab_size)) - set(si.keys())

    start_ix = vocab_size - infs

    for i in list(direct):
        perm[i] = i

    for inf_value, count in si_values.items():
        val_set = filter(lambda x: x[1] == inf_value, list(si.items()))
        val_set = [x[0] for x in val_set]
        for ix, i in enumerate(val_set):
            source_i = ix + start_ix
            perm[i] = source_i

        start_ix += len(val_set)

    for ix, i in enumerate(list(indirect)):
        source_i = list(si.keys())[ix]
        perm[i] = source_i
    concat = concat_embeddings(all_embeddings, axis=1)

    perm_bijector = tfp.bijectors.Permute(perm, axis=-2)
    embedding = tfp.distributions.TransformedDistribution(
        concat, perm_bijector
    )
    def init():
        conit = concat.init()
        return perm_bijector.forward(conit)
    embedding.init = lambda: init()

    return embedding

class DynamicInformativeEmbedding(Embedding):
    def __init__(self, vocab_size, timesteps, si=dict(), scale=1.0, dim=25):
        noninformative = DynamicNormalEmbedding(vocab_size, timesteps, scale=scale, dim=dim-1)
        informative = _informative_dimension(vocab_size, timesteps, si=si, scale=scale)
        dist = concat_embeddings([noninformative, informative], axis=2)
        super().__init__(dist)
        self.init = lambda: self.dist.init()

if __name__ == '__main__':
    V, D, T = 70000,100,2
    si = {2: 1.0, 4: 1.0, 5: -1.0}

    embedding = dynamic_informative_embedding(V, T, si=si, dim=D)
    print("Init")

    sample = embedding.init()
    
    print(sample.shape)
    print(embedding.log_prob(sample))

    print("Sample")
    sample = embedding.sample()
    print(sample.shape)
    print(embedding.log_prob(sample))


def concat_embeddings(embeddings, axis=0):
    """
    Concatenates two or more embeddings along a given axis.

    Args:
        embeddings: A list of embeddings. Their shapes should match apart from the axis along which they are concatenated.
        axis: The index of the axis to concatenate the embeddings.
    """
    # Validate args
    assert len(embeddings) >= 2, "There needs to be at least 2 embeddings to concat"
    em0_shape = list(embeddings[0].event_shape)
    rank = len(em0_shape)
    for em in embeddings:
        em_shape = list(em.event_shape)
        assert rank == len(em_shape), "The embeddings have to be of same tensor rank. Got " + str(rank) + " and" + str(len(em_shape))
        for ix in range(rank):
            valid_dims = em0_shape[ix] == em_shape[ix] or ix == axis
            assert valid_dims, "All dimensions except 'axis' must be equal. Got " + str(em0_shape[ix]) + " and " + str(em_shape[ix]) + " at index " + str(ix)
    
    def init():
        inits = [em.init() for em in embeddings]
        return tf.concat(inits, axis=axis)
    
    shape = em0_shape.copy()
    for em in embeddings[1:]:
        em_shape = list(em.event_shape)
        shape[axis] += em_shape[axis]
   
    perm = list(range(rank))
    perm[axis] = 0
    perm[0] = axis

    transposed_embeddings = []
    for em in embeddings:
        transposed_em = tfd.TransformedDistribution(
            distribution=em,
            bijector=tfp.bijectors.Transpose(perm=perm)
        )
        transposed_embeddings.append(transposed_em)
    
    axis_shape = shape[axis]
    shape_0 = shape[0]
    shape[0] = axis_shape
    shape[axis] = shape_0
    
    dist = tfp.distributions.independent_joint_distribution_from_structure(transposed_embeddings)
    dist = tfd.Blockwise(dist)
    dist = tfd.TransformedDistribution(
        distribution=dist,
        bijector=tfp.bijectors.Reshape(shape)
    )
    
    if axis != 0:
        dist = tfd.TransformedDistribution(
            distribution=dist,
            bijector=tfp.bijectors.Transpose(perm=perm)
        )

    dist.init = lambda: init()
    return dist