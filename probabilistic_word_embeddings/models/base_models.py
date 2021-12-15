import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probabilistic_word_embeddings.distributions as custom_dist
from probabilistic_word_embeddings.embeddings import NormalEmbedding, concat_embeddings

# Bernoulli Skip-Gram with Negative Samples
def sgns_model(vocab_size=None, dim=25, ws=2, ns=5, batch_size=25000):
    """
    Bernoulli SGNS model with an arbitrary prior distribution of the embedding \\( \\theta = [\\rho, \\alpha] \\)

    Args:
        theta: Joint prior distribution of the word and the context vectors
        ws: Size of the context window
        ns: Number of negative samples
        batch_size: Number of data points per training iteration
    """
    total_minibatch = batch_size * ws * (1 + ns)
    shape = (vocab_size * 2, dim)
    model = tfd.JointDistributionNamedAutoBatched(dict(
        theta=        custom_dist.Noninformative(shape=shape),

        # i,j and ns_j: Indices of the word types in the data
        i=             tfd.Sample(custom_dist.UniformCategorical(vocab_size), total_minibatch),
        j=             tfd.Sample(custom_dist.UniformCategorical(vocab_size), total_minibatch),

        # x: Bernoulli distributed data points
        x=lambda     theta,i,j: tfd.Bernoulli(logits=tf.reduce_sum(
                                                            tf.multiply(
                                                                tf.gather(theta[:vocab_size],   i),
                                                                tf.gather(theta[vocab_size:], j)
                                                            ),
                                                            axis=-1
                                                        )
        )
    ))
    
    # Define loss function
    x_batch = x_batch_sgns(batch_size, ws, ns)
    def get_batch(data, start_ix, dataset_ix):
        i,j = _generate_sgns_batch(data, vocab_size, D=dim, ws=ws, ns=ns,
                        batch=batch_size, start_ix=start_ix, dataset_ix=dataset_ix)
        return i, j

    def loss(batch, theta):
        i, j = batch[0], batch[1]
        return -model.log_prob(dict(theta=theta, i=i, j=j, x=x_batch))
    
    # Copy function attributes to the object
    model.get_batch = lambda data, start_ix, dataset_ix: get_batch(data, start_ix, dataset_ix)
    model.loss = lambda batch, theta: loss(batch, theta)
    model.init = lambda: [theta.init()]
    model.V, model.D, model.ws, model.ns, model.batch = vocab_size, dim, ws, ns, batch_size
    
    model.theta_split = 0, vocab_size # where to split theta into rho and alpha
    
    return model

# Generate a random i,j batch of the data.
#@tf.function
def _generate_sgns_batch(data, V, D=100, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    i,j = _generate_cbow_batch(data, V, D=D, ws=ws, ns=ns, batch=batch, start_ix=start_ix, dataset_ix=dataset_ix)
    
    i = tf.transpose(tf.tile([i], [ws * 2, 1]))
    
    i = tf.reshape(i, [ws * 2 * (1 + ns) * batch])
    j = tf.reshape(j, [ws * 2 * (1 + ns) * batch])
    return i,j

def x_batch_sgns(minibatch, ws, ns):
    x_pos = tf.ones((minibatch * ws * 2,), dtype=tf.int32)
    x_neg = tf.zeros((minibatch * ws * ns * 2,), dtype=tf.int32)
    return tf.concat((x_pos, x_neg), 0)

# Bernoulli Continuous Bag of Words
def cbow_model(vocab_size=None, dim=25, ws=2, ns=5, batch_size=25000):
    """
    Bernoulli CBOW model with an arbitrary prior distribution of the embedding \\( \\theta = [\\rho, \\alpha] \\)

    Args:
        theta: Joint prior distribution of the word and the context vectors
        ws: Size of the context window
        ns: Number of negative samples
        batch_size: Number of data points per training iteration
    """
    total_minibatch = batch_size * (1 + ns)
    shape = (vocab_size * 2, dim)
    model = tfd.JointDistributionNamedAutoBatched(dict(
        # rho and alpha: Embedding vectors
        theta=         custom_dist.Noninformative(shape=shape), # Word vectors

        # i,j and ns_j: Indices of the word types in the data
        i=             tfd.Sample(custom_dist.UniformCategorical(vocab_size), total_minibatch),
        j=             tfd.Sample(custom_dist.UniformCategorical(vocab_size), (total_minibatch, 2 * ws) ),

        # x: Bernoulli distributed data points
        x=lambda       theta,i,j: tfd.Bernoulli(logits=tf.reduce_sum(
                                                            tf.multiply(
                                                                tf.gather(theta[:vocab_size], i),
                                                                tf.reduce_sum(
                                                                    tf.gather(theta[vocab_size:], j), axis=-2
                                                                )
                                                            ),
                                                            axis=-1
                                                        )
        )
    ))
    
    # Define loss function
    x_batch = x_batch_cbow(batch_size, ns)
    def get_batch(data, start_ix, dataset_ix):
        i,j = _generate_cbow_batch(data, vocab_size, D=dim, ws=ws, ns=ns,
                        batch=batch_size, start_ix=start_ix, dataset_ix=dataset_ix)
        return i, j

    @tf.function
    def loss(batch, theta):
        i,j = batch[0], batch[1]
        return - model.log_prob(dict(theta=theta, i=i, j=j, x=x_batch))
    
    # Copy function attributes to the object
    model.get_batch = lambda data, start_ix, dataset_ix: get_batch(data, start_ix, dataset_ix)
    model.loss = lambda batch, theta: loss(batch, theta)
    model.init = lambda: [theta.init()]
    model.V, model.D, model.ws, model.ns, model.batch = vocab_size, dim, ws, ns, batch_size
    
    model.theta_split = 0, vocab_size # where to split theta into rho and alpha
    return model

# Generate a random i,j batch of the data.
#@tf.function
def _generate_cbow_batch(datasets, V, D=100, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    # the dog saw the cat
    # data = [0, 1, 2, ..., 0, 3]
    data = datasets[dataset_ix]
    N = data.shape[0]

    i = tf.reshape(tf.range(start_ix, start_ix + batch, dtype=tf.int32), [batch, 1])
    i_return = tf.reshape(i, [batch])
    i = tf.tile(i, [1, 2 * ws])

    j_range = tf.range(1, ws + 1, dtype=tf.int32)
    j = tf.concat([-j_range, j_range], axis=0)
    j = tf.reshape(j, [1, ws * 2])
    j = tf.tile(j, [batch, 1])
    j = ((i + j) + N) % N

    # Negative sampling
    ns_i = tf.random.uniform(maxval=N, shape=[batch * ns], dtype=tf.int32)
    ns_j = tf.tile(j, [ns, 1])

    # Get word types at the indices i, j
    i = tf.gather(data , i_return)
    j = tf.gather(data , j)
    ns_i = tf.gather(data , ns_i)
    ns_j = tf.gather(data , ns_j)

    # Concatenate positive and negative samples
    i = tf.concat([i, ns_i], axis=0)
    j = tf.concat([j, ns_j], axis=0)
    
    return i,j

# Create a with the Bernoulli distribution
# [1, 1, ..., 0, 0 ..., 0]
def x_batch_cbow(minibatch, ns):
    x_pos   = tf.ones((minibatch,), dtype=tf.int32)
    x_neg   = tf.zeros((minibatch * ns,), dtype=tf.int32)
    return tf.concat((x_pos, x_neg), 0)


model_functions = dict(
    cbow= cbow_model,
    sgns= sgns_model
)

