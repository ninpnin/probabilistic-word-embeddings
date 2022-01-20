import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probabilistic_word_embeddings.distributions as custom_dist

# Bernoulli Skip-Gram with Negative Samples
class SGNS:
    """
    Bernoulli SGNS model with an arbitrary prior distribution of the embedding \\( \\theta = [\\rho, \\alpha] \\)

    Args:
        theta: Joint prior distribution of the word and the context vectors
        ws: Size of the context window
        ns: Number of negative samples
        batch_size: Number of data points per training iteration
    """
    def __init__(self,vocab_size=None, dim=25):
        # Define loss function
        
        # Copy function attributes to the object
        self.V, self.D = vocab_size, dim
        
        self.theta_split = 0, vocab_size # where to split theta into rho and alpha

    def get_batch(self, data, start_ix, dataset_ix, ws=2, ns=5):
        i,j = _generate_sgns_batch(data, vocab_size, D=dim, ws=ws, ns=ns,
                        batch=batch_size, start_ix=start_ix, dataset_ix=dataset_ix)
        return i, j

    def loss(self, batch, theta):
        i, j, x = batch[0], batch[1], batch[2]
        logits = tf.tensordot(theta[i], theta[j], axes=(-1,-1))
        ps = tf.math.sigmoid(logits)
        log_ps = tf.math.log(ps)
        return tf.reduce_sum(log_ps) + theta.log_prob()
    
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
