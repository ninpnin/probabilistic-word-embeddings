import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probabilistic_word_embeddings.distributions as custom_dist

# Bernoulli Skip-Gram with Negative Samples
def sgns_likelihood(batch, theta):
    i, j, x = batch[0], batch[1], batch[2]
    logits = tf.tensordot(theta[i], theta[j], axes=(-1,-1))
    ps = tf.math.sigmoid(logits)
    log_ps = tf.math.log(ps)
    return tf.reduce_sum(log_ps) + theta.log_prob()
    
# Generate a random i,j batch of the data.
#@tf.function
def generate_sgns_batch(data, D=100, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    i,j = generate_cbow_batch(data, D=D, ws=ws, ns=ns, batch=batch, start_ix=start_ix, dataset_ix=dataset_ix)
    
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
def generate_cbow_batch(datasets, D=100, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
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
