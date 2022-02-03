import tensorflow as tf

# Bernoulli Skip-Gram with Negative Samples
@tf.function
def sgns_likelihood(embedding, i, j, x=None):
    rhos, alphas = embedding[i], embedding[j]
    logits = tf.reduce_sum(tf.multiply(rhos, alphas), axis=-1)

    # If x is provided, multiply logits by it
    if x is not None:
        # Map 1 => 1, 0 => -1
        x = 2 * x - 1
        logits = tf.multiply(logits, x)

    ps = tf.math.sigmoid(logits)
    log_ps = tf.math.log(ps)
    return log_ps

@tf.function
def cbow_likelihood(embedding, i, j, x=None):
    rhos, alphas = embedding[i], embedding[j]
    alphas = tf.reduce_sum(alphas, axis=-2)
    logits = tf.reduce_sum(tf.multiply(rhos, alphas), axis=-1)

    # If x is provided, multiply logits by it
    if x is not None:
        # Map 1 => 1, 0 => -1
        x = 2 * x - 1
        logits = tf.multiply(logits, x)

    ps = tf.math.sigmoid(logits)
    log_ps = tf.math.log(ps)
    return log_ps
    
# Generate a random i,j batch of the data.
#@tf.function
def generate_sgns_batch(data, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    i,j = _generate_cbow_batch(data, tf.constant(ws), tf.constant(ns), tf.constant(batch), tf.constant(start_ix))
    
    i = tf.transpose(tf.tile([i], [ws * 2, 1]))
    
    i = tf.reshape(i, [ws * 2 * (1 + ns) * batch])
    j = tf.reshape(j, [ws * 2 * (1 + ns) * batch])
    x = tf.concat([tf.ones(ws * 2 * batch,dtype=tf.float64), tf.zeros(ws * 2 * ns * batch, dtype=tf.float64)], axis=0,)
    return i,j,x

# Generate a random i,j batch of the data.
def generate_cbow_batch(data, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    #settings = tf.constant([ws, ns, batch, start_ix, dataset_ix])
    i,j = _generate_cbow_batch(data, tf.constant(ws), tf.constant(ns), tf.constant(batch), tf.constant(start_ix))
    x = tf.concat([tf.ones(batch, dtype=tf.float64), tf.zeros(ns * batch, dtype=tf.float64)], axis=0,)
    return i,j,x

@tf.function
def _generate_cbow_batch(data, ws, ns, batch, start_ix):
    #ws, ns, batch, start_ix, dataset_ix = settings
    # the dog saw the cat
    # data = [0, 1, 2, ..., 0, 3]
    #data = datasets[dataset_ix]
    N = data.shape[0]

    i = tf.range(start_ix, start_ix + batch, dtype=tf.int32) % N
    i = tf.reshape(i, [batch, 1])
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
    
    # add context marker
    j = j + "_c"
    
    return i,j