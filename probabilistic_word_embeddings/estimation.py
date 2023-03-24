"""Point estimation of the models"""
import tensorflow as tf
# Load model from models.py
from .utils import shuffled_indices
from .embeddings import Embedding
from .models import generate_sgns_batch, sgns_likelihood
from .models import generate_cbow_batch, cbow_likelihood
from .evaluation import evaluate_word_similarity, evaluate_on_holdout_set

import glob
import progressbar
from scipy.spatial.distance import cosine as cos_dist
import random, warnings

def map_estimate(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, vocab_freqs=None, valid_data=None, early_stopping=False, profile=False):
    """
    Perform MAP estimation.
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        data: Data as a list of NumPy arrays. The arrays should consist of word indices.
        model (str): Word embedding model, either sgns or cbow.
        ws (int): SGNS or CBOW window size
        ns (int): SGNS or CBOW number of negative samples
        batch_size (int): Batch size in the training process 
        epochs (int): The number of passes over the data.
        evaluate (bool): Whether to run word similarity evaluation during training on the standard English evaluation data sets
        valid_data: Data as a NumPy array, list or a tf.Tensor. The arrays should consist of word indices.
        early_stopping (bool): Data as a NumPy array, list or a tf.Tensor. The arrays should consist of word indices.
        profile (bool): whether to run the tensorflow profiler during training
    
    Returns:
        Trained embedding
    """
    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")
    if profile:
        tf.profiler.experimental.start("logs")

    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    e = embedding
    N = len(data)
    batches = N // batch_size
    if valid_data is not None:
        if not isinstance(valid_data, tf.Tensor):
            valid_data = tf.constant(valid_data)
        valid_batches = len(valid_data) // batch_size
        best_valid_performance = None
        best_valid_weights = None

    if vocab_freqs is None:
        ns_data = data
    else:
        vocab = [wd for wd in vocab_freqs]
        freqs = [vocab_freqs[wd] for wd in vocab]
        vocab = tf.constant(vocab)
        logits = 3/4 * tf.math.log(tf.constant([freqs]))
        ns_data = []
        print("Randomize negative sample dataset...")
        ns_batch_size = (500 * 1000 * 1000) // len(vocab)
        ns_batches = N // ns_batch_size
        for batch in progressbar.progressbar(range(ns_batches)):
            ns_i = tf.random.categorical(logits, ns_batch_size)
            ns_data_batch = tf.gather(vocab, ns_i)
            ns_data_batch = tf.reshape(ns_data_batch, [ns_batch_size])
            ns_data.append(ns_data_batch)

        ns_data.append(data[ns_batches * ns_batch_size:])
        ns_data = tf.concat(ns_data, axis=0)

        assert len(ns_data) == len(data), f"{len(ns_data)} vs {len(data)}"

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)
            wa = sum(similarity["Rank Correlation"] * similarity["No. of Observations"]) / sum(similarity["No. of Observations"])
        
            print("Weighted average", wa)

        if valid_data is not None:
            print("validate...")
            valid_ll = evaluate_on_holdout_set(embedding, valid_data, model=model, ws=ws, ns=ns, batch_size=batch_size)
            print(f"Mean validation likelihood: {valid_ll}")
            if best_valid_performance is not None and valid_ll > best_valid_performance:
                print(f"{valid_ll} is better than previous best {best_valid_performance}")
                best_valid_performance = valid_ll
                if early_stopping:
                    best_valid_weights = embedding.theta.numpy()
            elif best_valid_performance is None:
                best_valid_performance = valid_ll

        # Shuffle the order of batches
        for batch in progressbar.progressbar(random.sample(range(batches),batches)):
            start_ix = batch_size * batch
            if model == "sgns":
                i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix, ns_data=ns_data)
                objective = lambda: - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
            elif model == "cbow":
                i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix, ns_data=ns_data)
                objective = lambda: - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
            _ = opt.minimize(objective, [embedding.theta])
    if early_stopping and valid_data is not None and best_valid_weights is not None:
        print("Assign the weights corresponding to the best validation loss")
        embedding.theta.assign(best_valid_weights)
    return embedding

def mean_field_vi(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, valid_data=None):
    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")

    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=0.001)
    e = embedding
    N = len(data)
    batches = N // batch_size
    
    theta_mean = tf.Variable(tf.random.normal(e.theta.shape, dtype=tf.float64)* 0.00001)
    theta_std_log =  tf.Variable(tf.random.normal(e.theta.shape, dtype=tf.float64)* 0.00001)
    
    opt_mean_var = optimizer.add_variable_from_reference(theta_mean, "theta_mean", initial_value=theta_mean)
    opt_std_var = optimizer.add_variable_from_reference(theta_std_log, "theta_std_log", initial_value=theta_std_log)
    optimizer.build([opt_mean_var, opt_std_var])

    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # Shuffle the order of batches
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)

        for batch in progressbar.progressbar(random.sample(range(batches),batches)):
            epsilon = tf.random.normal(theta_mean.shape, dtype=tf.float64)
            z = theta_mean + tf.multiply(tf.math.exp(theta_std_log), epsilon)
            embedding.theta.assign(z)
            
            start_ix = batch_size * batch
            i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
            
            with tf.GradientTape() as tape:
                log_prob = tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) + e.log_prob(batch_size, N)
                d_l_d_theta = -tape.gradient(log_prob, embedding.theta) * N / batch_size
            
            d_l_d_theta_mean = d_l_d_theta
            d_l_d_theta_std_log = tf.multiply(tf.multiply(d_l_d_theta, epsilon), tf.math.exp(theta_std_log))
            d_l_d_theta_std_log = d_l_d_theta_std_log - tf.ones(d_l_d_theta_std_log.shape, dtype=tf.float64)

            optimizer.update_step(d_l_d_theta_mean, opt_mean_var)
            optimizer.update_step(d_l_d_theta_std_log, opt_std_var)
            print(opt_mean_var)
            print(opt_std_var)
            
            theta_mean.assign(opt_mean_var)
            theta_std_log.assign(opt_std_var)

            
        embedding.theta.assign(opt_mean_var)
    return embedding, theta_std_log
