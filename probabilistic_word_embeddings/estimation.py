"""Point estimation of the models"""
import tensorflow as tf
import numpy as np
# Load model from models.py
from .utils import shuffled_indices
from .embeddings import Embedding
from .models import generate_batch
from .models import generate_cbow_batch, cbow_likelihood, sgns_likelihood
from .evaluation import evaluate_word_similarity, evaluate_on_holdout_set

import glob
import progressbar
from scipy.spatial.distance import cosine as cos_dist
import random, warnings
import sys

def map_estimate(embedding, data, model="cbow", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, vocab_freqs=None, valid_data=None, early_stopping=False, profile=False, training_loss=False):
    """
    Perform MAP estimation.
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        data: Data as a list of python strings.
        model (str): Word embedding model, either 'sgns' or 'cbow'.
        ws (int): SGNS or CBOW window size
        ns (int): SGNS or CBOW number of negative samples
        batch_size (int): Batch size in the training process 
        epochs (int): The number of passes over the data.
        evaluate (bool): Whether to run word similarity evaluation during training on the standard English evaluation data sets
        valid_data: Data as a list of python strings.
        early_stopping (bool): Wheter to only save the best model according to the validation loss. Requires valid_data.
        profile (bool): whether to run the tensorflow profiler during training
        training_loss (bool): whether to print out the training loss during training.
    
    Returns:
        Trained embedding
    """
    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")
    if profile:
        tf.profiler.experimental.start("logs")

    datasets = data
    if not isinstance(data[0], list):
        datasets = [data]

    datasets = [tf.constant(data) for data in datasets]

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    e = embedding
    Ns = [len(data) for data in datasets]
    batches = [N // batch_size for N in Ns]
    if valid_data is not None:
        if not isinstance(valid_data, tf.Tensor):
            valid_data = tf.constant(valid_data)
        valid_batches = len(valid_data) // batch_size
        best_valid_performance = None
        best_valid_weights = None

    if vocab_freqs is None:
        ns_data_list = [data for data in datasets]
    else:
        ns_data_list = []
        vocab_freqs_list = vocab_freqs
        if type(vocab_freqs) is dict:
            vocab_freqs_list = [vocab_freqs]
        for N, vocab_freqs, data in zip(Ns, vocab_freqs_list, datasets):
            vocab = [wd for wd in vocab_freqs]
            freqs = [vocab_freqs[wd] for wd in vocab]
            vocab = tf.constant(vocab)
            logits = 3/4 * tf.math.log(tf.constant([freqs]))
            print("Randomize negative sample dataset...")
            ns_batch_size = (500 * 1000 * 1000) // len(vocab)
            ns_batches = N // ns_batch_size
            ns_data = []
            for batch in progressbar.progressbar(range(ns_batches)):
                ns_i = tf.random.categorical(logits, ns_batch_size)
                ns_data_batch = tf.gather(vocab, ns_i)
                ns_data_batch = tf.reshape(ns_data_batch, [ns_batch_size])
                ns_data.append(ns_data_batch)

            ns_data.append(data[ns_batches * ns_batch_size:])
            ns_data = tf.concat(ns_data, axis=0)
        ns_data_list.append(ns_data)

        for ns_data, data in zip(ns_data_list, datasets):
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
        epoch_training_loss = []
        batches_max = max(batches)
        randomized_batches = [[random.sample(range(batches_i), 1)[0] for batches_i in batches] for _ in range(batches_max)]
        for batch_list in progressbar.progressbar(randomized_batches, redirect_stdout=True):
            start_indices = [batch_size * batch for batch in batch_list]

            for dataset_ix in range(len(datasets)):
                start_ix, data, ns_data, N = start_indices[dataset_ix], datasets[dataset_ix], ns_data_list[dataset_ix], Ns[dataset_ix]
                i,j,x  = generate_batch(data, model=model, ws=ws, ns=ns, batch_size=batch_size, start_ix=start_ix, ns_data=ns_data)
                if model == "sgns":
                    objective = lambda: - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
                elif model == "cbow":
                    objective = lambda: - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
                _ = opt.minimize(objective, [embedding.theta])
                if training_loss:
                    epoch_training_loss.append(objective() / len(i))
                    batch_no = len(epoch_training_loss)
                    if batch_no % 250 == 0:
                        print(f"Epoch {epoch} mean training loss after {batch_no} batches: {np.mean(epoch_training_loss)}")

    if early_stopping and valid_data is not None and best_valid_weights is not None:
        print("Assign the weights corresponding to the best validation loss")
        embedding.theta.assign(best_valid_weights)
    return embedding

def mean_field_vi(embedding, data, model="cbow", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, valid_data=None, elbo_history=False):
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
    
    q_mean = tf.Variable(tf.random.normal(e.theta.shape, dtype=tf.float64)* 0.00001)
    q_std_log =  tf.Variable(tf.random.normal(e.theta.shape, dtype=tf.float64)* 0.00001 - 3.0)
    
    opt_mean_var = optimizer.add_variable_from_reference(q_mean, "q_mean", initial_value=q_mean)
    opt_std_var = optimizer.add_variable_from_reference(q_std_log, "q_std_log", initial_value=q_std_log)
    optimizer.build([opt_mean_var, opt_std_var])

    elbo_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # Shuffle the order of batches
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)

        epoch_logprobs = []
        for batch in progressbar.progressbar(random.sample(range(batches),batches)):
            # Reparametrization trick, Q = mu + sigma * epsilon
            epsilon = tf.random.normal(q_std_log.shape, dtype=tf.float64)
            z = q_mean + tf.multiply(tf.math.exp(q_std_log), epsilon)
            embedding.theta.assign(z)
            
            start_ix = batch_size * batch
            i,j,x  = generate_batch(data, model=model, ws=ws, ns=ns, batch_size=batch_size, start_ix=start_ix)

            with tf.GradientTape() as tape:
                if model == "cbow":
                    log_prob = tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) + e.log_prob(batch_size, N)
                elif model == "sgns":
                    log_prob = tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) + e.log_prob(batch_size, N)
                epoch_logprobs.append(log_prob)
                d_l_d_theta = -tape.gradient(log_prob, embedding.theta) * N / batch_size
            
            d_l_d_q_mean = d_l_d_theta
            d_l_q_std_log = tf.multiply(tf.multiply(d_l_d_theta, epsilon), tf.math.exp(q_std_log))

            # Add the entropy term
            d_l_q_std_log = d_l_q_std_log - tf.ones(d_l_q_std_log.shape, dtype=tf.float64)

            optimizer.update_step(d_l_d_q_mean, opt_mean_var)
            optimizer.update_step(d_l_q_std_log, opt_std_var)
            
            q_mean.assign(opt_mean_var)
            q_std_log.assign(opt_std_var)

        epoch_entropy = tf.reduce_sum(opt_std_var)
        epoch_elbo = tf.reduce_mean(epoch_logprobs) + epoch_entropy
        print(f"Epoch ELBO: {epoch_elbo.numpy()}")
        embedding.theta.assign(opt_mean_var)
        elbo_history.append(epoch_elbo.numpy())

    embedding_q_mean = embedding
    if elbo_history:
        return embedding_q_mean, q_std_log, elbo_history
    return embedding_q_mean, q_std_log






