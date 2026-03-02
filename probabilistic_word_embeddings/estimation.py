"""Point estimation of the models"""
import tensorflow as tf
import numpy as np
# Load model from models.py
from .utils import shuffled_indices
from .utils import get_logger
from .embeddings import Embedding
from .models import generate_batch
from .models import generate_cbow_batch, cbow_likelihood, sgns_likelihood
from .evaluation import evaluate_word_similarity, evaluate_on_holdout_set

import glob
import progressbar
from scipy.spatial.distance import cosine as cos_dist
import random, warnings
import sys
import logging
import copy

def map_estimate(embedding, data=None, ns_data=None, data_generator=None, N=None, model="cbow", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=False, valid_data=None, early_stopping=False, profile=False, training_loss=True, loglevel="DEBUG"):
    """
    Perform MAP estimation.
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        data: Data as a list of python strings.
        data_generator: Data as a generator that yields (i, j, x) tuples, where i are the center words as tf.Tensor (str),
            j are the context words as tf.Tensor (str), and x the Bernoulli outcomes as tf.Tensor (int). Alternative to 'data'.
        N (int): number of observations. Only necessary when using 'data_generator'.
        model (str): Word embedding model, either 'sgns' or 'cbow'.
        ws (int): SGNS or CBOW window size. Only necessary when using 'data'.
        ns (int): SGNS or CBOW number of negative samples. Only necessary when using 'data'.
        batch_size (int): Batch size in the training process 
        epochs (int): The number of passes over the data.
        evaluate (bool): Whether to run word similarity evaluation during training on the standard English evaluation data sets
        valid_data: Data as a list of python strings.
        early_stopping (bool): Wheter to only save the best model according to the validation loss. Requires 'valid_data'.
        profile (bool): whether to run the tensorflow profiler during training
        training_loss (bool): whether to print out the training loss during training.
        loglevel (bool): log level for the training script
    
    Returns:
        Trained embedding
    """
    logger = get_logger(loglevel, name="map")

    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")
    if (data is None) == (data_generator is None):
        raise ValueError("Provide either 'data' or 'data_generator'")
    if (data is None) and (N is None):
        raise ValueError("Provide 'N' when using 'data_generator'")
    if (data is None) and ((ws is not None) or (ns is not None)):
        warnings.warn("The arguments 'ws' and 'ns' are ignored when using 'data_generator'")
    if profile:
        tf.profiler.experimental.start("logs")

    if data is not None:
        if not isinstance(data, tf.Tensor):
            data = tf.constant(data)
            N = len(data)
    
    batches = N // batch_size
    
    if ns_data is None:
        ns_data = data

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt_theta = opt.add_variable_from_reference(embedding.theta, "theta")#, initial_value=embedding.theta)
    opt.build([opt_theta])
    opt_theta.assign(embedding.theta)

    e = embedding
    if valid_data is not None:
        if not isinstance(valid_data, tf.Tensor):
            valid_data = tf.constant(valid_data)
        valid_batches = len(valid_data) // batch_size
        best_valid_performance = None
        best_valid_weights = None

    for epoch in range(epochs):
        logger.log(logging.TRAIN, f"Epoch {epoch}")
        
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)
            wa = sum(similarity["Rank Correlation"] * similarity["No. of Observations"]) / sum(similarity["No. of Observations"])
        
            print("Weighted average", wa)

        if valid_data is not None:
            logger.debug("validate...")
            valid_ll = evaluate_on_holdout_set(embedding, valid_data, model=model, ws=ws, ns=ns, batch_size=batch_size)
            logger.log(logging.TRAIN, f"Mean validation likelihood: {valid_ll}")
            if best_valid_performance is not None and valid_ll > best_valid_performance:
                logger.log(logging.TRAIN, f"{valid_ll} is better than previous best {best_valid_performance}")
                best_valid_performance = valid_ll
                if early_stopping:
                    best_valid_weights = embedding.theta.numpy()
            elif best_valid_performance is None:
                best_valid_performance = valid_ll

        # Shuffle the order of batches
        epoch_training_loss = []
        randomized_batches = random.sample(range(batches),batches)
        for batch in progressbar.progressbar(randomized_batches, redirect_stdout=True):
            start_ix = batch_size * batch
            if data is None:
                i,j,x = next(data_generator)
            else:
                i,j,x  = generate_batch(data, model=model, ws=ws, ns=ns, batch_size=batch_size, start_ix=start_ix, ns_data=ns_data)
            with tf.GradientTape() as tape:
                if model == "sgns":
                    objective = - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
                elif model == "cbow":
                    objective = - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
                d_l_d_theta = - tape.gradient(objective, e.theta)

            opt.update_step(d_l_d_theta, opt_theta, 0.001)
            embedding.theta.assign(opt_theta)
            if training_loss:
                epoch_training_loss.append(objective / len(i))
                batch_no = len(epoch_training_loss)
                if batch_no % 250 == 0:
                    logger.log(logging.TRAIN, f"Epoch {epoch} mean training loss after {batch_no} batches: {np.mean(epoch_training_loss)}")

    if early_stopping and valid_data is not None and best_valid_weights is not None:
        logger.info("Assign the weights corresponding to the best validation loss")
        embedding.theta.assign(best_valid_weights)
    return embedding

def mean_field_vi(embedding, data=None, data_generator=None, N=None, model="cbow", ws=5, ns=5, batch_size=25000,
                   epochs=5, init_mean=True, init_std=0.05, evaluate=True, valid_data=None, elbo_history=False, words_to_fix_rotation=None, loglevel="DEBUG"):
    """
    Perform mean-field variational inference.
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        data: Data as a list of python strings.
        data_generator: Data as a generator that yields (i, j, x) tuples, where i are the center words as tf.Tensor (str),
            j are the context words as tf.Tensor (str), and x the Bernoulli outcomes as tf.Tensor (int). Alternative to 'data'.
        N (int): number of observations. Only necessary when using 'data_generator'.
        model (str): Word embedding model, either 'sgns' or 'cbow'.
        ws (int): SGNS or CBOW window size
        ns (int): SGNS or CBOW number of negative samples
        batch_size (int): Batch size in the training process 
        epochs (int): The number of passes over the data.
        init_mean (bool): Randomize the initial values of the embedding. If False, uses the values provided in the 'embedding' argument.
        init_std (float / np.array / tf.Tensor): Default value for the standard deviations. Can be a scalar (float) or an array (np.array / tf.Tensor)
        evaluate (bool): Whether to run word similarity evaluation during training on the standard English evaluation data sets
        valid_data: Data as a list of python strings.
        elbo_history (bool): Whether to return the ELBO history as a list
        words_to_fix_rotation (array): Words for which to fix rotation of. First word gets 1 dimension and last word gets d-1 dimensions.

    Returns:
        A tuple consisting of the means as a pwe.Embedding and the standard deviations as an np.array
    """
    logger = get_logger(loglevel, name="vi")
    if not isinstance(embedding, Embedding):
        warnings.warn("embedding is not a subclass of probabilistic_word_embeddings.Embedding")
    if model not in ["sgns", "cbow"]:
        raise ValueError("model must be 'sgns' or 'cbow'")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    e = embedding

    if words_to_fix_rotation:
        assert len(words_to_fix_rotation) == e.dimensionality-1, f"Need to have embedding.dimensionality-1 number of words to fix. embedding.dimensionality = {e.dimensionality}"

    if data is not None:
        if not isinstance(data, tf.Tensor):
            data = tf.constant(data)
        N = len(data)

    batches = N // batch_size
    
    q_mean_init = embedding.theta
    if init_mean:
        q_mean_init = tf.random.normal(e.theta.shape, dtype=tf.float64) * 0.00001
    q_mean = tf.Variable(q_mean_init)
    if type(init_std) == float:
        init_std = tf.random.normal(e.theta.shape, dtype=tf.float64)* 0.00001 + tf.cast(tf.math.log(init_std), dtype=tf.float64)
        logger.info(f"Init std: {init_std}")
    q_std_log =  tf.Variable(init_std)
    
    opt_mean_var = optimizer.add_variable_from_reference(q_mean, "q_mean")
    opt_std_var = optimizer.add_variable_from_reference(q_std_log, "q_std_log")
    optimizer.build([opt_mean_var, opt_std_var])

    opt_mean_var.assign(q_mean)
    opt_std_var.assign(q_std_log)

    elbos = []
    for epoch in range(epochs):
        logger.log(logging.TRAIN, f"Epoch {epoch}")
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
            
            if data is None:
                i,j,x = next(data_generator)
            else:
                start_ix = batch_size * batch
                i,j,x  = generate_batch(data, model=model, ws=ws, ns=ns, batch_size=batch_size, start_ix=start_ix)

            with tf.GradientTape() as tape:
                if model == "cbow":
                    log_prob = tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) + e.log_prob(batch_size, N)
                elif model == "sgns":
                    log_prob = tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) + e.log_prob(batch_size, N)
                epoch_logprobs.append(log_prob * N / batch_size)
                d_l_d_theta = -tape.gradient(log_prob, embedding.theta) * N / batch_size
            
            d_l_d_q_mean = d_l_d_theta
            d_l_q_std_log = tf.multiply(tf.multiply(d_l_d_theta, epsilon), tf.math.exp(q_std_log))

            # Add the entropy term
            d_l_q_std_log = d_l_q_std_log - tf.ones(d_l_q_std_log.shape, dtype=tf.float64)

            optimizer.update_step(d_l_d_q_mean, opt_mean_var, 0.001)
            optimizer.update_step(d_l_q_std_log, opt_std_var, 0.001)
            

            std_numerical_stability_constant = 10.0
            if words_to_fix_rotation: 
                mask = np.ones_like(q_mean.numpy())  
                for idx, word in enumerate(words_to_fix_rotation):
                    word_index = e.vocabulary[word]
                    mask[word_index, idx+1:] = 0
                mask = tf.constant(mask, dtype=tf.float64)

                opt_mean_var.assign(tf.multiply(opt_mean_var, mask))
                opt_std_var.assign( tf.multiply(opt_std_var, mask) + (mask-1.0) * std_numerical_stability_constant ) 


            q_mean.assign(opt_mean_var)
            q_std_log.assign(opt_std_var)

        epoch_entropy = tf.reduce_sum(opt_std_var)
        epoch_elbo = tf.reduce_mean(epoch_logprobs) + epoch_entropy
        logger.log(logging.TRAIN, f"Epoch ELBO: {epoch_elbo.numpy()}")
        embedding.theta.assign(opt_mean_var)
        elbos.append(epoch_elbo.numpy())

    embedding_q_mean = embedding
    embedding_q_std = copy.deepcopy(embedding_q_mean)
    embedding_q_std.theta.assign(tf.math.exp(q_std_log))
    if elbo_history:
        return embedding_q_mean, embedding_q_std, elbos
    return embedding_q_mean, embedding_q_std





