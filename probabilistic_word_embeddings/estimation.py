"""Point estimation of the models"""
import tensorflow as tf
# Load model from models.py
from .utils import shuffled_indices
from .models import generate_sgns_batch, sgns_likelihood
from .models import generate_cbow_batch, cbow_likelihood
from .evaluation import evaluate_word_similarity

import glob
import progressbar
from scipy.spatial.distance import cosine as cos_dist
import random

def map_estimate(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, valid_data=None, profile=False, history=False):
    """
    Perform MAP estimation.
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        model (str): Word embedding model, either sgns or cbow.
        data: Data as a list of NumPy arrays. The arrays should consist of word indices.
        ws (int): SGNS or CBOW window size
        ns (int): SGNS or CBOW number of negative samples
        batch_size (int): Batch size in the training process 
        epochs (int): The number of passes over the data.
        evaluate (bool): Whether to run word similarity evaluation during training on the standard English evaluation data sets
        profile (bool): whether to run the tensorflow profiler during training
    
    Returns:
        Trained embedding
    """
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
        if model == "sgns":
            valid_data = generate_sgns_batch(valid_data, ws=ws, ns=ns, batch=batch_size, start_ix=0)
        if model == "cbow":
            valid_data = generate_cbow_batch(valid_data, ws=ws, ns=ns, batch=batch_size, start_ix=0)
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)
            wa = sum(similarity["Rank Correlation"] * similarity["No. of Observations"]) / sum(similarity["No. of Observations"])
        
            print("Weighted average", wa)
        if valid_data is not None:
            i,j,x = valid_data
            if model == "sgns":
                valid_ll = tf.reduce_sum(sgns_likelihood(embedding, i, j, x=x))
            elif model == "cbow":
                valid_ll = tf.reduce_sum(cbow_likelihood(embedding, i, j, x=x))

            print(f"Validation likelihood: {valid_ll}")

        # Shuffle the order of batches
        for batch in progressbar.progressbar(random.sample(range(batches),batches)):
            start_ix = batch_size * batch
            if model == "sgns":
                i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                objective = lambda: - tf.reduce_sum(sgns_likelihood(embedding, i, j, x=x)) - embedding.log_prob(batch_size, N)
            elif model == "cbow":
                i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                objective = lambda: - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
            _ = opt.minimize(objective, [embedding.theta])
    return embedding