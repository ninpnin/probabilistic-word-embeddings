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
from sklearn.utils import shuffle

def map_estimate(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, epochs=5, evaluate=True, profile=False, history=False):
    """
    This function performs MAP estimation.

    Args:
        model: Word embedding model.

            The model is expected to have 'loss' and 'init' functions.
        data: Data as a list of of NumPy arrays. The arrays should consist of word indices.
        init: Variable parameters in the model as a list of tf.Variable's. If None, will be initialized from the models init function.
        epochs: The number of passes over the data.
    """
    if profile:
        tf.profiler.experimental.start("logs")

    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    e = embedding
    N = len(data)
    batches = N // batch_size
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if evaluate:
            similarity = evaluate_word_similarity(embedding)
            print(similarity)
            wa = sum(similarity["Rank Correlation"] * similarity["No. of Observations"]) / sum(similarity["No. of Observations"])
        
            print("Weighted average", wa)

        for batch in progressbar.progressbar(shuffle(list(range(batches)))):
            start_ix = batch_size * batch
            if model == "sgns":
                i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                objective = lambda: - tf.reduce_sum(sgns_likelihood(embedding, i, j, x=x)) - embedding.log_prob(batch_size, N)
            elif model == "cbow":
                i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
                objective = lambda: - tf.reduce_sum(cbow_likelihood(e, i, j, x=x)) - e.log_prob(batch_size, N)
            _ = opt.minimize(objective, [embedding.theta])
    return embedding