"""
Evaluation of the embedding accuracy, for both monolingual and multilingual embeddings.
"""

# TODO: Sort out GPLv3 License of Dict2Vec

import os
import sys
import math
import argparse
import numpy as np
import scipy.stats as st
import pickle
import progressbar
import pandas as pd
import pkg_resources
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import spearmanr as rank_correlation
import tensorflow as tf
import copy
from .embeddings import Embedding
import warnings

###################
# INTRINSIC EVALUATION #
###################

from .models import generate_sgns_batch, sgns_likelihood
from .models import generate_cbow_batch, cbow_likelihood
def evaluate_on_holdout_set(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, metric="likelihood", reduce_mean=True):
    """
    Evaluate the performance of an embedding on a holdout set
    
    Args:
        embedding: Embedding with a suitable vocabulary and log_prob function. Subclass of pwe.Embedding
        data: Data as a list of NumPy arrays. The arrays should consist of word indices.
        model (str): Word embedding model, either sgns or cbow.
        ws (int): SGNS or CBOW window size
        ns (int): SGNS or CBOW number of negative samples
        batch_size (int): Batch size in the training process 
        reduce_mean (bool): whether to return the mean of the losses, or a 1D tf.Tensor list of them

    Returns:
        Validation performance as a tf.Tensor
    """
    assert metric in ["likelihood", "accuracy"]
    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)
    valid_batches = len(data) // batch_size

    valid_ll = tf.constant([], dtype=tf.float64)
    for batch in progressbar.progressbar(range(valid_batches)):
        start_ix = batch_size * batch
        if model == "sgns":
            i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
            valid_ll_batch = sgns_likelihood(embedding, i, j, x=x)
        elif model == "cbow":
            i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
            valid_ll_batch = cbow_likelihood(embedding, i, j, x=x)

        if metric == "accuracy":
            limit = tf.ones(valid_ll_batch.shape, dtype=valid_ll_batch.dtype) * tf.cast(tf.math.log(0.5), dtype=valid_ll_batch.dtype)
            valid_ll_batch = tf.math.sign(valid_ll_batch - limit)
            valid_ll_batch = valid_ll_batch / 2.0 + 0.5
        # Reduce memory footprint by calculating the mean on the fly
        if reduce_mean:
            valid_ll_batch = tf.expand_dims(tf.reduce_mean(valid_ll_batch), 0)

        valid_ll = tf.concat([valid_ll, valid_ll_batch], axis=0)

    if reduce_mean:
        return tf.reduce_mean(valid_ll)
    else:
        return valid_ll

###################
# WORD SIMILARITY #
###################

def _get_eval_file(dataset_name):
    try:
        stream = pkg_resources.resource_stream(__name__, f'data/eval/{dataset_name}.tsv')
    except FileNotFoundError:
        stream = open(dataset_name)
    return pd.read_csv(stream, sep='\t', names=["word1", "word2", "similarity"])

def embedding_similarities(df, embedding):
    words1, words2 = [], []
    rows = []
    for word1, word2 in zip(df["word1"], df["word2"]):
        if word1 in embedding and word2 in embedding:
            words1.append(word1)
            words2.append(word2)
        else:
            rows.append([word1, word2, None])

    embeddings1 = embedding[words1]
    embeddings2 = embedding[words2]
    dots = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=-1)
    norms1 = tf.sqrt(tf.reduce_sum(tf.multiply(embeddings1, embeddings1), axis=-1))
    norms2 = tf.sqrt(tf.reduce_sum(tf.multiply(embeddings2, embeddings2), axis=-1))

    similarities = dots / (norms1 * norms2)
    df = pd.DataFrame({"word1": words1, "word2": words2, "similarity": similarities}) 
    df_na = pd.DataFrame(rows, columns=["word1", "word2", "similarity"])
    return pd.concat([df, df_na])

def word_similarity_datasets():
    return ["Card-660", "MC-30", "MEN-TR-3k", "MTurk-287", "MTurk-771", "RG-65", "RW-STANFORD", "SimLex999", "SimVerb-3500", "WS-353-ALL", "WS-353-REL", "WS-353-SIM", "YP-130"]

def evaluate_word_similarity(embedding, dataset_names=None):
    """
    Evaluate embedding performance on word similarity tasks.
    
    Args:
        embedding: embedding as pwe.embeddings.Embedding
        dataset_names (list): List of dataset names to evaluate on, as names or paths to TSV files.
            If None, evaluate on all datasets that this module provides.
    """
    if dataset_names is None:
        dataset_names = word_similarity_datasets()
    rows = []
    for dataset in dataset_names:
        eval_df = _get_eval_file(dataset)
        embedding_df = embedding_similarities(eval_df, embedding)
        embedding_df = embedding_df.dropna()

        merged_df = pd.merge(eval_df, embedding_df, on=["word1", "word2"])
        corr, p_value = rank_correlation(merged_df["similarity_x"], merged_df["similarity_y"])

        rows.append([dataset, corr, len(merged_df), p_value])
        #print(f"Rank Correlation {corr}")

    return pd.DataFrame(rows, columns=["Dataset", "Rank Correlation", "No. of Observations", "p-value"])

def nearest_neighbors(embedding, words, K=25, epsilon=1e-10):
    """
    Get the K nearest neighbors using cosine distance
    
    Args:
        embedding: embedding as pwe.embeddings.Embedding
        words (list): List of words for which the nearest neighbors are fetched
        K (int): the number of nearest neighbors per word
        epsilon (float): the noise injected to the embedding to numerical issues. For most uses, the default value is ok.
    """
    e = copy.deepcopy(embedding)
    noise = tf.random.normal(shape=e.theta.shape, dtype=tf.float64) * epsilon
    e.theta.assign_add(noise)
    words = [wd for wd in words if wd in embedding]
    r = len(words)
    X = embedding[words]

    inv_vocab = {v: k for k, v in e.vocabulary.items()}
    eliminate = list(range(len(inv_vocab)))
    eliminate = [-100.0 if ("_c" in inv_vocab[i] or inv_vocab[i] in words) else 0.0 for i in eliminate]
    eliminate = tf.transpose(tf.constant([eliminate], dtype=tf.float64))

    X, _ = tf.linalg.normalize(X, axis=1)

    theta, _ = tf.linalg.normalize(e.theta, axis=1)
    e.theta.assign(theta)

    Ex = tf.linalg.tensordot(e.theta, X, axes=[1,1])
    tiled_eliminate = tf.tile(eliminate, [1,r])
    Ex = tf.add(Ex, tiled_eliminate)

    rows = []

    for i in range(r):
        y_hat = Ex[:,i]
        _, tops = tf.math.top_k(y_hat, k=K)
        topwords = [inv_vocab[int(i)] for i in tops]
        rows.append([words[i]] + topwords)
    
    columns = ["target"] + [f"@{i+1}" for i in range(K)]
    df = pd.DataFrame(rows, columns=columns)
    return df

def evaluate_analogy(embedding, dataset, K=25):
    """
    Evaluate embedding performance on word analogy tasks.
    
    Args:
        embedding: embedding as pwe.embeddings.Embedding
        dataset (pd.DataFrame): Dataframe where each row has four words, word1 - word2 + word3 ≈ word4
    """
    df = dataset
    e = embedding
    columns = df.columns[:4]
    def words_in_e(row):
        w1, w2, w3, w4 = row[columns]
        # force cast to str
        w1, w2, w3, w4 = str(w1), str(w2), str(w3), str(w4)

        return w1 in e and w2 in e and w3 in e and w4 in e 

    df["included"] = df.apply(words_in_e, axis=1)
    df = df[df["included"]]
    df = df[columns]
    r = len(df)
    target_words = list(df[columns[-1]])

    X1 = embedding[df[columns[0]]]
    X2 = embedding[df[columns[1]]]
    X3 = embedding[df[columns[2]]]
    X = X1 - X2 + X3

    inv_vocab = {v: k for k, v in e.vocabulary.items()}
    eliminate = list(range(len(inv_vocab)))
    eliminate = [0.0 if "_c" in inv_vocab[i] else 1.0 for i in eliminate]
    eliminate = tf.transpose(tf.constant([eliminate], dtype=tf.float64))

    X, _ = tf.linalg.normalize(X, axis=1)
    Ex = tf.linalg.tensordot(e.theta, X, axes=[1,1])
    tiled_eliminate = tf.tile(eliminate, [1,r])
    Ex = tf.multiply(Ex, tiled_eliminate)

    rows = []

    for i in range(r):
        y_hat = Ex[:,i]
        _, tops = tf.math.top_k(y_hat, k=K)
        topwords = [inv_vocab[int(i)] for i in tops]
        correct = target_words[i] in topwords
        rows.append(topwords + [correct])
    
    df = pd.DataFrame(rows)
    return df



###############################
# BILINGUAL LEXICON INDUCTION #
###############################

def bli(pairs, e, precision=[1,5,15], reverse=False):
    """
    Calculate the Bilingual Lexicon Induction performance of a crosslingual word embedding.
    
    Args:
        pairs: list of word pairs
        e: embedding as a pwe.Embedding
        precision: Precision level of the BLI score.
        reverse: Reverse the prediction; target language becomes the source language and vice versa.
    """
    if reverse:
        pairs = [(w2,w1) for w1,w2 in pairs]

    pairs_prime = []
    for w1, w2 in pairs:
        if w1 in e and w2 in e:
            pairs_prime.append((w1,w2))
        else:
            warnings.warn(f"Pair {w1}~{w2} not in embedding. Skipping...")

    pairs = pairs_prime
    l1, l2 = pairs[0][0].split("_")[-1], pairs[0][1].split("_")[-1]
    print(l1, "->", l2)

    vocab = [wd for wd in e.vocabulary if "_c" != wd[-2:]]
    assert e.theta.shape[1] < e.theta.shape[0]
    dim = e.theta.shape[1]
    embedding = Embedding(set(vocab), dimensionality=dim)

    embedding[vocab] = e[vocab]
    embedding_normalized, _ = tf.linalg.normalize(embedding[vocab], axis=1)
    embedding[vocab] = embedding_normalized
    
    source_words = [p[0] for p in pairs]
    target_words = [p[1] for p in pairs]

    target_vocab = [wd for wd in vocab if f"_{l2}" in wd]
    E_target = embedding[target_vocab]
    E_source = embedding[source_words]

    A = tf.tensordot(E_target, E_source, axes=[1,1])
    rows = []
    print("A shape", A.shape)
    for i, target_word in enumerate(target_words):
        source_word = source_words[i]
        y_hat = A[:,i]
        _, tops = tf.math.top_k(y_hat, k=max(precision))
        topwords = [target_vocab[i] for i in tops]
        row_correct = []
        for p in precision:
            topwords_p = topwords[:p]
            x = target_word in topwords_p
            row_correct.append(x)

        row = [source_word, target_word] + row_correct + topwords
        rows.append(row)

    columns = ["source", "target"] + [f"p@{p}" for p in precision] + [f"guess-{g}" for g in range(max(precision))]
    return pd.DataFrame(rows, columns=columns)
