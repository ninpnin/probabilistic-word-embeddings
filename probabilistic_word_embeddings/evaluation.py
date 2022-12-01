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

###################
# INTRINSIC EVALUATION #
###################

from .models import generate_sgns_batch, sgns_likelihood
from .models import generate_cbow_batch, cbow_likelihood
def evaluate_on_holdout_set(embedding, data, model="sgns", ws=5, ns=5, batch_size=25000, reduce_mean=True):
    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)
    valid_batches = len(data) // batch_size

    valid_ll = tf.constant([], dtype=tf.float64)
    for batch in progressbar.progressbar(range(valid_batches)):
        start_ix = batch_size * batch
        if model == "sgns":
            i,j,x  = generate_sgns_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
            valid_ll = tf.concat([valid_ll, sgns_likelihood(embedding, i, j, x=x)], axis=0)
        elif model == "cbow":
            i,j,x  = generate_cbow_batch(data, ws=ws, ns=ns, batch=batch_size, start_ix=start_ix)
            valid_ll = tf.concat([valid_ll, cbow_likelihood(embedding, i, j, x=x)], axis=0)

    if reduce_mean:
        return tf.reduce_mean(valid_ll)
    else:
        return valid_ll

###################
# WORD SIMILARITY #
###################

def get_eval_file(dataset_name):
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
        eval_df = get_eval_file(dataset)
        embedding_df = embedding_similarities(eval_df, embedding)
        embedding_df = embedding_df.dropna()

        merged_df = pd.merge(eval_df, embedding_df, on=["word1", "word2"])
        corr, p_value = rank_correlation(merged_df["similarity_x"], merged_df["similarity_y"])

        rows.append([dataset, corr, len(merged_df), p_value])
        #print(f"Rank Correlation {corr}")

    return pd.DataFrame(rows, columns=["Dataset", "Rank Correlation", "No. of Observations", "p-value"])

def evaluate_analogy(embedding, dataset, K=25):
    """
    Evaluate embedding performance on word similarity tasks.
    
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

def bli(pairs, embedding, vocab, precision=[1,5,15], reverse=False):
    """
    Calculates the Bilingual Lexicon Induction performance of a crosslingual word embedding.
    
    Args:
        lang1: Source language
        lang2: Target language
        embedding: embedding as a numpy matrix
        folder: folder of the vocab file
        precision: Precision level of the BLI score.
        reverse: Reverse the prediction; target language becomes the source language and vice versa.
    """

    inv_dictionary = {v: k for k, v in vocab.items()}
    vocab_size = len([wd for wd in vocab if len(wd.split("_")) == 2])

    assert embedding.shape[1] < 1000
    dim = embedding.shape[1]

    # Normalize word vectors
    for wd in range(embedding.shape[0]):
        norm = np.linalg.norm(embedding[wd])
        if norm > 0.0:
            embedding[wd] = embedding[wd] / norm
    
    correct = { precision_level: 0 for precision_level in precision}
    N = 0

    if reverse:
        pairs = [(wd_b, wd_a) for wd_a, wd_b in pairs]

    TARGET_LAN = pairs[0][1].split("_")[-1]


    pairs = [(txt1, txt2) for txt1, txt2 in pairs if txt1 in vocab]
    outdim = len(pairs)

    ETE = np.zeros((outdim, dim))
    for ix, (txt1, txt2) in enumerate(pairs):
        ETE[ix] = embedding[vocab[txt1]]

    target_vocab = {}
    for wd2 in range(vocab_size):
        langcode = inv_dictionary[wd2].split("_")[-1]
        if langcode == TARGET_LAN:
            target_vocab[inv_dictionary[wd2]] = wd2

    ETE_T = np.zeros((len(target_vocab), dim))
    for ix, (_, wd2) in enumerate(target_vocab.items()):
        ETE_T[ix] = embedding[wd2]

    ETE = ETE @ ETE_T.T
    assert ETE.shape[0] != dim and ETE.shape[1] != dim
    print(ETE)
    for ix, (txt1, txt2) in progressbar.progressbar(list(enumerate(pairs))):                    
        wd1 = vocab[txt1]
    
        dists = {}
        for ix2, (_, wd2) in enumerate(target_vocab.items()):
            #vec2 = embedding[wd2]
            dot = ETE[ix, ix2]
            dist = 1 - dot
            dists[inv_dictionary[wd2]] = dist
        
        dists = sorted(dists.keys(), key=lambda k: dists[k])
        
        #print(dists_head)
        for precision_level in precision:
            dists_head = dists[:precision_level]
            if txt2 in dists_head:
                correct[precision_level] += 1
        
        N += 1
        #print(txt1, txt2, found)

    for precision_level in precision:
        correct_no = correct[precision_level]
        print("BLI", correct_no, "/", N, "=", correct_no / N, "precision @", precision_level)
    return { p: c / N for p, c in correct.items()}
