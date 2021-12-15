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

###################
# WORD SIMILARITY #
###################

def _cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _eval_files(eval_folder):
    results = dict()
    print("List test datasets...")
    """Read the filename for each file in the evaluation directory"""
    for filename in os.listdir(eval_folder):
        if not filename in results:
            results[eval_folder + filename] = []

    return results

def _evaluate_embedding(embedding, word_dictionary, evaluation_files):
    missed_pairs = {}
    missed_words = {}
    results = {}

    print("Embedding shape", embedding.shape)
    print("Max key", max(word_dictionary.keys()))

    for filename in evaluation_files:
        pairs_not_found, total_pairs = 0, 0
        words_not_found, total_words = 0, 0

        results[filename] = []

        with open(filename) as f:
            file_similarity = []
            embedding_similarity = []
            for line in f:
                w1, w2, val = line.split()
                w1, w2, val = w1.lower(), w2.lower(), float(val)
                total_words += 2
                total_pairs += 1
                if not w1 in word_dictionary:
                    words_not_found += 1
                if not w2 in word_dictionary:
                    words_not_found += 1

                if not w1 in word_dictionary or not w2 in word_dictionary:
                    pairs_not_found += 1
                else:
                    v1, v2 = embedding[word_dictionary[w1]], embedding[word_dictionary[w2]]
                    cosine = _cosineSim(v1, v2)
                    file_similarity.append(val)
                    embedding_similarity.append(cosine)

            rho, p_val = st.spearmanr(file_similarity, embedding_similarity)
            results[filename].append(rho)
            missed_pairs[filename] = (pairs_not_found, total_pairs)
            missed_words[filename] = (words_not_found, total_words)

    return results, missed_pairs, missed_words


def _weighted_average(results, missed_pairs, missed_words):
    """Compute statistics on results"""
    title = "{}| {}| {}| {}| {}| {} ".format("Filename".ljust(16),
                              "AVG".ljust(5), "MIN".ljust(5), "MAX".ljust(5),
                              "STD".ljust(5), "Missed words/pairs")
    print(title)
    print("="*len(title))

    weighted_avg = 0
    total_found  = 0

    for filename in sorted(results.keys()):
        average = np.mean(results[filename])
        minimum = min(results[filename])
        maximum = max(results[filename])
        std = np.std(results[filename])

        # For the weighted average, each file has a weight proportional to the
        # number of pairs on which it has been evaluated.
        # pairs evaluated = pairs_found = total_pairs - number of missed pairs
        pairs_found = missed_pairs[filename][1] - missed_pairs[filename][0]
        weighted_avg += pairs_found * average
        total_found  += pairs_found

        # ratio = number of missed / total
        ratio_words = missed_words[filename][0] / missed_words[filename][1]
        ratio_pairs = missed_pairs[filename][0] / missed_pairs[filename][1]
        missed_infos = "{:.0f}% / {:.0f}%".format(
                round(ratio_words*100), round(ratio_pairs*100))

        filename = filename.split("/")[-1]
        print("{}| {:.3f}| {:.3f}| {:.3f}| {:.3f}| {} ".format(
              filename.ljust(16),
              average, minimum, maximum, std, missed_infos.center(20)))

    print("-"*len(title))
    print("{0}| {1:.3f}".format("W.Average".ljust(16),
                                weighted_avg / total_found))
    return (weighted_avg / total_found)

def wordsim(rhos, vocab, eval_folder="data/eval/"):
    """
    Evaluates the accuracy of an embedding matrix based on word similarity.

    The evaluation uses 13 word similarity datasets.

    Args:
        rhos: The embedding matrix. A numpy array of shape (vocab_size, dim)
        vocab: A dictionary from word types to their indices.
        eval_folder: the path to the folder containing the evaluation data sets.
    """
    files = _eval_files(eval_folder)
    performances, missed_pairs, missed_words = _evaluate_embedding(rhos, vocab, files)
    mean = _weighted_average(performances, missed_pairs, missed_words)
    return performances, mean


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
