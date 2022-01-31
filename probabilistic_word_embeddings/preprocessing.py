import re, pickle, os, json
import progressbar
import numpy as np
import tensorflow as tf

from probabilistic_word_embeddings.utils import dict_to_tf, transitive_dict

def filter_rare_words(data, limit=5):
    vocab = {}
    for wd in data:
        vocab[wd] = vocab.get(wd, 0) + 1
    
    outdata = []
    for wd in data:
        if vocab[wd] >= limit:
            outdata.append(wd)

    return outdata