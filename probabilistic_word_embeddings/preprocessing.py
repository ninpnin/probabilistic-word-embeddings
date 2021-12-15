import re, pickle, os, json
import progressbar
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from probabilistic_word_embeddings.utils import dict_to_tf, transitive_dict

def _batch_data(paths, batch=50000000):
    #print("Process file which starts with: '", data_strs[0][:10], "[...]'")
    for fpath in progressbar.progressbar(paths):
        print("Preprocess", fpath, "...")
        s = open(fpath, encoding="utf8", errors='ignore').read()
        if batch == None:
            yield s
        else:
            N = len(s)
            batches = N // batch + 1

            print("Split to", batches, "batches...")
            for i in progressbar.progressbar(range(batches)):
                #print("Batch", i, "out of", batches)
                start_ix = i * batch
                end_ix = start_ix + batch
                s_i = s[start_ix : end_ix]
                
                yield s_i

# Discard words with the probability of 1 - sqrt(10^-5 / freq)
# Initially proposed by Mikolov et al. (2013)
#@tf.function
def _discard(data, counts, cutoff=0.00001):
    print("Discard some instances of the most common words...")
    N = sum(counts.values())
    counts_tf = dict_to_tf(counts)
    
    frequencies = counts_tf.lookup(data) / N
    # Discard probability based on relative frequency
    probs = 1. - tf.sqrt(cutoff / frequencies)
    # Randomize and fetch by this probability
    rands = tf.random.uniform(shape=data.shape, dtype=tf.float64)
    kepts = rands > probs
    indices = tf.where(kepts)
    newdata = tf.gather_nd(data, indices)

    # Convert tf.int32 to tf.int64.
    newdata = tf.cast(newdata, dtype=tf.int64)

    return newdata

def _get_tokenizer(generator, limit):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(generator)
    config = tokenizer.get_config()
    word_counts = json.loads(config["word_counts"])
    V = tf.constant(list(word_counts.values())) >= limit
    V = tf.math.count_nonzero(V).numpy() + 1
    # +1 since the tokenizer only takes n-1 most common word types into account
    
    tokenizer.num_words = V
    return tokenizer

def _fit_tokenizer(paths, limit):
    generator = _batch_data(paths)
    tokenizer = _get_tokenizer(generator, limit)
    
    #tokenizer.
    config = tokenizer.get_config()
    index_word = json.loads(config["index_word"])
    index_word = {int(k): v for k,v in index_word.items()}
    word_index = json.loads(config["word_index"])
    word_counts = json.loads(config["word_counts"])
    index_freq = transitive_dict(index_word, word_counts)
    
    return tokenizer, word_index, word_counts, index_freq

def _preprocess_strs(paths, limit=5, cutoff=0.00001):
    tokenizer, word_index, word_counts, index_freq = _fit_tokenizer(paths, limit)
    
    # Remove rare words from word_index
    word_index = dict(filter(lambda elem : word_counts[elem[0]] >= limit, word_index.items()))
    print("Convert texts to sequences...")
    generator = _batch_data(paths, batch=None)
    data = tokenizer.texts_to_sequences(generator)
    
    datas = [_discard(tf.constant(d), index_freq, cutoff=cutoff) for d in data]
    return datas, word_index

def _fix_offsets(data, vocab):
    new_data = [dataset - tf.ones(dataset.shape, dtype=dataset.dtype) for dataset in data]
    new_vocab = {key: value - 1 for key, value in vocab.items()}
    return new_data, new_vocab


def _preprocess(inpath, limit, cutoff):
    print("Preprocess static data:")
    folder          = inpath
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        print(filename)
    
    paths = [folder + filename for filename in filenames]
    data, vocab = _preprocess_strs(paths, limit=limit, cutoff=cutoff)
    data = tf.concat(data, axis=0)
    data_arrays = [data]

    data_arrays, vocab = _fix_offsets(data_arrays, vocab)
    vocab_size = max(vocab.values()) + 1
    return data_arrays, vocab, vocab_size
    
def _preprocess_dynamic(inpath, limit, cutoff):
    print("Preprocess dynamic data.")
    folder = inpath

    V = 0
    vocab = {}
    
    filenames = sorted(os.listdir(folder))
    paths = [folder + filename for filename in filenames]
    
    data_arrays, vocab = _preprocess_strs(paths, limit=limit, cutoff=cutoff)
    data_arrays, vocab = _fix_offsets(data_arrays, vocab)
    vocab_size = max(vocab.values()) + 1
    return data_arrays, vocab, vocab_size
    
def _preprocess_crosslingual(inpath, limit, cutoff):
    print("Preprocess crosslingual data.")
    folder = inpath

    V = 0
    full_unigram = {}
    
    data_arrays = []

    for f in sorted(os.listdir(folder)):
        if not os.path.isfile(folder + f):
            continue
        
        langcode = f.split("_")[-1].split(".")[0]
        path = folder + f
        data, unigram = _preprocess_strs([path], limit=limit, cutoff=cutoff)
        data = tf.concat(data, axis=0)
        
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)

        for key in unigram.keys():
            if unigram[key] - data_min <= data_max:
                full_unigram[key + "_" + langcode] = unigram[key] - data_min + V

        data += V
        data_arrays.append(data)
        
        # Increment since each language has unique word indices.
        V = max(full_unigram.values()) + 1

    vocab_size = max(full_unigram.values()) + 1
    return data_arrays, full_unigram, vocab_size

def preprocess(inpath, data_type="monolingual", limit=5, cutoff=0.0001):
    """
    Preprocess all files in a given folder. Remove rare words, skip some instances of the most common words.
    
    Returns a list of NumPy arrays with indices of word types, and the vocabulary associated with those indices.

    Args:
        inpath: Path to the folder containing the data
        data_type: monolingual / crosslingual / dynamic
        limit: Words occuring fewer times than this are discarded from the data and the vocab.
        cutoff: Frequency above which some occurences of a word are randomly discarded

    Returns:
        data: Data as a list of tf.constants
        vocab: Vocabulary as a str->int dictionary from words to their numeric indices
        vocab_size: Size of the vocabulary as an integer
    """
    print("Preprocess...")
    if data_type == "monolingual":
        return _preprocess(inpath, limit, cutoff)
    elif data_type == "crosslingual":
        return _preprocess_crosslingual(inpath, limit, cutoff)
    elif data_type == "dynamic":
        return _preprocess_dynamic(inpath, limit, cutoff)
    else:
        return ValueError("Data type not recognized.")


if __name__ == '__main__':
    info = """This script:

    1. Reads a txt file, lowercases it and removes special charactes
    2. Splits the data by whitespace and counts the occurences of each word
    3. Removes words that only occur 'limit' or fewer times
    4. Probabilistically discards some of the common words
    5. Converts the data to their numeric indices in a dictionary
    6. Saves the data to data.npy and the dictionary to dictionary.npy
    """

    import argparse
    parser = argparse.ArgumentParser(description=info, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', "--type", default="static", help="Type of the preprocessing: static, dynamic or crosslingual.")
    parser.add_argument('-l', "--limit", default=5, type=int, help="Only include words that occur at least 'limit' times in the data")
    parser.add_argument('-c', "--cutoff", default=0.00001, type=float, help="Discard process (Mikolov et al 2013) cutoff frequency.")
    parser.add_argument("--folder", default=None, help="Path to the folder where the input .txt files are located. Defaults are raw/static/ raw/dynamic/ and raw/crosslingual for their respective preprocessing types.")
    args = parser.parse_args()

    if args.type == "crosslingual":
        preprocess_crosslingual(args)
    elif args.type == "dynamic":
        preprocess_dynamic(args)
    else:
        preprocess(args)
