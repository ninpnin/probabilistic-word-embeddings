import numpy as np
import tensorflow as tf
import pickle
import warnings
import logging, colorlog
import sys
logging.TRAIN = 25
logging.addLevelName(logging.TRAIN, 'TRAIN')

def get_logger(loglevel, name="log"):
    handler = colorlog.StreamHandler(stream=sys.stdout)
    LOG_COLORS = {'DEBUG':'cyan', 'INFO':'green', 'TRAIN':'blue', 'WARNING':'yellow', 'ERROR': 'red', 'CRITICAL':'red,bg_white'}
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(asctime)s [%(levelname)s] %(white)s(%(name)s)%(reset)s: %(message)s',
        log_colors=LOG_COLORS,
        datefmt="%H:%M:%S",
        stream=sys.stdout))
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    logger.propagate = False

    return logger

# MAP estimation
#@tf.function
def shuffled_indices(data_len, batch_len):
    if type(data_len) == int: # One dataset => List<Int>
        indices = tf.range(data_len // batch_len - 1) * batch_len
        indices = tf.random.shuffle(indices)
        return indices
    else: # Multiple datasets => List<(Int, Int)>
        lists = [shuffled_indices(data, batch_len) for data in data_len]
        data_indices = [tf.ones(indices.shape, dtype=tf.int32) * ix for ix, indices in enumerate(lists)]
        zipped = list(zip(lists, data_indices))
        
        array = [tf.concat([[a],[b]],axis=0) for a,b in zipped]
        array = tf.concat(array, axis=1)
        array = tf.transpose(array)
        array = tf.random.shuffle(array)
        return array

def dict_to_tf(d):
    keys = list(d.keys())
    values = list(d.values())
    
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="class_weight"
    )
    return table

def align(e_reference, e, words, words_reference=None):
    """
    Orthogonally rotate an embedding to minimize the Euclidian distance
    to a reference embedding
    """

    if words_reference is None:
        words_reference = words
    else:
        assert len(words_reference) == len(words), "'words' and 'words_reference' need to be of equal length"
    assert all([(w in e) for w in words]), 'not all words are in e'
    assert all([(w in e_reference) for w in words_reference]), 'not all words are in e_reference'
    assert e.theta.shape[1] == e_reference.theta.shape[1], f'embedding size needs to be the same e:{e.theta.shape[1]}, e_reference:{e_reference.theta.shape[1]}'
    
    # Alert if the numbers of words are less than (undetermined system): (D-1)/2
    if len(words) < (e.theta.shape[1]-1)/2:
        warnings.warn(f"Numbers of words={len(words)} is less than (D-1)/2={(e.theta.shape[1]-1)/2}, and the system is thus undetermined.")
    

    x_prime = e_reference[words_reference]
    x = e[words]

    a = tf.tensordot(x_prime, x, axes=(0,0))
    a_sum = tf.reduce_mean(a)

    s, u, v = tf.linalg.svd(a, full_matrices=True)

    W = v @ tf.transpose(u)
    e.theta.assign(e.theta @ W)

    return e

def _normalize_word(wd):
    context_vector = "_c" in wd
    wd = wd.split("_")[0]

    if context_vector:
        wd = f"{wd}_c"

    return wd

def transfer_embeddings(e_source, e_target, ignore_group=False):
    """
    Transfer all embeddings for shared vocabulary from one embedding to another.
    """
    source_vocab = e_source.vocabulary
    target_vocab = e_target.vocabulary
    if not ignore_group:
        vocab = {wd: ix for wd, ix in target_vocab.items() if wd in source_vocab}

        # Make unique
        vocab = {v: k for k, v in vocab.items()}
        words = list(vocab.values())
        e_target[words] = e_source[words]
    else:
        source_vocab_normalized = {_normalize_word(wd): wd for wd in source_vocab}
        target_vocab = {wd: ix for wd, ix in target_vocab.items() if _normalize_word(wd) in source_vocab_normalized}
        target_vocab = {v: k for k, v in target_vocab.items()}
        target_words = list(target_vocab.values())
        source_words = [source_vocab_normalized[_normalize_word(wd)] for wd in target_words]
        e_target[target_words] = e_source[source_words]
    return e_target
