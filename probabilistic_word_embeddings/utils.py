import numpy as np
import tensorflow as tf
import pickle
    
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

# Sparse matrices
def dict_to_sparse(tensor_dict, shape):
    indices = []
    values = []
    
    for index, value in tensor_dict.items():
        ix1, ix2 = index
        
        index = [ix1, ix2]
        indices.append(index)
        values.append(value)
        
    tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape, dtype=dtype)

def scipy_to_tf_sparse(X, dtype=None):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    values = coo.data
    if dtype is not None:
        values = tf.constant(values, dtype=dtype)

    return tf.SparseTensor(indices, values, coo.shape)

def save_sparse(stensor, fpath):
    tensor_f = open(fpath, "wb")
    pickle.dump(stensor, tensor_f)
    
def load_sparse(fpath):
    tensor_f = open(fpath, 'rb')
    stensor = pickle.load(tensor_f)
    tensor_f.close()
    return stensor

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

# Side-information
def dynamic_si(si_path, d_folder="fits/"):
    d_filename = d_folder + "dictionary.pkl"
    d = pickle.load(open(d_filename, "rb"))

    word_dict = {}
    f = open(si_path, "r")
    for line in f:
        s = line.split()
        key, value = s[0], float(s[1])
        key = d[key]
        word_dict[key] = value

    return word_dict

# Combined function x : c(x) = b(a(x))
def transitive_dict(a, b):
    c = {}
    
    for key_a in a.keys():
        key_b = a[key_a]
        if key_b in b:
            c[key_a] = b[key_b]
        
    return c

def inverse_dict(d):
    assert len(set(d.keys())) == len(set(d.values())), "Not invertible"
    inv_d = {v: k for k, v in d.items()}

    return inv_d
    