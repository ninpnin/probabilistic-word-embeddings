"""
Definitions of the prior distributions of the embeddings.
"""
import numpy as np
import tensorflow as tf
import networkx as nx
import random, pickle
import progressbar
from .utils import dict_to_tf
import warnings

class Embedding:
    '''
    Generic class for probabilistic embeddings.
    '''
    def __init__(self, vocabulary=None, dimensionality=100, lambda0=1.0, shared_context_vectors=True, saved_model_path=None):
        if saved_model_path is None:
            if not isinstance(vocabulary, set):
                raise TypeError("vocabulary must be a set")
            keys = list(vocabulary)
            if not shared_context_vectors:
                keys = keys + list(set([key + "_c" for key in keys]))
                self.vocabulary = {wd: ix for ix, wd in enumerate(sorted(keys))}
            else:
                context_keys = {key + "_c": key.split("_")[0] + "_c" for key in keys}
                all_indices = {key: ix for ix, key in enumerate(list(keys) + list(set(context_keys.values())))}
                word_dict = {key: all_indices[key] for key in keys}
                context_dict = {key: all_indices[value] for key, value in context_keys.items()}
                self.vocabulary = {**word_dict, **context_dict}
                assert max(self.vocabulary.values()) + 1 == len(set(context_dict.values())) + len(keys)
            unique_parameters = len(set(self.vocabulary.values()))
            self.tf_vocabulary = dict_to_tf(self.vocabulary)
            self.theta = tf.Variable((np.random.rand(unique_parameters, dimensionality)- 0.5)/dimensionality, dtype=tf.float64)
            self.lambda0 = lambda0
        else:
            if type(saved_model_path) != str:
                raise TypeError("saved_model_path must be a str")
            with open(saved_model_path, "rb") as f:
                d = pickle.load(f)
            self.vocabulary = d["vocabulary"]
            self.tf_vocabulary = dict_to_tf(self.vocabulary)
            self.theta = tf.Variable(d["theta"])
            self.lambda0 = d["lambda0"]

    #@tf.function
    def __getitem__(self, item):
        if type(item) == str:
            return self.theta[self.vocabulary[item]]
        elif isinstance(item, list):
            item = tf.constant(item)
        ix = self.tf_vocabulary.lookup(item)
        return tf.gather(self.theta, ix, axis=0)

    def __setitem__(self, item, new_value):
        #self.theta = self.theta - self.theta[]
        if type(item) == str:
            item = [item]
            new_value = tf.expand_dims(new_value, axis=0)
        if isinstance(item, list):
            item = tf.constant(item)
        ix = self.tf_vocabulary.lookup(item)
        ix = list(ix.numpy())
        ix = [[i] for i in ix]
        old_value = self[item]
        old_scattered = tf.scatter_nd(ix, old_value, self.theta.shape)
        new_scattered = tf.scatter_nd(ix, new_value, self.theta.shape)
        
        self.theta = self.theta - old_scattered + new_scattered
    
    def __contains__(self, key):
        if type(key) == str:
            return key in self.vocabulary
        ix = self.tf_vocabulary.lookup(key)
        return ix != -1

    def log_prob(self, batch_size, data_size):
        """
        Calculate the log (prior) probability of the embedding taking its current value
        
        Args:
            batch_size (int): Batch size. Used to scale the log prob for the whole dataset.
            data_size (int): Whole dataset size. Used to scale the log prob for the whole dataset.
        
        Returns:
            Log probability as 1D tensor as tf.EagerTensor
        """
        plain_loss = - 0.5 * tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0
        return (batch_size / data_size) * plain_loss

    def save(self, path):
        """
        Save embedding as a pickled file
        
        Args:
            path (str): Path where the embedding is saved to as a str.
        """
        theta = self.theta.numpy()
        d = {}
        d["theta"] = theta
        d["vocabulary"] = self.vocabulary
        d["lambda0"] = self.lambda0
        if hasattr(self, 'lambda1'):
            d["lambda1"] = self.lambda1
        if hasattr(self, 'graph'):
            d["graph"] = self.graph

        with open(path, "wb") as f:
            pickle.dump(d, f)

class LaplacianEmbedding(Embedding):
    """
    Probabilistic embedding with a Laplacian graph prior
    """
    def __init__(self, vocabulary=None, dimensionality=100, graph=None, lambda0=1.0, lambda1=1.0, shared_context_vectors=True, saved_model_path=None):
        if saved_model_path is None:
            if type(graph) != nx.Graph:
                raise TypeError("graph must be a nx.Graph")
            self.lambda1 = lambda1
            super().__init__(vocabulary, dimensionality, lambda0=lambda0, shared_context_vectors=shared_context_vectors)
            for wd in list(graph.nodes):
                if wd in graph.nodes and wd not in self.vocabulary:
                    graph.remove_node(wd)
                    omitted_word_warning = f"'{wd}' does not exist in embedding vocabulary and will be omitted."
                    warnings.warn(omitted_word_warning)
            self.graph = graph
        else:
            with open(saved_model_path, "rb") as f:
                d = pickle.load(f)
            self.vocabulary = d["vocabulary"]
            self.tf_vocabulary = dict_to_tf(self.vocabulary)
            self.theta = tf.Variable(d["theta"])
            self.lambda0 = d["lambda0"]
            self.lambda1 = d["lambda1"]
            self.graph = d["graph"]

    def log_prob(self, batch_size, data_size):
        """
        Calculate the log (prior) probability of the embedding taking its current value
        
        Args:
            batch_size (int): Batch size. Used to scale the log prob for the whole dataset.
            data_size (int): Whole dataset size. Used to scale the log prob for the whole dataset.
        
        Returns:
            Log probability as tf.EagerTensor
        """
        g = self.graph
        triple = [(e_i, e_j, g[e_i][e_j].get("weight", 1.0)) for e_i, e_j in g.edges]
        edges_i, edges_j, weights = zip(*triple)
        edges_i, edges_j = tf.constant(edges_i), tf.constant(edges_j)
        weights = tf.constant(weights, dtype=tf.float64) * self.lambda1
        theta_i, theta_j = self[edges_i], self[edges_j]
        diffs = theta_i - theta_j
        squared_diffs = tf.reduce_sum(tf.multiply(diffs, diffs), axis=-1)
        weighted_diff_sum = tf.reduce_sum(tf.multiply(squared_diffs, weights))
        diagonal_sum = tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0
        plain_loss = - 0.5 * (diagonal_sum + weighted_diff_sum)
        return (batch_size / data_size) * plain_loss
