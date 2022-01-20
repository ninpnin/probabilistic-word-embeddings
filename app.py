from probabilistic_word_embeddings.models.base_models import sgns_likelihood, generate_cbow_batch, generate_sgns_batch
import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding
import tensorflow as tf

text = open("nietzsche.txt").read().lower().split()
vocabulary = set(text)

vocab_size = len(vocabulary)
dim = 25
theta = Embedding(vocabulary=vocabulary, dimensionality=dim)

i = tf.constant(text[1:3])
j = tf.constant(text[3:5])
print(i,j)
x = 1

batch = i,j,x
print(theta[i])
lp = sgns_likelihood(batch, theta)

print(lp)


batch = generate_sgns_batch([tf.constant(text)], D=100, ws=5, ns=5, batch=2, start_ix=0, dataset_ix=0)

print(batch)

i, j = batch
print(i.shape, j.shape)

batch = i,j,x
lp = sgns_likelihood(batch, theta)

print(lp)