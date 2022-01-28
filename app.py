from probabilistic_word_embeddings.models import sgns_likelihood, generate_cbow_batch, generate_sgns_batch
import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding
import tensorflow as tf

text = open("nietzsche.txt").read().lower().split()
vocabulary = set(text)

vocab_size = len(vocabulary)
dim = 25
e = Embedding(vocabulary=vocabulary, dimensionality=dim)

i = tf.constant(text[1:3])
j = tf.constant(text[3:5])
print(i,j)
x = 1

batch = i,j,x
print(e[i])

print(batch)

i, j, x = batch
print(i.shape, j.shape)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(1000):
	batch = generate_sgns_batch([tf.constant(text)], D=100, ws=5, ns=5, batch=2, start_ix=0, dataset_ix=0)
	i,j,x = batch
	#print(batch)
	objective = lambda: - tf.reduce_sum(sgns_likelihood(batch, e)) + e.log_prob()
	step_count = opt.minimize(objective, [e.theta]).numpy()
	#print(step_count)
	print(objective().numpy())

print(i)