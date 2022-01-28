from probabilistic_word_embeddings.models import sgns_likelihood, generate_cbow_batch, generate_sgns_batch
import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import filter_rare_words
import tensorflow as tf
import scipy.spatial.distance
import progressbar

text = open("wiki.txt").read().lower().split()
text = filter_rare_words(text)

vocabulary = set(text)
print("'man' in vocabulary", 'man' in vocabulary)

vocab_size = len(vocabulary)
print(f"Text length: {len(text)}, vocab size {vocab_size}")
dim = 25
e = Embedding(vocabulary=vocabulary, dimensionality=dim)

i = tf.constant(text[1:3])
j = tf.constant(text[3:5])
print(i,j)
print(e[i])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

data = tf.constant(text)
N = len(data)
batch_size = 25000
batches = N // batch_size
for epoch in range(100):
	print(f"Epoch {epoch}")
	#i,j,x = generate_sgns_batch(data, ws=5, ns=5, batch=2, start_ix=0)
	for batch in progressbar.progressbar(range(batches)):
		start_ix = batch_size * batch
		
		i,j,x  = generate_sgns_batch(data, ws=5, ns=5, batch=batch_size, start_ix=start_ix)
		objective = lambda: - tf.reduce_sum(sgns_likelihood(e, i, j, x=x)) + e.log_prob()
		step_count = opt.minimize(objective, [e.theta]).numpy()

	print("Cosdist 'this', 'this'", scipy.spatial.distance.cosine(e["this"], e["this"]))
	print("Cosdist 'this', 'that'", scipy.spatial.distance.cosine(e["this"], e["that"]))
	print("Cosdist 'this', 'man'", scipy.spatial.distance.cosine(e["this"], e["man"]))
	print("Cosdist 'this', 'motion'", scipy.spatial.distance.cosine(e["this"], e["motion"]))