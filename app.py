from probabilistic_word_embeddings.models.base_models import SGNS
import numpy as np
from probabilistic_word_embeddings.embeddings import Embedding
import tensorflow as tf

text = open("nietzsche.txt").read().lower().split()
vocabulary = set(text)

vocab_size = len(vocabulary)
dim = 25

model = SGNS(vocab_size=vocab_size, dim=dim)

print("SGNS")
print(model)

theta = Embedding(vocabulary=vocabulary, dimensionality=dim)

i = tf.constant(text[1:3])
j = tf.constant(text[3:5])
print(i,j)
x = 1

batch = i,j,x
print(theta[i])
lp = model.loss(batch, theta)

print(lp)


