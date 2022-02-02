from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
import scipy.spatial.distance
import tensorflow as tf

text = open("wiki.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

dim = 25
e = Embedding(vocabulary=vocabulary, dimensionality=dim)

# Perform MAP estimation
e = map_estimate(e, text, model="sgns", epochs=3)

# Do some sanity checks
print("Cosdist 'this', 'this'", scipy.spatial.distance.cosine(e["this"], e["this"]))
print("Cosdist 'this', 'that'", scipy.spatial.distance.cosine(e["this"], e["that"]))
print("Cosdist 'this', 'man'", scipy.spatial.distance.cosine(e["this"], e["man"]))
print("Cosdist 'this', 'motion'", scipy.spatial.distance.cosine(e["this"], e["motion"]))