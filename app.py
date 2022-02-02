from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import filter_rare_words, downsample_common_words
from probabilistic_word_embeddings.estimation import map_estimate
import scipy.spatial.distance
import tensorflow as tf

text = open("wiki.txt").read().lower().split()
text, counts = filter_rare_words(text)
text = downsample_common_words(text, counts)

vocabulary = set(text)
vocab_size = len(vocabulary)
print(f"Text length: {len(text)}, vocab size {vocab_size}")

dim = 25
e = Embedding(vocabulary=vocabulary, dimensionality=dim)

# Perform MAP estimation
e = map_estimate(e, text, model="sgns", epochs=1)

# Do some sanity checks
print("Cosdist 'this', 'this'", scipy.spatial.distance.cosine(e["this"], e["this"]))
print("Cosdist 'this', 'that'", scipy.spatial.distance.cosine(e["this"], e["that"]))
print("Cosdist 'this', 'man'", scipy.spatial.distance.cosine(e["this"], e["man"]))
print("Cosdist 'this', 'motion'", scipy.spatial.distance.cosine(e["this"], e["motion"]))