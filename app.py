from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

eval_df = get_eval_file("MC-30")
print(eval_df)
text = open("wiki.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

dim = 100
e = Embedding(vocabulary=vocabulary, dimensionality=dim)
estimate_df = embedding_similarities(eval_df, e)
similarity = evaluate_word_similarity(e)
print(similarity)

# Perform MAP estimation
e = map_estimate(e, text, model="cbow", ws=5, epochs=15)

similarity = evaluate_word_similarity(e)
print(similarity)

# Do some sanity checks
print("Cosdist 'this', 'this'", cos_dist(e["this"], e["this"]))
print("Cosdist 'this', 'that'", cos_dist(e["this"], e["that"]))
print("Cosdist 'this', 'man'", cos_dist(e["this"], e["man"]))
print("Cosdist 'this', 'motion'", cos_dist(e["this"], e["motion"]))