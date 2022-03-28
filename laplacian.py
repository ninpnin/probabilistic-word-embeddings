import networkx as nx
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

text = open("wiki.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

g = nx.Graph()
g.add_edge("this", "that")
dim = 100
e = LaplacianEmbedding(vocabulary, dim, g)
# Perform MAP estimation

e = map_estimate(e, text, model="cbow", ws=5, epochs=15)
similarity = evaluate_word_similarity(e)
print(similarity)