from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import mean_field_vi
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity
import numpy as np
import copy

text = open("examples/data/wikismall.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

dim = 100
e = Embedding(vocabulary, dim)

# Perform mean-field variatinal inference
e_mu, e_std = mean_field_vi(e, text, model="cbow", ws=5, epochs=1)
e = copy.deepcopy(e_mu)

for i in range(5):
    vocabulary = list(e.vocabulary)
    z = e_mu[vocabulary]
    z += np.random.randn(z.shape[0], z.shape[1]) * e_std[vocabulary]
    e[vocabulary] = z

    similarity = evaluate_word_similarity(e)
    similarity = (similarity["Rank Correlation"] * similarity["No. of Observations"]).sum() / similarity["No. of Observations"].sum()
    print(f"Sample {i}, Word Similarity", similarity)