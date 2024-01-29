from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity

text = open("examples/data/wikismall.txt").read().lower().split()
text, vocabulary = preprocess_standard(text)
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

dim = 100
e = Embedding(vocabulary, dim)
e = map_estimate(e, text, model="sgns", ws=2, epochs=5)
similarity = evaluate_word_similarity(e)
print(similarity)

e.save("sgns_wikismall.pkl")