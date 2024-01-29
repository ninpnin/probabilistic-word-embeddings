from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

print("Preprocess...")
with open("examples/data/wikismall.txt") as f:
	text = f.read().lower().split()
text, vocabulary = preprocess_standard(text)

train_size = 0.7
valid_size = 0.15

train_size = int(train_size * len(text))
valid_size = int(valid_size * len(text))

text_train = text[:train_size]
text_valid = text[train_size:train_size+valid_size]
text_test = text[train_size+valid_size:]
print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

dim = 100
e = Embedding(vocabulary=vocabulary, dimensionality=dim)

# Perform MAP estimation
e = map_estimate(e, text_train, model="cbow", ws=5, valid_data=text_valid, epochs=5, evaluate=False, early_stopping=True)

similarity = evaluate_word_similarity(e)
print(similarity)

# Do some sanity checks
print("Cosdist 'this', 'this'", cos_dist(e["this"], e["this"]))
print("Cosdist 'this', 'that'", cos_dist(e["this"], e["that"]))
print("Cosdist 'this', 'man'", cos_dist(e["this"], e["man"]))
print("Cosdist 'this', 'motion'", cos_dist(e["this"], e["motion"]))

# Evaluate on test set
test_ll = evaluate_on_holdout_set(e, text_test, model="cbow", ws=5)
test_acc = evaluate_on_holdout_set(e, text_test, model="cbow", ws=5, metric="accuracy")

print(f"Test set likelihood: {test_ll}")
print(f"Test set accuracy: {test_acc}")

e.save("embeddings.pkl")
