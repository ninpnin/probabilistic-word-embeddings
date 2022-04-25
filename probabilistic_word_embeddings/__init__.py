"""
A collection of algorithms for the estimation and evaluation of different kinds of Bernoulli Word Embeddings.

We strive to keep all functionality clear, concise and easy to use.
For instance, MAP estimation of the basic Bernoulli Embeddings only requires a couple of lines of code:

```py
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
import tensorflow as tf

with open("wiki.txt") as f:
    text = f.read().lower().split()
text, vocabulary = preprocess_standard(text)

e = Embedding(vocabulary=vocabulary, dimensionality=100)

# Perform MAP estimation
e = map_estimate(e, text, model="cbow", ws=5, epochs=15)

# Evaluate performance on word similarity tasks
similarity = evaluate_word_similarity(e)
print(similarity)

# Save the embedding
e.save("embeddings.pkl")
```

The estimation of _dynamic_ Bernoulli embeddings is relatively straightforward, too.
Additionally, informative priors can be added to trace shifts in word meaning:

```py
import networkx as nx
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

with open("wiki.txt") as f:
    text = f.read().lower().split()
text, vocabulary = preprocess_standard(text)

g = nx.Graph()
g.add_edge("this", "that")
dim = 100
e = LaplacianEmbedding(vocabulary, dim, g)
# Perform MAP estimation

e = map_estimate(e, text, model="cbow", ws=5, epochs=15)

# Evaluate performance on word similarity tasks
similarity = evaluate_word_similarity(e)
print(similarity)

# Save the embedding
e.save("embeddings.pkl")
```
"""