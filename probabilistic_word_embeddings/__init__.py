"""
A collection of algorithms for the estimation and evaluation of different kinds of Bernoulli Word Embeddings.

We strive to keep all functionality clear, concise and easy to use.
For instance, MAP estimation of the basic Bernoulli Embeddings only requires a couple of lines of code:

```py
from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity
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

```py
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate
import networkx as nx
import pandas as pd
import re, itertools

# Aggregate to yearly string objects and their associated years
df = pd.read_csv("dynamic.csv")
texts, years = [], sorted(list(set(df["year"])))
for year in years:
    current = df[df["year"] == year]
    text = " ".join(current["speech"])
    text = re.sub(r"\W+ ", '', text).lower()
    texts.append(text.split())

# Preprocess
texts, vocabulary = preprocess_partitioned(texts, years)
text = list(itertools.chain(*texts))
print(text[:10], text[-10:])

# Create side information graph, dog_2010 ~ dog_2011 etc.
g = nx.Graph()
for year0, year1 in zip(years[:-1], years[1:]):
    for wd in set([wd.split("_")[0] for wd in vocabulary]):
        wd0 = f"{wd}_{year0}"
        wd1 = f"{wd}_{year1}"
        if wd0 in vocabulary and wd1 in vocabulary:
            g.add_edge(wd0, wd1)
print(list(g.edges)[:10])

# Define embedding and perform MAP estimation
dim = 100
e = LaplacianEmbedding(vocabulary, dim, g)
e = map_estimate(e, text, model="cbow", ws=5, epochs=5, evaluate=False)

# Save embedding
e.save("dynamic_embedding.pkl")
```

Additionally, informative priors can be added to trace shifts in word meaning:

```py
import networkx as nx
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity
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