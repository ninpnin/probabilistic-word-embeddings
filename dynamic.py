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