import networkx as nx
import polars as pl
import re
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

# Aggregate to yearly string objects and their associated years
yearly = pl.read_csv("yearly.csv")
yearly = yearly.with_column(pl.col("date") // 10000) # Flatten date to just the year
texts, years = [], sorted(list(set(yearly["date"])))
for year in years:
	current = yearly.filter(pl.col("date") == year)
	text = " ".join(current["speech"]).lower()
	text = re.sub(r'[^ \w+]', '', text)
	text = text.split()
	texts.append(text)

# Preprocess
texts, vocabulary = preprocess_partitioned(texts, years)
text = []
for t in texts:
	text = text + t
print(text[:10], text[-10:])

# Create side information graph
g = nx.Graph()
for year0, year1 in zip(years[:-1], years[1:]):
	for wd in set([wd.split("_")[0] for wd in vocabulary]):
		wd0 = f"{wd}_{year0}"
		wd1 = f"{wd}_{year1}"
		if wd0 in vocabulary and wd1 in vocabulary:
			g.add_edge(wd0, wd1)


# Define embedding and perform MAP estimation
dim = 100
e = LaplacianEmbedding(vocabulary, dim, g)
e = map_estimate(e, text, model="cbow", ws=5, epochs=1, evaluate=False)

# Save embedding
e.save("dynamic_embedding.pkl")
