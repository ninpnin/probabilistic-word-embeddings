import networkx as nx
from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import get_eval_file, embedding_similarities, evaluate_word_similarity
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf
import pandas as pd
import argparse

def main(args):
    text = open("wiki.txt").read().lower().split()
    text, vocabulary = preprocess_standard(text)
    print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

    g = nx.Graph()
    df = pd.read_csv("d2v.txt", delimiter=" ", names=["w1", "w2"])
    for _, row in df.iterrows():
        w1, w2 = row["w1"], row["w2"]
        if w1 in vocabulary and w2 in vocabulary:
            g.add_edge(w1, w2 + "_c")
            g.add_edge(w2, w1 + "_c")

    if args.lambda1 == 0.0:
        e = Embedding(vocabulary, args.dim, lambda0=args.lambda0)
    else:
        e = LaplacianEmbedding(vocabulary, args.dim, g, lambda0=args.lambda0, lambda1=args.lambda1)
    
    # Perform MAP estimation
    e = map_estimate(e, text, model=args.likelihood, ws=5, batch_size=args.batch_size, epochs=args.epochs)
    similarity = evaluate_word_similarity(e)
    print(similarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lambda0', type=float, default=1.0)
    parser.add_argument('--lambda1', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--likelihood', type=str, default="cbow")

    args = parser.parse_args()
    main(args)