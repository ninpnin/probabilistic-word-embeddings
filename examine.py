from probabilistic_word_embeddings.embeddings import SavedEmbedding
import pandas as pd

e = SavedEmbedding("embeddings.pkl")
print("Vocab len", len(e.vocabulary))
print("Theta:", e.theta.shape)

words = ["some", "words", "i", "like", "this", "is", "an", "example"]
print(words)
words = list(filter(lambda x: x in e, words))
print(words)

vectors = e[words]
print(vectors)

df = pd.DataFrame(vectors)
df["word"] = words

print(df)