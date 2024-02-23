import unittest

from probabilistic_word_embeddings.embeddings import Embedding, LaplacianEmbedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import embedding_similarities, evaluate_word_similarity, evaluate_analogy, nearest_neighbors
from probabilistic_word_embeddings.utils import align, transfer_embeddings
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import random
import networkx as nx

class EmbeddingTest(unittest.TestCase):

    def test_embedding_saving_1(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        saved_model_path = "test.pkl"
        e.save(saved_model_path)
        e2 = Embedding(saved_model_path=saved_model_path)

        theta1 = e.theta.numpy()
        theta2 = e2.theta.numpy()
        self.assertAlmostEqual(np.max(np.abs(theta1-theta2)), 0.0)
        self.assertEqual(e.lambda0, e2.lambda0)

    def test_embedding_saving_2(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        graph = nx.Graph()
        graph.add_edge("this", "that")
        e = LaplacianEmbedding(vocabulary=vocabulary, dimensionality=dim, graph=graph)
        saved_model_path = "test.pkl"
        e.save(saved_model_path)
        e2 = LaplacianEmbedding(saved_model_path=saved_model_path)

        theta1 = e.theta.numpy()
        theta2 = e2.theta.numpy()
        self.assertAlmostEqual(np.max(np.abs(theta1-theta2)), 0.0)
        self.assertEqual(e.lambda0, e2.lambda0)
        self.assertEqual(e.lambda1, e2.lambda1)

    def test_embedding_setting(self):
        """
        Test MAP estimation with example dataset
        """
        with open("tests/data/0.txt") as f:
            text = f.read().lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        words = list(vocabulary)[:10]
        new_embs = np.random.rand(10, dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        words = list(vocabulary)[-1]
        new_embs = np.random.rand(dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        words = tf.constant(list(vocabulary))
        new_embs = np.random.rand(len(vocabulary), dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        words = tf.constant(list(vocabulary)) + "_c"
        new_embs = np.random.rand(len(vocabulary), dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)


    def test_dynamic_embedding_setting(self):
        paths = sorted(list(Path("tests/data/").glob("*.txt")))
        names, texts = [], []
        for p in paths:
            names.append(p.stem)
            with p.open() as f:
                t = f.read().lower().replace("_", "").replace(".", "").replace(",", "")
                t = t.split()
                texts.append(t)

        texts, vocabulary = preprocess_partitioned(texts, names)
        text = []
        for t in texts:
            text = text + t

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25
        
        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        # Word vectors
        no_words = 10
        words = [random.choice(list(e.vocabulary)) for _ in range(no_words)]
        print(words)
        new_embs = np.random.rand(no_words, dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        new_embs = np.random.rand(no_words, dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        words = tf.constant(list(vocabulary))
        new_embs = np.random.rand(len(vocabulary), dim)
        e[words] = new_embs

        old_embs = new_embs
        new_embs = e[words]

        self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)

        vocabulary = set([wd.split("_")[0] for wd in vocabulary])
        for label in [0, 1]:
            words = [wd + f"_{label}_c" for wd in vocabulary]
            words = [wd for wd in words if wd in e]

            new_embs = np.random.rand(len(words), dim)
            e[words] = new_embs

            old_embs = new_embs
            new_embs = e[words]

            self.assertAlmostEqual(np.max(np.abs(old_embs - new_embs)), 0.0)


    def test_evaluation(self):
        """
        Test word analogy evaluation
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)

        vocab_size = len(vocabulary)
        batch_size = 250
        dim = 25

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)
        words = list(vocabulary)[:12]

        d = {"w1": words[:3], "w2": words[3:6], "w3": words[6:9], "w4": words[9:]}
        df = pd.DataFrame(d)
        d2 = pd.DataFrame([[words[9], words[9], words[9], words[9]]], columns=["w1", "w2", "w3", "w4"])
        df = pd.concat([df, d2])

        df = evaluate_analogy(e, df)
        print(df)

    def test_alignment(self):
        """
        Test embedding alignment
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        vocabulary = set(list(set(text))[:300])
        vocab_size = len(vocabulary)
        testwords = list(vocabulary)[:100]
        dim = 25

        e1 = Embedding(vocabulary=vocabulary, dimensionality=dim)
        e2 = Embedding(vocabulary=vocabulary, dimensionality=dim)

        reversed_testwords = list(reversed(testwords))
        dots = tf.reduce_sum(tf.multiply(e2[testwords], e2[reversed_testwords]), axis=1)

        mean_diff = tf.reduce_sum(tf.multiply(e1.theta - e2.theta, e1.theta - e2.theta))        
        self.assertGreater(mean_diff, 0.0)

        words = e1
        e2 = align(e1, e2, list(vocabulary))

        mean_diff_prime = tf.reduce_sum(tf.multiply(e1.theta - e2.theta, e1.theta - e2.theta))
        self.assertGreater(mean_diff, mean_diff_prime)

        dots_prime = tf.reduce_sum(tf.multiply(e2[testwords], e2[reversed_testwords]), axis=1)

        # Check that the dot products remain unchanged
        max_diff = tf.reduce_max(tf.abs(dots - dots_prime)).numpy()
        self.assertAlmostEqual(max_diff, 0.0)

    def test_nearest_neighbors(self):
        """
        Test embedding alignment
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)
        dim = 25

        target = "friend"
        nearest = ["face", "distress", "sleep"]

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        for _ in range(5):
            e_target = np.random.randn(25)
            EPSILON = 0.01
            DEVIATION = np.random.randn(25)
            e_nearest = [e_target + EPSILON * DEVIATION * (i + 1) for i, _ in enumerate(nearest)]

            e[target] = e_target
            for wd, e_i in zip(nearest, e_nearest):
                e[wd] = e_i

            nearest_hat = nearest_neighbors(e, [target], K=len(nearest))
            nearest_hat = list(nearest_hat.loc[0])[1:]
            #print(nearest_hat)
            for k, wordpair in enumerate(zip(nearest, nearest_hat)):
                k = k + 1
                wd, wd_hat = wordpair
                self.assertEqual(wd, wd_hat)

    def test_nearest_neighbors_arguments(self):
        """
        Test that different values of K give same results
        """
        with open("tests/data/0.txt") as f:
            text = f.read().replace(".", "").lower().split()
        text, vocabulary = preprocess_standard(text)
        dim = 25

        target = "friend"
        nearest = ["face", "distress", "sleep"]

        e = Embedding(vocabulary=vocabulary, dimensionality=dim)

        nearest25 = nearest_neighbors(e, [target], K=25)
        nearest3 = nearest_neighbors(e, [target], K=3)
        nearest2 = nearest_neighbors(e, [target], K=2)
        nearest1 = nearest_neighbors(e, [target], K=1)

        p2_25 = list(nearest25["@2"])[0]
        p2_3 = list(nearest3["@2"])[0]
        p2_2 = list(nearest3["@2"])[0]

        self.assertEqual(p2_3, p2_25)
        self.assertEqual(p2_3, p2_2)

        p1_3 = list(nearest3["@1"])[0]
        p1_2 = list(nearest3["@1"])[0]
        p1_1 = list(nearest3["@1"])[0]

        self.assertEqual(p1_3, p1_2)
        self.assertEqual(p1_3, p1_1)

    def test_transfer_embeddings(self):
        vocabulary1 = {"moi", "mitä", "kuuluu"}
        vocabulary2 = {"moi", "mitä", "joo"}
        dim = 10
        e1 = Embedding(vocabulary=vocabulary1, dimensionality=dim)
        e2 = Embedding(vocabulary=vocabulary2, dimensionality=dim)

        vocab_intersection = list(vocabulary1.intersection(vocabulary2))
        diff = tf.reduce_sum(tf.math.abs(e1[vocab_intersection] - e2[vocab_intersection])).numpy()
        self.assertNotEqual(0.0, diff)

        e2 = transfer_embeddings(e1, e2)

        diff = tf.reduce_sum(tf.math.abs(e1[vocab_intersection] - e2[vocab_intersection])).numpy()
        self.assertAlmostEqual(0.0, diff)

        vocabulary1 = [f"{wd}_time1" for wd in list(vocabulary1)] + [f"{wd}_time2" for wd in list(vocabulary1)]
        vocabulary1 = set(vocabulary1)
        e1 = Embedding(vocabulary=vocabulary1, dimensionality=dim)
        e2 = Embedding(vocabulary=vocabulary2, dimensionality=dim)

        for wd in list(vocabulary1):
            wd_normalized = wd.split("_")[0]
            if wd_normalized in e2:
                diff = tf.reduce_sum(tf.math.abs(e1[wd] - e2[wd_normalized])).numpy()
                self.assertNotEqual(0.0, diff)

                diff = tf.reduce_sum(tf.math.abs(e1[wd + "_c"] - e2[wd_normalized + "_c"])).numpy()
                self.assertNotEqual(0.0, diff)

        e1 = transfer_embeddings(e2, e1, ignore_group=True)

        for wd in list(vocabulary1):
            wd_normalized = wd.split("_")[0]
            if wd_normalized in e2:
                diff = tf.reduce_sum(tf.math.abs(e1[wd] - e2[wd_normalized])).numpy()
                self.assertAlmostEqual(0.0, diff)

                diff = tf.reduce_sum(tf.math.abs(e1[wd + "_c"] - e2[wd_normalized + "_c"])).numpy()
                self.assertAlmostEqual(0.0, diff)

    def test_keyerror(self):
        vocabulary1 = {"moi", "mitä", "kuuluu"}
        dim = 10
        e1 = Embedding(vocabulary=vocabulary1, dimensionality=dim)

        try:
            val = e1["äksdee"]
            assert val == None
        except ValueError as e:
            print(f"Caught error {e}")

        val = e1[["moi"]]

        try:
            val = e1[["äksdee", "moi"]]
            assert val == None
        except ValueError as e:
            print(f"Caught error {e}")

    def test_seed(self):
        vocabulary1 = {"moi", "mitä", "kuuluu"}
        vocab_list = list(vocabulary1)
        seed = 123
        dim = 10
        e1 = Embedding(vocabulary=vocabulary1, dimensionality=dim, seed=seed)
        e2 = Embedding(vocabulary=vocabulary1, dimensionality=dim, seed=seed)

        max_diff = tf.reduce_max(tf.abs(e1[vocab_list] - e2[vocab_list])).numpy()
        self.assertAlmostEqual(max_diff, 0.0)

        e1 = LaplacianEmbedding(vocabulary=vocabulary1, dimensionality=dim, seed=seed)
        e2 = LaplacianEmbedding(vocabulary=vocabulary1, dimensionality=dim, seed=seed)

        max_diff = tf.reduce_max(tf.abs(e1[vocab_list] - e2[vocab_list])).numpy()
        self.assertAlmostEqual(max_diff, 0.0)

   

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
