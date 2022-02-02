"""
A collection of algorithms for the estimation and evaluation of different kinds of Bernoulli Word Embeddings.

We strive to keep all functionality clear, concise and easy to use.
For instance, MAP estimation of the basic Bernoulli Embeddings only requires a couple of lines of code:

```py
from probabilistic_word_embeddings.eval import wordsim
from probabilistic_word_embeddings import preprocess
from probabilistic_word_embeddings.models import init_cbow
from probabilistic_word_embeddings.estimation import map_estimate

# Reads all files from datapath.
datapath = "raw/static/"
data, vocab = preprocess(datapath)
vocab_size = max(vocab.values()) + 1

cbow = init_cbow(lambda0=30.0, vocab_size=vocab_size, dim=100, batch_size=2000)
theta_history = map_estimate(cbow, data)

# Get value of theta at last training epoch
theta_last = theta_history[-1][0]
rhos = theta_last[0][:V]

results = wordsim(rhos, d)
print(results)
```

The estimation of _dynamic_ Bernoulli embeddings is relatively straightforward, too.
Additionally, informative priors can be added to trace shifts in word meaning:

```py
from probabilistic_word_embeddings import preprocess
from probabilistic_word_embeddings.models import init_dynamic_informative
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.utils import transitive_dict, inverse_dict

# Reads all files from datapath. They should be in alphabetical order.
datapath = "raw/dynamic/"
data, vocab = preprocess(datapath, data_type="dynamic")
timesteps = len(data)

inv_vocab = inverse_dict(vocab)
vocab_size = max(vocab.values()) + 1

# Add your SI in this format:
si = {"good": 1.0, "bad": -1.0}

# Convert str->int in the side information dictionary
si = transitive_dict(inv_vocab, si)
# The keys are now the indices of the words {168: 1.0, 378: -1.0}

cbow = init_dynamic_informative(lambda0=30.0, vocab_size=vocab_size,
	timesteps=timesteps, dim=100, si=si, batch_size=2000)
theta_history = map_estimate(cbow, data, epochs=5)

# Get value of theta at last training epoch
theta_last = theta_history[-1][0]
print("Shape of the embedding:", theta_last.shape)

# Let's get some words
great = 	vocab["great"]
terrible = 	vocab["terrible"]
great0 =  	theta_last[0, great, -1] 	# timestep=0, dim=-1
terrible1 =	theta_last[1, terrible, -1] # timestep=1, dim=-1

print("Great",    "last dim", "t0", great0)
print("Terrible", "last dim", "t1", terrible1)
```
"""