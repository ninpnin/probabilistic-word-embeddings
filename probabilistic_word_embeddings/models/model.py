import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from probabilistic_word_embeddings.embeddings import NormalEmbedding, concat_embeddings
from probabilistic_word_embeddings.models.base_models import sgns_model, cbow_model
from probabilistic_word_embeddings.models.dynamic_models import dynamic_cbow
import collections.abc

class Model:
    """
    Probabilistic word embedding model, which consists of
    a prior part and a likelihood part. 
    """
    def __init__(self, prior, likelihood, ws=2, ns=5, batch_size=25000):
        self.prior = prior
        """The prior distribution of the model's parameters"""
        self.likelihood = likelihood
        """How the model relates to data"""
        self.data_size = None
        """Token count of the data that the model is trained on."""
        self.batch_size = batch_size

    def get_batch(self, data, start_ix, dataset_ix):
        """Generate a batch from a data set"""
        return self.likelihood.get_batch(data, start_ix, dataset_ix)

    def init(self):
        """Initialize the latent parameters of the model"""
        if hasattr(self.prior, "init"):
            x = self.prior.init()
            return x
        else:
            xs = self.prior.sample()
            return xs

    def prior_weight(self):
        assert self.data_size is not None, "Define model.data_size to get loss"
        return self.batch_size / self.data_size

    def loss(self, batch, x):
        """Returns the posterior probability of a batch of the data"""
        prior_weight = self.prior_weight()

        prior_prob = self.prior.log_prob(x)
        prior_loss = - prior_prob * prior_weight
        if isinstance(x, dict):
            likelihood = self.likelihood.loss(batch, x["theta"])
        elif isinstance(x, collections.abc.Sequence):
            likelihood = self.likelihood.loss(batch, x[-1])
        else:
            likelihood = self.likelihood.loss(batch, x)
        return prior_loss + likelihood

class SGNS(Model):
    """
    A model with the SGNS likelihood function.

    Args:
        prior: the prior probability distribution of the latent parameters
    """
    def __init__(self, prior, ws=2, ns=5, batch_size=25000):
        vocab_size, dim = prior.event_shape
        vocab_size = vocab_size // 2
        likelihood = sgns_model(vocab_size=vocab_size, dim=dim, ws=ws, ns=ns, batch_size=batch_size)
        super(SGNS, self).__init__(prior, likelihood, ws=ws, ns=ns, batch_size=batch_size)

class CBOW(Model):
    """
    A model with the CBOW likelihood function.

    Args:
        prior: the prior probability distribution of the latent parameters
    """
    def __init__(self, prior, ws=2, ns=5, batch_size=25000):
        vocab_size, dim = prior.event_shape
        vocab_size = vocab_size // 2

        print("V D", vocab_size, dim)
        likelihood = cbow_model(vocab_size=vocab_size, dim=dim, ws=ws, ns=ns, batch_size=batch_size)
        super(CBOW, self).__init__(prior, likelihood, ws=ws, ns=ns, batch_size=batch_size)

class DynamicSGNS(Model):
    """
    A dynamic model with the SGNS likelihood function.

    Args:
        prior: the prior probability distribution of the latent parameters
    """
    def __init__(self, prior):
        pass

class DynamicCBOW(Model):
    """
    A dynamic model with the CBOW likelihood function.

    Args:
        prior: the prior probability distribution of the latent parameters
    """
    def __init__(self, prior, ws=2, ns=5, batch_size=25000):
        timesteps = prior.event_shape[0] - 1 # -1 since the last element is alphas
        vocab_size = prior.event_shape[1]
        dim = prior.event_shape[2]

        print("T", timesteps, "V", vocab_size, "D", dim)
        likelihood = dynamic_cbow(timesteps=timesteps, vocab_size=vocab_size, dim=dim, ws=ws, ns=ns, batch_size=batch_size)
        super(DynamicCBOW, self).__init__(prior, likelihood, ws=ws, ns=ns, batch_size=batch_size)
