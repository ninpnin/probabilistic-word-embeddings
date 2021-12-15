import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import probabilistic_word_embeddings.distributions as ctd
import functools
Root = tfd.JointDistributionCoroutine.Root
from probabilistic_word_embeddings.embeddings import DynamicNormalEmbedding, DynamicInformativeEmbedding, concat_embeddings
from probabilistic_word_embeddings.models.base_models import x_batch_cbow, _generate_cbow_batch
from probabilistic_word_embeddings.utils import dynamic_si

##################################
# Base models for internal logic #
##################################

# TODO: Refactor dynamic SGNS to use theta instead of rho, alpha
def dynamic_sgns(theta=None, ns=5, ws=3, batch_size=25000):
    return NotImplementedError
''' # Old code for reference
def init_dynamic_sgns(timesteps, dim=25, vocab_size=10, minibatch=7):
    dynamic_sgns = tfd.JointDistributionNamedAutoBatched(dict(
        # rho and alpha: Embedding vectors
        rho=    DynamicNormalEmbedding(vocab_size, timesteps), # Word vectors
        alpha=  DynamicNormalEmbedding(vocab_size, timesteps), #  Context vectors

        i=      tfd.Sample(uniform_categorical.UniformCategorical(vocab_size), minibatch),
        j=      tfd.Sample(uniform_categorical.UniformCategorical(vocab_size), minibatch),
        t=      tfd.Sample(uniform_categorical.UniformCategorical(timesteps), minibatch),

        x = lambda rho,alpha,i,j,t:   tfd.Bernoulli(logits=
                                        tf.reduce_sum(
                                            tf.multiply(
                                                tf.gather_nd(
                                                    rho,
                                                    tf.stack([t,i], axis=1)
                                                ),
                                                tf.gather_nd(
                                                    alpha,
                                                    tf.stack([t,j], axis=1)
                                                )
                                            ),
                                            axis=-1
                                        )
                                    )
    ))

    return dynamic_sgns
'''

def dynamic_cbow(timesteps=None, vocab_size=None, dim=25, theta=None, ns=5, ws=3, batch_size=25000):
    print("MOIDE L ")
    total_minibatch = batch_size * (1 + ns)
    #sample = theta.sample()
    #print("sample", sample)    
    print("Timesteps", timesteps, "vocab", vocab_size, "dim", dim)
    
    tiling = tf.constant([ws * 2, 1])
    
    shape = (timesteps +1, vocab_size, dim)
    model = tfd.JointDistributionNamedAutoBatched(dict(
        theta=          ctd.Noninformative(shape=shape),
        i=              tfd.Sample(ctd.UniformCategorical(vocab_size), total_minibatch),
        j=              tfd.Sample(ctd.UniformCategorical(vocab_size), (2 * ws, total_minibatch)),
        t=              tfd.Sample(ctd.UniformCategorical(timesteps), total_minibatch),

        x = lambda theta,i,j,t:   tfd.Bernoulli(logits=
                                            tf.reduce_sum(
                                                tf.multiply(
                                                    tf.gather_nd( # rhos
                                                        theta,
                                                        tf.stack([t,i], axis=1)
                                                    ),
                                                    tf.reduce_sum(
                                                        tf.gather( # alphas
                                                            theta[-1],
                                                            j,
                                                            axis=0
                                                        ),
                                                        axis=0
                                                    )
                                                ),
                                                axis=-1
                                            )
                                        )

        ))

    # Define batch generation
    x_batch = x_batch_cbow(batch_size, ns)
    def get_batch(data, start_ix, dataset_ix):
        i,j,t = _generate_dynamic_cbow_batch(data, vocab_size, D=dim, ws=ws, ns=ns,
                        batch=batch_size, start_ix=start_ix, dataset_ix=dataset_ix)
        return i,j,t

    def loss(batch, theta):
        print(batch)
        i,j,t = batch[0], batch[1], batch[2]
        print("SHAPES", i.shape, j.shape, t.shape)
        return - model.log_prob(dict(theta=theta, i=i, j=j, t=t, x=x_batch))
    
    model.loss = lambda batch, theta: loss(batch, theta)
    model.get_batch = lambda data, start_ix, dataset_ix: get_batch(data, start_ix, dataset_ix)

    model.init = lambda: theta.init()
    # Copy function attributes to the object
    model.V, model.D, model.ws, model.ns, model.batch, model.T = vocab_size, dim, ws, ns, batch_size, timesteps
    
    model.theta_split = 0, timesteps # where to split theta into rho and alpha
    
    return model

#@tf.function
def _generate_dynamic_cbow_batch(datasets, V, D=100, ws=5, ns=5, batch=150000, start_ix=0, dataset_ix=0):
    timestep = dataset_ix
    i, j = _generate_cbow_batch(datasets, V, D=D, ws=ws, ns=ns, batch=batch, start_ix=start_ix, dataset_ix=dataset_ix)
    j = tf.transpose(j)
    total_minibatch = batch * (1 + ns)
    t = tf.ones(total_minibatch, dtype=tf.int64) * timestep
    return i, j, t

dynamic_model_functions = dict(
    cbow= dynamic_cbow,
    sgns= dynamic_sgns
)