"""Multivariate Normal distribution defined by a sparse precision matrix"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_diag
import tensorflow_probability as tfp
#import tfp.distributions
tfd = tfp.distributions

import math

class Noninformative(mvn_diag.MultivariateNormalDiag):

  def __init__(self,
               loc=None,
               validate_args=False,
               shape=None,
               allow_nan_stats=True,
               name='MultivariateNormalPrecision'):

    parameters = dict(locals())
    if loc is None and shape is None:
      raise ValueError('Must specify one of `loc`, `shape`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc], tf.float32)
      if loc == None:
        loc = tf.zeros(shape)
      loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
      self._loc = loc
      scale_diag = tf.ones(shape, dtype=dtype)
      
      super(Noninformative, self).__init__(
          loc=loc,
          scale_diag=scale_diag,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    
  def _log_prob(self, x):
    if x.shape.rank == 1:
      return tf.constant(0.0)
    elif x.shape.rank >= 2:
      return tf.zeros(x.shape[1:])
