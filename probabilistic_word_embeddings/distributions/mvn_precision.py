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

class MultivariateNormalPrecision(mvn_diag.MultivariateNormalDiag):

  def __init__(self,
               loc=None,
               precision=None,
               normalize_log_prob=False,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateNormalPrecision'):

    parameters = dict(locals())
    if loc is None and precision is None:
      raise ValueError('Must specify one or both of `loc`, `scale_tril`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, precision], tf.float32)
      if loc == None:
        loc = tf.zeros(precision.shape[0])
      loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
      #precision = tensor_util.convert_nonref_to_tensor(precision, name='precision', dtype=dtype)
      self._loc = loc
      self._precision = precision
      self._sample_dist = None
      self._normalize_log_prob = normalize_log_prob
      self._normalization_factor = None
      
      scale_diag = tf.ones(precision.shape[0], dtype=dtype)
      
      super(MultivariateNormalPrecision, self).__init__(
          loc=loc,
          scale_diag=scale_diag,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
  
  def calculate_covariance(self):
    precision_dense = tf.sparse.to_dense(tf.sparse.reorder(self._precision))
    return tf.linalg.inv(precision_dense)

  def set_normalization_factor(self):
    if self._normalization_factor == None:
      precision_dense = tf.sparse.to_dense(tf.sparse.reorder(self._precision))

      part1 = -0.5 * tf.math.log(2. * math.pi) * self._event_shape
      part2 = 0.5 * tf.math.log(tf.linalg.det(precision_dense))

      self._normalization_factor = part1 + part2
    
  def _log_prob(self, x):
    xLx = None
    if x.shape.rank == 1:
      x = tf.reshape(x - self._loc, (len(x), 1))
      Lx = tf.sparse.sparse_dense_matmul(self._precision, x)
      xLx =  tf.reduce_sum(tf.matmul(tf.transpose(x), Lx))
    elif x.shape.rank == 2:
      #X = x - self._loc
      X = tf.transpose(x)
      #print("X shape", X.shape)
      Lx = tf.sparse.sparse_dense_matmul(self._precision, X)
      #print("Lx", Lx.shape)
      xLx =  tf.multiply(X, Lx)
      xLx = tf.reduce_sum(xLx, axis=0)
      #xLx = tf.linalg.diag_part(xLx)

    if self._normalize_log_prob:
      self.set_normalization_factor()
      return - 0.5 * xLx + self._normalization_factor
    else:
      return - 0.5 * xLx

  def set_sample_dist(self):
    if self._sample_dist == None:
      '''
      covariance_matrix = self.calculate_covariance()
      print("Covariance:\n", covariance_matrix)
      transformation = tf.linalg.cholesky(covariance_matrix)
      self._sample_dist = tfd.MultivariateNormalTriL(loc=self._loc, scale_tril=transformation)
      '''
      self._sample_dist = tfd.MultivariateNormalDiag(loc=self._loc)
      
  
  """
  def _sample_n(self, n, seed=None):
    self.set_sample_dist()
    return self._sample_dist._sample_n(n, seed=seed)

  def _sample(self, seed=None):
    self.set_sample_dist()
    return self._sample_dist.sample(seed=seed)

  def _sample_n(self, n, seed=None):
    self.set_sample_dist()
    return self._sample_dist.sample(n, seed=seed)
    '''
"""

def main():
    pass
  
if __name__ == '__main__':
  main()
