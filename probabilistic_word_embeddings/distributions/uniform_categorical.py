# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Uniform categorical distribution"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
#from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

'''
CLASS DESCRIPTION:

This is a custom implementation of the uniform categorical distribution.
The implementation is based off the categorical distribution implementation
from tfp.distributions, and its main goal is to improve the performance of
log probability calculation, taking advantage of the fact that the distribution
is uniform. Apart from performance bottlenecks, all functions are the same as
in the original.
'''
class UniformCategorical(distribution.Distribution):

    def __init__(
        self,
        categories,
        dist_shape=None,
        dtype=tf.int32,
        validate_args=False,
        allow_nan_stats=True,
        name='UniformCategorical'):

        if dist_shape == None:
            dist_shape = (categories,)

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._categories = categories
            self._dist_shape = dist_shape

        super(UniformCategorical, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)


    @classmethod
    def _params_event_ndims(cls):
        return dict(logits=1, probs=1)

    @property
    def logits(self):
        """Input argument `logits`."""
        return - np.ones(self._categories) * np.log(self._categories)

    @property
    def probs(self):
        """Input argument `probs`."""
        return np.ones(self._categories) / self._categories

    def _batch_shape_tensor(self, x=None):
        return self._dist_shape[:-1]

    def _batch_shape(self):
        return ()

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        #'''
        np_sample = np.random.randint(self._categories, size=n)
        return tf.cast(np_sample, self.dtype)
        '''
        logits = self._logits_parameter_no_checks()
        logits_2d = tf.reshape(logits, [-1, self._num_categories(logits)])
        sample_dtype = tf.int64 if dtype_util.size(self.dtype) > 4 else tf.int32
        # TODO(b/147874898): Remove workaround for seed-sensitive tests.
        if seed is None or isinstance(seed, six.integer_types):
            draws = tf.random.categorical(
                logits_2d, n, dtype=sample_dtype, seed=seed)
        else:
            draws = samplers.categorical(
                logits_2d, n, dtype=sample_dtype, seed=seed)
        draws = tf.cast(draws, self.dtype)
        print(draws)
        return tf.reshape(
            tf.transpose(draws),
            shape=tf.concat([[n], self._batch_shape_tensor(logits)], axis=0))
        #'''

    def _cdf(self, k):
        # TODO(b/135263541): Improve numerical precision of categorical.cdf.
        return (np.range(k.shape) + 1) / self._categories

    def _log_prob(self, k):
        log_p = - np.log(self._categories)
        return tf.ones(k.shape) * log_p

    def _entropy(self):
        return 0.0

    def _mode(self):
        return 0

    def logits_parameter(self, name=None):
        """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
        with self._name_and_control_scope(name or 'logits_parameter'):
            return self._logits_parameter_no_checks()

    def _logits_parameter_no_checks(self):
        return tf.identity(- np.ones(self._categories) * np.log(self._categories))

    def probs_parameter(self, name=None):
        with self._name_and_control_scope(name or 'probs_parameter'):
            return self._probs_parameter_no_checks()

    def _probs_parameter_no_checks(self):
        return tf.identity(1.0 / self._categories)

    def _num_categories(self, x=None):
        """Scalar `int32` tensor: the number of categories."""
        with tf.name_scope('num_categories'):
            return self._categories

    def _default_event_space_bijector(self):
        return

    def _parameter_control_dependencies(self, is_init):
        return maybe_assert_categorical_param_correctness(
            is_init, self.validate_args, self._probs_parameter_no_checks(), self._logits_parameter_no_checks())

    def _sample_control_dependencies(self, x):
        assertions = []
        if not self.validate_args:
            return assertions
        assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
        assertions.append(
            assert_util.assert_less_equal(
                x, tf.cast(self._num_categories(), x.dtype),
                message=('Categorical samples must be between `0` and `n-1` '
                         'where `n` is the number of categories.')))
        return assertions

def maybe_assert_categorical_param_correctness(
    is_init, validate_args, probs, logits):
    """Return assertions for `Categorical`-type distributions."""
    assertions = []

    # In init, we can always build shape and dtype checks because
    # we assume shape doesn't change for Variable backed args.
    if is_init:
        x, name = (probs, 'probs') if logits is None else (logits, 'logits')

        if not dtype_util.is_floating(x.dtype):
            raise TypeError('Argument `{}` must having floating type.'.format(name))

        msg = 'Argument `{}` must have rank at least 1.'.format(name)
        ndims = tensorshape_util.rank(x.shape)
        if ndims is not None:
            if ndims < 1:
                raise ValueError(msg)
        elif validate_args:
            x = tf.convert_to_tensor(x)
            probs = x if logits is None else None  # Retain tensor conversion.
            logits = x if probs is None else None
            assertions.append(assert_util.assert_rank_at_least(x, 1, message=msg))

    if not validate_args:
        assert not assertions  # Should never happen.
        return []

    if logits is not None:
        if is_init != tensor_util.is_ref(logits):
            logits = tf.convert_to_tensor(logits)
            assertions.extend(
                distribution_util.assert_categorical_event_shape(logits))

    if probs is not None:
        if is_init != tensor_util.is_ref(probs):
            probs = tf.convert_to_tensor(probs)
            assertions.extend([
                assert_util.assert_non_negative(probs),
                assert_util.assert_near(
                    tf.reduce_sum(probs, axis=-1),
                    np.array(1, dtype=dtype_util.as_numpy_dtype(probs.dtype)),
                    message='Argument `probs` must sum to 1.')
            ])
            assertions.extend(distribution_util.assert_categorical_event_shape(probs))

    return assertions
    '''

    TODO: calculate KL divergence between this and an arbirtary categorical distibution.

    @kullback_leibler.RegisterKL(Categorical, Categorical)
    def _kl_categorical_categorical(a, b, name=None):
    """Calculate the batched KL divergence KL(a || b) with a and b Categorical.
    Args:
    a: instance of a Categorical distribution object.
    b: instance of a Categorical distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_categorical_categorical'`).
    Returns:
    Batchwise KL(a || b)
    """
    with tf.name_scope(name or 'kl_categorical_categorical'):
    a_logits = a._logits_parameter_no_checks()  # pylint:disable=protected-access
    b_logits = b._logits_parameter_no_checks()  # pylint:disable=protected-access
    return tf.reduce_sum(
        (tf.math.softmax(a_logits) *
         (tf.math.log_softmax(a_logits) - tf.math.log_softmax(b_logits))),
        axis=-1)

    '''