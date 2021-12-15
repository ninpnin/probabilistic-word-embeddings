# Custom TFP Distributions

## Multivariate Normal with Sparse Precision

Defines a multivariate normal distribution with a sparse precision matrix. Sampling is not accurate, and uses a spherical normal distribution instead.

## Uniform Categorical

Defines the uniform categorical distribution. Functionally equivalent to the tfd.Categorical class with the probs ```tf.ones() / N```, and the calculation of log probs is more efficient.
