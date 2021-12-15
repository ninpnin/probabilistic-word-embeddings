import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from probabilistic_word_embeddings.embeddings import NormalEmbedding, LaplacianEmbedding, concat_embeddings
from probabilistic_word_embeddings.models.base_models import model_functions

"""
This file contains the definitions for the Laplacian SGNS and CBOW models
"""

def init_laplacian_model(model_type="cbow", vocab_size=1, dim=25, laplacian=None, lambda0=1.0, lambda1=1.0, ws=2, ns=5, batch_size=25000):
    """
    Returns a probabilistic CBOW model with the word vectors \\( \\rho_i \\) , \\( \\alpha_i \\) distributed with a Laplacian prior.
    Context vectors follow an uncorrelated normal prior.

    $$ \\rho_i \\sim \\mathcal{N}(\\mathbf{0}, \\lambda_0 \\mathbf{I} + \\lambda_1 \\mathbf{L}) $$

    $$ \\alpha_i \\sim \\mathcal{N}(\\mathbf{0}, \\lambda_0 \\mathbf{I} ) $$
    
    The likelihood is then

    $$ x_{ij} \\sim \\text{Ber}( \\sigma (\\sum_{k \\in C_j} \\alpha_k^T \\rho_i) )$$

    for the CBOW and

    $$ x_{ij} \\sim \\text{Ber}( \\sigma (\\alpha_j^T \\rho_i) )$$

    for SGNS.

    Args:
        model_type: Type of the data likelihood. 'cbow' or 'sgns'.
        vocab_size: size of the vocabulary
        dim: dimensionality of the embedding vectors
        laplacian: The Laplacian matrix that is used to create the precision matrix. Requires the tf.SparseMatrix format.

            If no Laplacian is provided, it falls back to a zero matrix.
        lambda0: The diagonal weighting of the precision matrix. Corresponds to standard deviation if the Laplacian is a zero matrix.
        lambda1: The off-diagonal weighting of the precision matrix.
        ws: Size of the context window
        ns: Number of negative samples
        batch_size: Number of data points per training iteration
    """

    # Word and context vectors
    rho =   LaplacianEmbedding(vocab_size, laplacian=laplacian, lambda0=lambda0, lambda1=lambda1, dim=dim)
    alpha = LaplacianEmbedding(vocab_size, laplacian=None, lambda0=lambda0, dim=dim)
    theta = concat_embeddings([rho, alpha], axis=0)

    model_type = model_functions[model_type]
    return model_type(theta=theta, ws=ws, ns=ns, batch_size=batch_size)

def init_laplacian_model_theta(model_type="cbow", vocab_size=1, dim=25, laplacian=None, lambda0=1.0, lambda1=1.0, ws=2, ns=5, batch_size=25000):
    """
    Returns a probabilistic CBOW model with embedding vectors \\( \\rho_i \\) , \\( \\alpha_i \\) distributed with a Laplacian prior

    $$ \\theta_i \\sim \\mathcal{N}(\\mathbf{0}, \\lambda_0 \\mathbf{I} + \\lambda_1 \\mathbf{L}) $$
    
    The likelihood is then

    $$ x_{ij} \\sim \\text{Ber}( \\sigma (\\sum_{k \\in C_j} \\alpha_k^T \\rho_i) )$$

    for the CBOW and

    $$ x_{ij} \\sim \\text{Ber}( \\sigma (\\alpha_j^T \\rho_i) )$$

    for SGNS.

    Args:
        model_type: Type of the data likelihood. 'cbow' or 'sgns'.
        vocab_size: size of the vocabulary
        dim: dimensionality of the embedding vectors
        laplacian: The Laplacian matrix that is used to create the precision matrix. Requires the tf.SparseMatrix format.

            If no Laplacian is provided, it falls back to a zero matrix.
        lambda0: The diagonal weighting of the precision matrix. Corresponds to standard deviation if the Laplacian is a zero matrix.
        lambda1: The off-diagonal weighting of the precision matrix.
        ws: Size of the context window
        ns: Number of negative samples
        batch_size: Number of data points per training iteration
    """
    # Word and context vectors
    total_minibatch = batch_size * (1 + ns)
    theta = LaplacianEmbedding(vocab_size * 2, laplacian=laplacian, lambda0=lambda0, lambda1=lambda1, dim=dim)

    model_type = model_functions[model_type]
    return model_type(theta=theta, ws=ws, ns=ns, batch_size=batch_size)
    
if __name__ == "__main__":
    # Example values for vocab size and dim
    V, D = 3, 5
    
    # You can mix and match model types and priors for embeddings:
    # Here we have a Laplacian prior that spans across word and
    # context vectors used in a CBOW model
    model_type = "cbow"
    model = init_laplacian_model_theta(model_type=model_type, vocab_size=V, dim=D)
    print(model.sample())
    
    # And here we have a Laplacian prior on the word vectors and
    # a spherical Gaussian for the context vectors used in a SGNS model 
    model_type = "sgns"
    model = init_laplacian_model(model_type=model_type, vocab_size=V, dim=D)
    print(model.sample())
