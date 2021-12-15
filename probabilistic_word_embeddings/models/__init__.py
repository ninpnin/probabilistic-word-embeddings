"""
Likelihood functions for different model configurations.
These range from static SGNS to dynamic CBOW models. 
"""
from probabilistic_word_embeddings.models.model import CBOW, SGNS, DynamicSGNS, DynamicCBOW
from probabilistic_word_embeddings.models.laplacian_models import init_laplacian_model_theta