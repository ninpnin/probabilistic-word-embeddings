"""Point estimation of the models"""
import tensorflow as tf
import tensorflow_probability as tfp
# Load model from models.py
from probabilistic_word_embeddings.utils import shuffled_indices
import glob
import progressbar

@tf.function
def _optimize_step(model, batch, optimizer, theta):
    losses = tfp.math.minimize(lambda: model.loss(batch, theta), optimizer=optimizer, num_steps=1)
    return losses

    
def _variable_values(variables):
    if isinstance(variables, dict):
        return {label: variable.numpy() for label, variable in variables.items()}
    elif isinstance(variables, list):
        return [variable.numpy() for variable in variables]
    else:
        return variables.numpy()

def map_estimate(model, data, init=None, epochs=5, profile=False, history=False):
    """
    This function performs MAP estimation.

    Args:
        model: Word embedding model.

            The model is expected to have 'loss' and 'init' functions.
        data: Data as a list of of NumPy arrays. The arrays should consist of word indices.
        init: Variable parameters in the model as a list of tf.Variable's. If None, will be initialized from the models init function.
        epochs: The number of passes over the data.
    """
    if profile:
        tf.profiler.experimental.start("logs")

    # Initialize variables for the embedding matrices
    if init == None:
        init = model.init()
    if isinstance(init, dict):
        theta = init
    else:
        theta = tf.Variable(init, dtype=tf.float32)
    adam_optimizer = tf.optimizers.Adam()
    theta_history = []

    for epoch in range(epochs):
        print("Epoch", epoch, "of", epochs)
        Ns = [len(dataset) for dataset in data]
        start_indices = shuffled_indices(Ns, model.batch_size)
        for batch_no, start_ix in progressbar.progressbar(list(enumerate(start_indices))):
            start_ix, dataset_ix = start_ix[0], int(start_ix[1].numpy())

            batch = model.get_batch(data, start_ix, dataset_ix)
            _optimize_step(model, batch, adam_optimizer, theta)

            if batch_no == 20 and epoch == 0 and profile:
                tf.profiler.experimental.stop()

        # Save after each epoch
        if history:
            theta_history.append(_variable_values(theta))
    
    if history:
        return theta_history
    else:
        return _variable_values(theta)
