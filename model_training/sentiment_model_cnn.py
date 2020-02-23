"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    cnn_model = None

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))
