"""Model definition for CNN sentiment training."""

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant


def keras_model_fn(_, config, args):
    """Creates a CNN model for sentiment modeling."""
    embedding_matrix = np.zeros((
        config["embeddings_dictionary_size"],
        config["embeddings_vector_size"]))

    with tf.io.gfile.GFile(config["embeddings_path"], "r") as file:
        input_data = file.read()
        split = input_data.split("\n")

    for index, _ in enumerate(split):
        data = np.asarray(split[index].split()[1:], dtype='float32')
        if len(data) == config["embeddings_vector_size"]:
            embedding_matrix[index + 2] = data
        else:
            padded = np.zeros((config["embeddings_vector_size"]), 'float32')
            padded[:len(data)] = data
            embedding_matrix[index + 2] = padded

    cnn_model = keras.Sequential()
    cnn_model.add(layers.Embedding(
        input_dim=config["embeddings_dictionary_size"],
        input_length=config["padding_size"],
        embeddings_initializer=Constant(embedding_matrix),
        output_dim=config["embeddings_vector_size"],
        trainable=True))
    cnn_filters = [
        min(1000,
            max(8, int(
            args.first_filter_size * args.cnn_layer_sizes_scale_factor**i)))
        for i in range(args.num_cnn_layers)
    ]
    for i in range(args.num_cnn_layers):
        cnn_model.add(layers.Conv1D(
            filters=cnn_filters[i],
            kernel_size=2,
            strides=1,
            padding="valid",
            activation="relu"))
    cnn_model.add(layers.GlobalMaxPool1D())
    dense_layers = [
        min(1024,
            max(8, int(
            args.first_layer_size * args.dense_layer_sizes_scale_factor**i)))
        for i in range(args.num_dense_layers)
    ]
    for i in range(args.num_dense_layers):
        cnn_model.add(layers.Dense(dense_layers[i], activation="relu"))
    cnn_model.add(layers.Dense(1, activation="sigmoid"))

    cnn_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return cnn_model


def save_model(model, output):
    """Saves models in SaveModel format with signature to support serving."""
    tf.saved_model.save(model, os.path.join(output, "1"))
    print("Model successfully saved at: {}".format(output))
