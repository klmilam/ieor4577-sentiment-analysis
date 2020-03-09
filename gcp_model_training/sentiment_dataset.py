"""
Module holing dataset methods

Author pharnoux

"""

import os
import json
import math
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset

def train_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "train")

def validation_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "validation")

def eval_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "eval")

def serving_input_fn(_, config):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def _input_fn(directory, config, mode):

    print("Fetching {} data...".format(mode))

    def _parse_function(example):
        features_description = {
            "Sentiment": tf.io.FixedLenFeature([], tf.float32),
            "features": tf.io.FixedLenFeature([40], tf.float32)
        }

        parsed_example = tf.parse_single_example(
            serialized=example,
            features=features_description)
        features = parsed_example["features"]
        label = parsed_example["Sentiment"] / 4
        return features, label


    filenames = tf.matching_files(os.path.join(directory, "*"))
    dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=1000)
    dataset = dataset.map(_parse_function, num_parallel_calls=100)
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.repeat()
    if mode == "train":
        dataset = dataset.shuffle(1000, seed=12345).repeat(config["num_epoch"])

    iterator = dataset.make_one_shot_iterator()
    dataset_features, dataset_labels = iterator.get_next()

    return [{config["input_tensor_name"]: dataset_features}, dataset_labels]
