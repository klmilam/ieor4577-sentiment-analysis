# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os
import random

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform import coders

from preprocess import preprocess
from preprocess import features


class ParseCSV(beam.DoFn):
    def __init__(self, header):
        self.header = header

    def process(self, element):
        values = element.split(",")
        row = dict(zip(self.header, values))
        yield row


def load_embedding_from_cloud(filename):
    with tf.io.gfile.GFile(filename) as file:
        return json.load(file)


class ProcessTweet(beam.DoFn):
    def process(self, element, token_indices_file):
        tweet = element["Tweet"]
        embedding = load_embedding_from_cloud(token_indices_file)
        features = preprocess.run_pipeline(
            tweet,
            max_length_tweet=40,
            embedding=embedding,
            max_length_dictionary=10000)
        element["features"] = features
        element["Sentiment"] = int(element["Sentiment"].strip("'").strip('"'))
        yield element


@beam.ptransform_fn
def randomly_split(p, train_size, validation_size, test_size):
    """Randomly splits input pipeline in three sets based on input ratio.
    Args:
        p: PCollection, input pipeline.
        train_size: float, ratio of data going to train set.
        validation_size: float, ratio of data going to validation set.
        test_size: float, ratio of data going to test set.
    Returns:
        Tuple of PCollection.
    Raises:
        ValueError: Train validation and test sizes don`t add up to 1.0.
    """
    if train_size + validation_size + test_size != 1.0:
        raise ValueError(
            'Train validation and test sizes don`t add up to 1.0.')

    class _SplitData(beam.DoFn):
        def process(self, element):
            r = random.random()
            if r < test_size:
                yield beam.pvalue.TaggedOutput("Test", element)
            elif r < 1 - train_size:
                yield beam.pvalue.TaggedOutput("Val", element)
            else:
                yield element

    # Group by Tweet to avoid data leakage
    grouped_data = (
        p
        | "KeyByTweet" >> beam.Map(lambda row: (row["Tweet"], row))
        | "GroupByTweet" >> beam.GroupByKey()
    )

    split_data = (
        grouped_data
        | "SplitData" >> beam.ParDo(_SplitData()).with_outputs(
            "Test",
            "Val",
            main='Train')
    )

    return split_data["Train"], split_data["Val"], split_data["Test"]


@beam.ptransform_fn
def WriteOutput(p, prefix, output_dir):
    path = os.path.join(output_dir, prefix)
    schema = dataset_schema.from_feature_spec(features.RAW_FEATURE_SPEC)
    coder = coders.ExampleProtoCoder(schema)
    _ = (
        p
        | "Unkey" >> beam.FlatMap(lambda x: x[1])
        | "WriteTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
            path,
            coder=coder,
            file_name_suffix=".tfrecord")
        )


def build_pipeline(pipeline, header, args):
    # embedding = load_embedding_from_cloud(args.token_indices_file)
    input_data = (
        pipeline
        | "ReadCSV" >> beam.io.ReadFromText(
            args.input_file, skip_header_lines=1)
        | "ParseCSV" >> beam.ParDo(ParseCSV(header))
        | "ProcessTweet" >> beam.ParDo(ProcessTweet(), args.token_indices_file)
    )
    raw_train, raw_eval, raw_test = (
        input_data
        | "RandomlySplitData" >> randomly_split(
            train_size=.7,
            validation_size=.15,
            test_size=.15))

    for dataset_type, dataset in [('Train', raw_train),
                                  ('Eval', raw_eval),
                                  ('Predict', raw_test)]:
        write_label = 'Write{}TFRecord'.format(dataset_type)
        dataset | write_label >> WriteOutput(
            dataset_type, args.output_dir)
