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

import apache_beam as beam
import tensorflow as tf

from preprocess import preprocess

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
        print(tweet)
        features = preprocess.run_pipeline(
            tweet,
            max_length_tweet=40,
            embedding=embedding,
            max_length_dictionary=10000)
        element["features"] = features
        return [element]


def build_pipeline(pipeline, header, args):
    embedding = load_embedding_from_cloud(args.token_indices_file)
    input_data = (
        pipeline
        | beam.io.ReadFromText(args.input_file, skip_header_lines=1)
        | beam.ParDo(ParseCSV(header))
        | beam.ParDo(ProcessTweet(), args.token_indices_file)
    )
    input_data | beam.Map(print)
