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

import argparse
from datetime import datetime
import logging
import os
import pandas as pd
import posixpath
import sys

import apache_beam as beam
from apache_beam.options import pipeline_options

from preprocess import create_tfrecords


def parse_arguments(argv):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args(args=argv[1:])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument(
        "--cloud",
        help="""Run preprocessing on the cloud. Default False.""",
        action="store_true",
        default=False)
    parser.add_argument(
        "--job_name",
        help="Dataflow job name.",
        type=str,
        default="{}-{}".format("tfrecords-sentiment-analysis-", timestamp))
    parser.add_argument(
        "--job_dir",
        type=str,
        default="gs://internal-klm/sentiment-analysis/job/" + timestamp,
        help="""GCS bucket to stage code and write temporary outputs for cloud
        runs.""")
    parser.add_argument(
        "--project_id",
        help="GCP project id",
        type=str,
        default="internal-klm")
    parser.add_argument(
        "--input_file",
        help="Local or cloud file containing input data.",
        type=str,
        default="gs://internal-klm/sentiment-analysis/training.full.csv")
    parser.add_argument(
        "--token_indices_file",
        help="Local or cloud file containing token indices.",
        type=str,
        default="gs://internal-klm/sentiment-analysis/token_indices.json")
    parser.add_argument(
        "--output_dir",
        help="Local or cloud directory to write output TFRecords.",
        type=str,
        default="gs://internal-klm/sentiment-analysis/output/{}".format(timestamp))
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def get_pipeline_options(args):
    """Returns pipeline options."""
    if not args.cloud:
        options = {"project": args.project_id}
        return pipeline_options.PipelineOptions(flags=[], **options)

    options = pipeline_options.PipelineOptions()
    worker_options = options.view_as(pipeline_options.WorkerOptions)
    worker_options.machine_type = "n1-highmem-8"
    setup_options = options.view_as(pipeline_options.SetupOptions)
    setup_options.setup_file = posixpath.abspath(
        posixpath.join(posixpath.dirname(__file__),
        "setup.py"))
    # setup_options.save_main_session = True
    print(setup_options.setup_file)
    if not args.job_dir:
        raise ValueError("Job directory must be specified for Dataflow.")
    google_cloud_options = options.view_as(pipeline_options.GoogleCloudOptions)
    google_cloud_options.project = args.project_id
    google_cloud_options.job_name = args.job_name
    google_cloud_options.staging_location = os.path.join(args.job_dir, "staging")
    google_cloud_options.temp_location = os.path.join(args.job_dir, "tmp")
    google_cloud_options.region = "us-central1"
    return options


def main():
    """Configures and runs an Apache Beam pipeline."""
    args = parse_arguments(sys.argv)
    header = ["Sentiment", "TwitterID", "Date", "QueryType", "UserID", "Tweet"]
    logging.getLogger().setLevel(logging.INFO)
    options = get_pipeline_options(args)
    runner = "DataflowRunner" if args.cloud else "DirectRunner"
    with beam.Pipeline(runner, options=options) as pipeline:
        create_tfrecords.build_pipeline(pipeline, header, args)


if __name__ == "__main__":
    main()
