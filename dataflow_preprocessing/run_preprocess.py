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
import sys

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from dataflow_preprocessing import create_tfrecords


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
        default="gs://internal-klm/job/" + timestamp,
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
    # parser.add_argument(
    #     "--output_directory",
    #     help="Local or cloud directory to write output TFRecords.",
    #     type=str,
    #     required=True)
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def get_pipeline_options(args):
    """Returns pipeline options."""
    options = {"project": args.project_id}
    if args.cloud:
        if not args.job_dir:
            raise ValueError("Job directory must be specified for Dataflow.")
        options.update({
            "job_name": args.job_name,
            "setup_file": args.setup_file,
            "staging_location": os.path.join(args.job_dir, "staging"),
            "temp_location": os.path.join(args.job_dir, "tmp"),
        })
    return PipelineOptions(flags=[], **options)


def main():
    """Configures and runs an Apache Beam pipeline."""
    args = parse_arguments(sys.argv)
    header = pd.read_csv(args.input_file, index_col=0, nrows=0).columns.tolist()
    logging.getLogger().setLevel(logging.INFO)
    options = get_pipeline_options(args)
    runner = "DataflowRunner" if args.cloud else "DirectRunner"
    with beam.Pipeline(runner, options=options) as pipeline:
        create_tfrecords.build_pipeline(pipeline, header, args)


if __name__ == "__main__":
    main()
