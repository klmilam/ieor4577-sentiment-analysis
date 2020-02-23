"""Create artifacts for preprocessing"""

import argparse
import sys
import json

import tensorflow as tf


def parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Input filename for original data, such as GloVe file.",
        type=str)
    parser.add_argument(
        "--output_file",
        help="Output filename for artifacts.",
        type=str)
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def create_token_to_index_dict(args):
    """Creates token to index dictionary."""
    with tf.io.gfile.GFile(args.input_file, "r") as file:
        input_data = file.read()
    split = input_data.split("\n")
    output = {}
    output["<pad>"] = 0
    output["<unknown>"] = 1
    index = 2
    for element in split:
        word = element.split(" ")[0]
        if word not in ["<unknown>", "<unk>", "<pad>"]:
            output[word] = index
            index += 1
    json_data = json.dumps(output)
    with tf.io.gfile.GFile(args.output_file, "w") as file:
        file.write(json_data)


def main():
    """Configures and runs artifact creation."""
    args = parse_arguments(sys.argv)
    create_token_to_index_dict(args)


if __name__ == "__main__":
    main()
