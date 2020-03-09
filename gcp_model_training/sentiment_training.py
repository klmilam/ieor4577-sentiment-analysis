"""
Main sentiment training script

Author pharnoux


"""

import os
import argparse
from datetime import datetime
import gcp_model_training.sentiment_dataset as sentiment_dataset
import gcp_model_training.sentiment_model_cnn as sentiment_model_cnn
import gcp_model_training.config_holder as config_holder

def main(args):
    """Main training method

    """

    print("Preparing for training...")
    training_config = config_holder.ConfigHolder(args.config_file).config

    training_config["num_epoch"] = args.num_epoch

    train_dataset = sentiment_dataset.train_input_fn(args.train, training_config)
    validation_dataset = sentiment_dataset.validation_input_fn(
        args.validation, training_config)
    eval_dataset = sentiment_dataset.eval_input_fn(args.eval, training_config)

    model = sentiment_model_cnn.keras_model_fn(None, training_config, args)

    print("Starting training...")

    model.fit(
        x=train_dataset[0], y=train_dataset[1],
        steps_per_epoch=int(args.num_train_samples / args.batch_size),
        epochs=training_config["num_epoch"],
        validation_data=(validation_dataset[0], validation_dataset[1]),
        validation_steps=int(args.num_val_samples / args.batch_size)
    )

    score = model.evaluate(
        eval_dataset[0], eval_dataset[1],
        steps=int(args.num_test_samples / args.batch_size),
        verbose=0)

    print("Test loss:{}".format(score[0]))
    print("Test accuracy:{}".format(score[1]))

    sentiment_model_cnn.save_model(model, os.path.join(args.model_output_dir, "sentiment_model.h5"))


def get_arg_parser():
    """
    Adding this method to unit test

    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="gs://internal-klm/sentiment-analysis/output/20200308231229/train",
        help="The directory where the training data is stored.")
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default="gs://internal-klm/sentiment-analysis/output/20200308231229/eval",
        help="The directory where the validation data is stored.")
    parser.add_argument(
        "--eval",
        type=str,
        required=False,
        default="gs://internal-klm/sentiment-analysis/output/20200308231229/predict")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=False,
        default="gs://internal-klm/sentiment-analysis/model-dir/model/" + timestamp)
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default="gs://internal-klm/sentiment-analysis/model-dir/" + timestamp)
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
        help="The number of steps to use for training.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="gs://internal-klm/sentiment-analysis/training_config.json",
        help="The path to the training config file.")
    parser.add_argument(
        "--job-dir",
        type=str)
    parser.add_argument(
        "--num_cnn_layers",
        type=int,
        default=3)
    parser.add_argument(
        '--cnn-layer-sizes-scale-factor',
        help="Determine how the sizes of the CNN filters decay.",
        default=2,
        type=float)
    parser.add_argument(
        '--first-filter-size',
        help="Filter size of the first CNN layer.",
        default=400,
        type=int
    )
    parser.add_argument(
        "--num_dense_layers",
        type=int,
        default=3)
    parser.add_argument(
        '--dense_layer_sizes_scale_factor',
        help="Determine how the sizes of the dense layers decay.",
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--first_layer_size',
        help="Filter size of the first dense layer.",
        default=1024,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int)
    parser.add_argument(
        "--num_train_samples",
        default=1120650,
        type=int)
    parser.add_argument(
        "--num_val_samples",
        default=232871,
        type=int)
    parser.add_argument(
        "--num_test_samples",
        default=232509,
        type=int)
    return parser


if __name__ == "__main__":
    PARSER = get_arg_parser()
    ARGS = PARSER.parse_args()
    main(ARGS)
