""" Command line interface for DeepEHRGraph. """
import argparse

from deepehrgraph.dataset.dataset import create_master_dataset
from deepehrgraph.training.train import train


def main() -> None:
    """Main function for the command line interface."""
    parser = argparse.ArgumentParser(
        prog="DeepEHRGraph",
        description="EHR data analysis and prediction with deep learning models.",
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="subcommand",
        description="Available subcommands",
        required=True,
    )

    parser_dataset = subparsers.add_parser(
        "dataset", help="Generate master mimic iv demo dataset as csv file/"
    )
    parser_dataset.add_argument(
        "--fir_name",
        type=str,
        default="data",
        help="Directory name to store the dataset.",
    )
    parser_dataset.set_defaults(func=create_master_dataset)

    parser_train = subparsers.add_parser("train", help="Train a model on a dataset.")
    parser_train.add_argument(
        "--nb-epochs",
        type=str,
        default=1,
        help="Number of epochs to train the model for.",
    )
    parser_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
