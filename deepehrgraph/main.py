""" Command line interface for DeepEHRGraph. """
import argparse

from deepehrgraph.dataset.dataset import generate_dataset
from deepehrgraph.dataset.eda import eda
from deepehrgraph.dataset.features_selection import features_selection
from deepehrgraph.training.enums import OutcomeType
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
        "--dir-name",
        type=str,
        default="data",
        help="Directory name to store the dataset.",
    )
    parser_dataset.set_defaults(func=generate_dataset)

    parser_eda = subparsers.add_parser("eda", help="eda")
    parser_eda.set_defaults(func=eda)

    parser_feat_selection = subparsers.add_parser(
        "feat_select", help="features selection"
    )
    parser_feat_selection.add_argument(
        "--dir-name",
        type=str,
        default="data",
        help="Directory name to store the dataset.",
    )
    parser_feat_selection.set_defaults(func=features_selection)

    parser_train = subparsers.add_parser(
        "train", help="Train a model on a dataset for a given outcome."
    )
    parser_train.add_argument(
        "--outcome",
        type=OutcomeType,
        choices=list(OutcomeType),
        default=OutcomeType.INHOSPITAL_MORTALITY,
        help="Outcome to predict.",
    )
    parser_train.add_argument(
        "--max-epochs", type=int, default=50, help="Max number of epochs."
    )
    parser_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
