import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a transformer model for OFDM channel estimation')

    # Add your arguments here
    parser.add_argument(
        '--dataset_version',
        type=str, required=True,
        choices=["v1", "v2"],
        help='Determines which dataset to use')
    parser.add_argument(
        '--exp_name',
        type=str, required=True,
        help='Experiment name to name the log file')

    args = parser.parse_args()
    return args
