import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a transformer model for OFDM channel estimation')

    parser.add_argument(
        '--model_name',
        type=str, required=True,
        help='model name for the log file')
    parser.add_argument(
        '--epoch',
        type=int, default=10,
        help='Number of epochs to train for')
    parser.add_argument(
        '--batch_size',
        type=int, default=64,
        help='Batch size for each gradient descent')

    args = parser.parse_args()
    return args
