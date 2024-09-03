import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a transformer model for OFDM channel estimation')

    parser.add_argument(
        '--model_name',
        type=str, required=True,
        help='model name for the log file')
    parser.add_argument(
        '--test_every_n',
        type=int, default=10,
        help='tests model every n epoch')
    parser.add_argument(
        '--max_epoch',
        type=int, default=10,
        help='Number of epochs')
    parser.add_argument(
        '--patience',
        type=int, default=3,
        help='Number of consecutive epochs of val loss non-decrease to early stop')
    parser.add_argument(
        '--batch_size',
        type=int, default=64,
        help='Batch size')

    args = parser.parse_args()
    return args
