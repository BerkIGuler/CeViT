import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains CeViT for OFDM channel estimation')

    parser.add_argument(
        '--train_set',
        type=str, required=True,
        help='train set folder name')
    parser.add_argument(
        '--val_set',
        type=str, required=True,
        help='val set folder name')
    parser.add_argument(
        '--test_set',
        type=str, required=True,
        help='test set folder name')
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
    parser.add_argument(
        '--cuda',
        type=int, default=0,
        help='Which CUDA interface to use, Always 0 for single GPU machines')
    parser.add_argument(
        '--lr',
        type=float, default=1e-3,
        help='initial learning rate')

    args = parser.parse_args()
    return args
