import numpy as np
import scipy.io as sio
import os
from prettytable import PrettyTable


def to_db(val):
    return 10 * np.log10(val)


def mse(x, y):
    """calculates mse between two complex np array"""
    mse_xy = np.mean(np.square(np.abs(x - y)))
    # to dB
    mse_xy_db = to_db(mse_xy)
    return mse_xy_db


def get_mse_per_folder(folders_dir):
    """calculates average mse for each sub-folder of a folder"""
    mse_sums = {}
    folders = os.listdir(folders_dir)
    folders = sorted(folders, key=lambda x: int(x.split("_")[1]))
    for folder in folders:
        _, val = folder.split("_")
        mse_sum = 0
        folder_size = len(os.listdir(os.path.join(folders_dir, folder)))
        for file in os.listdir(os.path.join(folders_dir, folder)):
            mat_data = sio.loadmat(os.path.join(folders_dir, folder, file))['H']
            ls_estimate = mat_data[:, :, 2]
            ideal = mat_data[:, :, 0]
            mse_sum += mse(ls_estimate, ideal)
        mse_sum /= folder_size
        mse_sums[int(val)] = mse_sum
    return mse_sums


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.remaining_patience = patience
        self.min_loss = None

    def early_stop(self, loss):
        return_val = False

        if self.min_loss is None:
            self.min_loss = loss
        elif loss < self.min_loss:
            self.min_loss = loss
            self.remaining_patience = self.patience
        else:
            self.remaining_patience -= 1

        if self.remaining_patience == 0:
            return_val = True

        return return_val