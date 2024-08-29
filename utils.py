import numpy as np
import scipy.io as sio
import os


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
