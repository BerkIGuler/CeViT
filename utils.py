import numpy as np
import scipy.io as sio
import os


def mse(x, y):
    """calculates mse between two complex np array"""
    mse_xy = np.mean(np.square(np.abs(x - y)))
    # to dB
    mse_xy_db = 20 * np.log10(mse_xy)
    return mse_xy_db


def get_mse_per_folder(folders_dir):
    """calculates average mse for each sub-folder of a folder"""
    mse_sums = {}
    for folder in os.listdir(folders_dir):
        mse_sum = 0
        folder_size = len(os.listdir(os.path.join(folders_dir, folder)))
        for file in os.listdir(os.path.join(folders_dir, folder)):
            mat_data = sio.loadmat(os.path.join(folders_dir, folder, file))['H']
            ls_estimate = mat_data[:, :, 3]
            ideal = mat_data[:, :, 1]
            mse_sum += mse(ls_estimate, ideal)
        mse_sum /= folder_size
        mse_sums[folder] = mse_sum
    return mse_sums
