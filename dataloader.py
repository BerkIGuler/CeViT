import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from preprocess import extract_values, concat_complex_channel


class MatDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mat_data = sio.loadmat(os.path.join(self.data_dir, self.file_list[idx]))
        # LS channel estimate
        x = torch.tensor(mat_data['H'][:, :, 3], dtype=torch.cfloat)
        # ideal channel
        y = torch.tensor(mat_data['H'][:, :, 1], dtype=torch.cfloat)
        meta_data = extract_values(self.file_list[idx])
        return concat_complex_channel(x), concat_complex_channel(y), meta_data
