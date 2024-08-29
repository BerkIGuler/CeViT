import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from preprocess import extract_values, concat_complex_channel


class MatDataset(Dataset):
    """Loads .mat channel data in chunks"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mat_data = sio.loadmat(os.path.join(self.data_dir, self.file_list[idx]))
        # LS channel estimate, data is complex numbers
        h_ls = torch.tensor(mat_data['H'][:, :, 2], dtype=torch.cfloat)
        # ideal channel, data is complex numbers
        h_ideal = torch.tensor(mat_data['H'][:, :, 0], dtype=torch.cfloat)
        # SNR, delay spread, max. dopp. shift, and delay profile values
        # are extracted from file name
        meta_data = extract_values(self.file_list[idx])
        if meta_data is None:
            raise ValueError("File name format could not be recognized")
        return concat_complex_channel(h_ls), concat_complex_channel(h_ideal), meta_data


def get_test_dataloaders(dataset_dir, batch_size):
    """returns a list of dataloaders, each for a folder in dataset_dir"""
    test_datasets = [(sub_folder, MatDataset(os.path.join(dataset_dir, sub_folder)))
                     for sub_folder in os.listdir(dataset_dir)]
    test_dataloaders = [(name, DataLoader(test_dataset, batch_size=batch_size, shuffle=True))
                        for (name, test_dataset) in test_datasets]
    return test_dataloaders
