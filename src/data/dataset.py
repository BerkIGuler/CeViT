import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random


class BilinearInterpolation:
    """
    Bilinear interpolation of LS channel estimates on a regular pilot grid.
    Pilots sit on even subcarriers (0, 2, …, 118). Pilot OFDM symbols can be
    at any subset of symbol indices (e.g. [2], [2, 3], or [2, 7, 11]).
    1) Linear interpolation across subcarriers (frequency).
    2) Piecewise linear interpolation / extrapolation across symbols (time).
    """

    def __init__(self):
        pass

    def __call__(self, hp_ls):
        hp_ls = hp_ls.clone() if hasattr(hp_ls, "clone") else np.asarray(hp_ls, dtype=np.complex128).copy()
        pilot_syms = np.where(hp_ls[0, :].real != 0)[0]

        # Frequency: average even-indexed neighbours to fill odd subcarrier rows
        hp_ls[1:-1:2, pilot_syms] = (hp_ls[:-2:2, pilot_syms] + hp_ls[2::2, pilot_syms]) / 2
        hp_ls[-1, pilot_syms] = hp_ls[-2, pilot_syms]

        # Time: nearest-neighbour or piecewise linear interpolation / extrapolation
        if len(pilot_syms) == 1:
            hp_ls[:] = hp_ls[:, pilot_syms[0] : pilot_syms[0] + 1]
        elif len(pilot_syms) == 2:
            p0, p1 = int(pilot_syms[0]), int(pilot_syms[1])
            if p1 - p0 == 1:
                mid = (p0 + p1) / 2.0
                for i in range(hp_ls.shape[1]):
                    if i <= mid:
                        hp_ls[:, i] = hp_ls[:, p0]
                    else:
                        hp_ls[:, i] = hp_ls[:, p1]
            else:
                slope = (hp_ls[:, p1] - hp_ls[:, p0]) / (p1 - p0)
                for i in range(hp_ls.shape[1]):
                    if i not in (p0, p1):
                        hp_ls[:, i] = hp_ls[:, p0] + slope * (i - p0)
        elif len(pilot_syms) == 3:
            p0, p1, p2 = int(pilot_syms[0]), int(pilot_syms[1]), int(pilot_syms[2])
            slope_1 = (hp_ls[:, p1] - hp_ls[:, p0]) / (p1 - p0)
            slope_2 = (hp_ls[:, p2] - hp_ls[:, p1]) / (p2 - p1)
            for i in range(hp_ls.shape[1]):
                if i < p0:
                    hp_ls[:, i] = hp_ls[:, p0] + slope_1 * (i - p0)
                elif p0 < i < p1:
                    hp_ls[:, i] = hp_ls[:, p0] + slope_1 * (i - p0)
                elif p1 < i < p2:
                    hp_ls[:, i] = hp_ls[:, p1] + slope_2 * (i - p1)
                elif i > p2:
                    hp_ls[:, i] = hp_ls[:, p2] + slope_2 * (i - p2)
        return hp_ls


class TDLDataset(Dataset):
    def __init__(
        self, data_path, *, normalization_stats=None,return_pilots_only=True, num_subcarriers=120,
        num_symbols=14, SNRs=[0, 5, 10, 15, 20, 25, 30],
        pilot_symbols=[2, 7, 11], pilot_every_n=2):
        """
        This class loads the data from the folder and returns a dataset of channels.

        data_path: path to the folder containing the data (root; all .npy under it are used)
        return_pilots_only: if True, only the LS channel estimate at pilots are returned
            if False, the LS channel estimate is returned as a sparse channel matrix with non-zero 
            values only at the pilot subcarriers and time instants.
        num_subcarriers: number of subcarriers
        num_symbols: number of OFDM symbols

        SNRs: list of SNR values to randomly sample from when return LS estimates.
            AWGN is added to simulate LS estimatation error
        pilot_symbols: list of OFDM symbol indices where pilots are placed
        pilot_every_n: number of subcarriers between pilot subcarriers
        """
        
        # file_size is accepted but no longer used; we infer per-file sizes from the data itself.
        self.normalization_stats = normalization_stats
        self.return_pilots_only = return_pilots_only
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.SNRs = SNRs
        self.pilot_symbols = pilot_symbols
        self.pilot_every_n = pilot_every_n
        self.noise_variance = self._get_noise_variance(SNRs)

        # Collect all .npy files under data_path (recursively)
        self.file_list = list(Path(data_path).rglob("*.npy"))
        self.stats = self._get_stats_per_file(self.file_list)
        self.data = self._load_data_from_folder(self.file_list, self.normalization_stats)

        # Build a flat index over all (file, sample_idx) pairs so we can handle
        # heterogeneous numbers of channels per .npy file.
        self.index = []
        for file_path in self.file_list:
            file_data = self.data[file_path]
            num_channels = file_data.shape[0]
            for sample_idx in range(num_channels):
                self.index.append((file_path, sample_idx))

        self.pilot_mask = self._get_pilot_mask()

        self.num_pilot_symbols = len(self.pilot_symbols)
        self.num_pilot_subcarriers = int(self.pilot_mask.sum()) // self.num_pilot_symbols

        self.bilinear_interp = BilinearInterpolation()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, sample_idx = self.index[idx]
        channels = self.data[file_path]
        channel = channels[sample_idx].squeeze().T

        SNR = random.choice(self.SNRs)
        LS_channel_at_pilots = self._get_LS_estimate_at_pilots(channel, SNR)
        stats = self.stats[file_path].copy()
        stats["SNR"] = SNR

        if not self.return_pilots_only:
            # Sparse 120×14: fill via bilinear interpolation so CeViT gets full grid
            LS_channel_at_pilots = self.bilinear_interp(LS_channel_at_pilots)

        LS_channel_torch = torch.from_numpy(LS_channel_at_pilots).to(torch.complex64)
        channel_torch = torch.from_numpy(channel).to(torch.complex64)
        return LS_channel_torch, channel_torch, stats
    
    @staticmethod
    def _load_data_from_folder(file_list, normalization_stats=None):
        data = {}
        for file_path in file_list:
            file_data = np.load(file_path)
            if normalization_stats is not None:
                normalized_real = (file_data.real - normalization_stats["real_mean"]) / normalization_stats["real_std"]
                normalized_imag = (file_data.imag - normalization_stats["imag_mean"]) / normalization_stats["imag_std"]
                file_data = normalized_real + 1j * normalized_imag
            data[file_path] = file_data
        return data

    @staticmethod
    def _get_stats_per_file(file_list):
        stats = {}

        for file_path in file_list:
            file_name = str(file_path.stem)
            file_parts = file_name.split("_")

            if file_parts[0] == "delay":
                try:
                    delay_spread = int(file_parts[2])  # [delay, spread, x, doppler, y] or [delay, x, doppler, y]
                except ValueError:
                    delay_spread = int(file_parts[1])  # [delay, x, doppler, y]
                doppler_shift = int(file_parts[-1])
            elif file_parts[0] == "doppler":
                doppler_shift = int(file_parts[1])  # [doppler, x, delay, spread, y]
                delay_spread = int(file_parts[-1])
            else:
                raise ValueError(f"File {file_name} has unexpected format")
            
            if file_path not in stats:
                stats[file_path] = {"doppler_shift": doppler_shift, "delay_spread": delay_spread}
            else:
                raise ValueError(f"File {file_path} already in stats, but should not be")
            
        return stats
    
    def _get_noise_variance(self, SNRs):
        noise_variances = []
        for SNR in SNRs:
            noise_variance = 1 / (10**(SNR / 10))
            noise_variances.append(noise_variance)
        return np.mean(np.array(noise_variances))
    
    def _get_LS_estimate_at_pilots(self, channel_matrix, SNR):
        # unit symbol power and unit channel power --> rx noise var = LS error var
        noise_std = np.sqrt(1 / (10**(SNR / 10)))
        noise_real_imag = noise_std / np.sqrt(2)

        if self.return_pilots_only:
            pilot_mask_bool = self.pilot_mask.astype(bool)
            channel_at_pilots = channel_matrix[pilot_mask_bool]
            channel_at_pilots = channel_at_pilots.reshape(self.num_pilot_subcarriers, self.num_pilot_symbols)
            noise_real = noise_real_imag * np.random.randn(self.num_pilot_subcarriers, self.num_pilot_symbols)
            noise_imag = noise_real_imag * np.random.randn(self.num_pilot_subcarriers, self.num_pilot_symbols)
            noise = noise_real + 1j * noise_imag
        else:
            channel_at_pilots = self.pilot_mask * channel_matrix
            noise_real = noise_real_imag * np.random.randn(self.num_subcarriers, self.num_symbols)
            noise_imag = noise_real_imag * np.random.randn(self.num_subcarriers, self.num_symbols)
            noise = noise_real + 1j * noise_imag
            noise = noise * self.pilot_mask
        
        channel_at_pilots_LS = channel_at_pilots + noise
            
        return channel_at_pilots_LS

    def _get_pilot_mask(self):
        pilot_mask = np.zeros((self.num_subcarriers, self.num_symbols))
        pilot_mask_subcarrier_indices = np.arange(0, self.num_subcarriers, self.pilot_every_n)
        pilot_mask[np.ix_(pilot_mask_subcarrier_indices, self.pilot_symbols)] = 1
        return pilot_mask

def get_in_distribution_test_datasets(test_path, return_pilots_only=False, SNRs=[20], pilot_symbols=[2]):
    folder_list = list(Path(test_path).glob("*"))
    for folder in folder_list:
        if folder.is_dir():
            dataset = TDLDataset(
                data_path=folder,
                normalization_stats=None,
                return_pilots_only=return_pilots_only,
                SNRs=SNRs,
                pilot_symbols=pilot_symbols
                )
            yield folder.name, dataset