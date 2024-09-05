import torch
import re


def extract_values(file_name):
    """extract channel info from file name

    :param file_name:
        str: file name
    :return
        tuple: channel info with fields
            (file_number, snr, delay_spread, max_doppler_shift,
             pilot placement frequency along subcarriers, channel_type)
    """

    pattern = r'(\d+)_SNR-(\d+)_DS-(\d+)_DOP-(\d+)_N-(\d+)_([A-Z\-]+)\.mat'
    match = re.match(pattern, file_name)
    if match:
        file_no = torch.tensor([int(match.group(1))], dtype=torch.float)
        snr_value = torch.tensor([int(match.group(2))], dtype=torch.float)
        ds_value = torch.tensor([int(match.group(3))], dtype=torch.float)
        dop_value = torch.tensor([int(match.group(4))], dtype=torch.float)
        n = torch.tensor([int(match.group(5))], dtype=torch.float)
        channel_type = [match.group(6)]
        return file_no, snr_value, ds_value, dop_value, n, channel_type
    else:
        raise ValueError("Cannot extract file information.")


def concat_complex_channel(channel_matrix):
    """makes channel_matrix real by doubling size
        :param
            channel_matrix (torch.Tensor): Complex channel matrix of size
            (num_channels, num_subcarriers, num_ofdm_symbols)
        :return
            channel_matrix (torch.Tensor): Real channel matrix of size
            (num_channels, num_subcarriers, 2 x num_ofdm_symbols)
    """

    real_channel_m = torch.real(channel_matrix)
    imag_channel_m = torch.imag(channel_matrix)
    cat_channel_m = torch.cat((real_channel_m, imag_channel_m), dim=1)
    return cat_channel_m


def inverse_concat_complex_channel(channel_matrix):
    """reverses channel matrix concatenation along OFDM symbols"""
    real_to_imag_idx = int(channel_matrix.shape[-1] / 2)
    real_channel_m = channel_matrix[:, :, :real_to_imag_idx]
    imag_channel_m = channel_matrix[:, :, real_to_imag_idx:]
    complex_channel_m = torch.complex(real_channel_m, imag_channel_m)
    return complex_channel_m
