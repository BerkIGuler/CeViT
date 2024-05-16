import os
from utils import get_mse_per_folder
from plot_helpers import plot_test_stats

dataset_version = "v2"

parent_dir = os.path.dirname(os.getcwd())
ds_test_data_dir = os.path.join(parent_dir, "datasets", dataset_version, "ds_test_dataset")
mds_test_data_dir = os.path.join(parent_dir, "datasets", dataset_version, "mds_test_dataset")
snr_test_data_dir = os.path.join(parent_dir, "datasets", dataset_version, "snr_test_dataset")
mismatched_test_data_dir = os.path.join(parent_dir, "datasets", dataset_version, "mismatched_test_dataset")

ds_ls_stats = get_mse_per_folder(ds_test_data_dir)
mds_ls_stats = get_mse_per_folder(mds_test_data_dir)
snr_ls_stats = get_mse_per_folder(snr_test_data_dir)
mismatched_ls_stats = get_mse_per_folder(mismatched_test_data_dir)

print(ds_ls_stats)
plot_test_stats("DS", ds_ls_stats, show=True)
plot_test_stats("MDS", mds_ls_stats, show=True)
plot_test_stats("SNR", snr_ls_stats, show=True)
plot_test_stats("mismatched", mismatched_ls_stats, show=True)
