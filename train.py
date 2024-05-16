import os
import torch
import numpy as np
from preprocess import PatchEmbedding, InversePatchEmbedding
from dataloader import MatDataset, get_test_dataloaders
from torch.utils.data import DataLoader
from token_module import TokenModule
from model import Encoder
from train_helpers import get_test_stats
from plot_helpers import plot_test_stats
from torch.utils.tensorboard import SummaryWriter
from parser import parse_arguments

args = parse_arguments()
log_dir = "runs"
writer = SummaryWriter(os.path.join(log_dir, args.exp_name))

test_set_size_per_point = 2000
batch_size = 32
epoch = 10
model_dim = 128  # transformer linear projection dim
n_head = 4
patch_dim = 40  # patch embedding dim
input_dim = 46
dropout = 0.1
token_embedding_dim = 168
device = "cuda:0"

parent_dir = os.path.dirname(os.getcwd())

# train and val set folders
train_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "train_dataset")
val_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "val_dataset")

# test set folders
ds_test_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "ds_test_dataset")
mds_test_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "mds_test_dataset")
snr_test_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "snr_test_dataset")
mismatched_test_data_dir = os.path.join(parent_dir, "datasets", args.dataset_version, "mismatched_test_dataset")

# train dataloader
train_dataset = MatDataset(train_data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val dataloader
val_dataset = MatDataset(val_data_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# test dataloaders
ds_test_dataloaders = get_test_dataloaders(ds_test_data_dir, batch_size=batch_size)
mds_test_dataloaders = get_test_dataloaders(mds_test_data_dir, batch_size=batch_size)
snr_test_dataloaders = get_test_dataloaders(snr_test_data_dir, batch_size=batch_size)
mismatched_test_dataloaders = get_test_dataloaders(mismatched_test_data_dir, batch_size=batch_size)

# modules
patcher = PatchEmbedding().to(device)
inverse_patcher = InversePatchEmbedding().to(device)
tokenizer = TokenModule(input_size=1, embedding_dim=token_embedding_dim).to(device)
encoder = Encoder(
    input_dim=input_dim,
    output_dim=patch_dim,
    d_model=model_dim,
    nhead=n_head,
    activation="gelu",
    dropout=dropout).to(device)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for ep in range(epoch):
    train_loss = 0.0
    encoder.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        ls_channel, ideal_channel, meta_data = batch
        ls_channel, ideal_channel = ls_channel.to(device), ideal_channel.to(device)
        file_no, SNR, delay_spread, max_dop_shift, ch_type = meta_data
        SNR, delay_spread, max_dop_shift = SNR.to(device), delay_spread.to(device), max_dop_shift.to(device)

        ls_channel = patcher(ls_channel)
        token_encodings = tokenizer(SNR, delay_spread, max_dop_shift)
        model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

        encoder_output = encoder(model_input)
        estimated_channel = inverse_patcher(encoder_output)

        loss = torch.nn.MSELoss()
        output = loss(estimated_channel, ideal_channel)
        output.backward()

        optimizer.step()
        train_loss += output.item() * batch[0].size(0)  # Accumulate batch loss

    train_loss /= len(train_dataset)  # Calculate average epoch loss
    writer.add_scalar(tag='training loss',
                      scalar_value=train_loss,
                      global_step=ep + 1)
    print(f"Train [{ep+1}/{epoch}], Loss: {train_loss}")
    print(f"Train [{ep + 1}/{epoch}], MSE Error: {20 * np.log10(train_loss):.4f} dB")

    val_loss = 0.0
    encoder.eval()
    for batch in val_dataloader:
        ls_channel, ideal_channel, meta_data = batch
        ls_channel, ideal_channel = ls_channel.to(device), ideal_channel.to(device)
        file_no, SNR, delay_spread, max_dop_shift, ch_type = meta_data
        SNR, delay_spread, max_dop_shift = SNR.to(device), delay_spread.to(device), max_dop_shift.to(device)

        ls_channel = patcher(ls_channel)
        token_encodings = tokenizer(SNR, delay_spread, max_dop_shift)
        model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

        encoder_output = encoder(model_input)
        estimated_channel = inverse_patcher(encoder_output)

        loss = torch.nn.MSELoss()
        output = loss(estimated_channel, ideal_channel)
        val_loss += output.item() * batch[0].size(0)  # Accumulate batch loss

    val_loss /= len(val_dataset)  # Calculate average epoch loss
    writer.add_scalar(tag='val loss',
                      scalar_value=val_loss,
                      global_step=ep + 1)
    print(f"Val [{ep+1}/{epoch}], Loss: {val_loss}")
    print(f"Val [{ep+1}/{epoch}], Error: {20 * np.log10(val_loss):.4f} dB")


ds_stats = get_test_stats(
    encoder=encoder, patcher=patcher,
    inverse_patcher=inverse_patcher, tokenizer=tokenizer,
    test_dataloaders=ds_test_dataloaders,
    device=device, var_name="DS",
    test_set_size_per_point=test_set_size_per_point
)

mds_stats = get_test_stats(
    encoder=encoder, patcher=patcher,
    inverse_patcher=inverse_patcher, tokenizer=tokenizer,
    test_dataloaders=mds_test_dataloaders,
    device=device, var_name="MDS",
    test_set_size_per_point=test_set_size_per_point
)

snr_stats = get_test_stats(
    encoder=encoder, patcher=patcher,
    inverse_patcher=inverse_patcher, tokenizer=tokenizer,
    test_dataloaders=snr_test_dataloaders,
    device=device, var_name="SNR",
    test_set_size_per_point=test_set_size_per_point
)

mismatched_stats = get_test_stats(
    encoder=encoder, patcher=patcher,
    inverse_patcher=inverse_patcher, tokenizer=tokenizer,
    test_dataloaders=mismatched_test_dataloaders,
    device=device, var_name="Mismatched",
    test_set_size_per_point=test_set_size_per_point
)

writer.add_figure(tag='MSE vs. Doppler Spread',
                  figure=plot_test_stats(var_name="Doppler Spread (ns)", stats=ds_stats))
writer.add_figure(tag='MSE vs. Max. Doppler Shift',
                  figure=plot_test_stats(var_name="Max. Doppler Shift (Hz)", stats=mds_stats))
writer.add_figure(tag='MSE vs. SNR',
                  figure=plot_test_stats(var_name="SNR (dB)", stats=snr_stats))
writer.add_figure(tag='MSE vs. SNR if Mismatch',
                  figure=plot_test_stats(var_name="SNR(dB)", stats=mismatched_stats))






