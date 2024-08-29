import os
import torch
from preprocess import PatchEmbedding, InversePatchEmbedding
from dataloader import MatDataset, get_test_dataloaders
from torch.utils.data import DataLoader
from token_module import TokenModule
from model import Encoder
from train_helpers import get_all_test_stats, train_model, eval_model
from plot_helpers import plot_test_stats
from torch.utils.tensorboard import SummaryWriter
from parser import parse_arguments
from utils import get_mse_per_folder


def main():
    args = parse_arguments()
    log_dir = "runs"
    exp_name = f"{args.model_name}_ep-{args.epoch}_bs-{args.batch_size}"
    writer = SummaryWriter(os.path.join(log_dir, exp_name))

    batch_size = args.batch_size
    epoch = args.epoch
    model_dim = 128  # transformer linear projection dim
    n_head = 4
    patch_dim = 40  # patch embedding dim
    input_dim = 46
    dropout = 0.1
    token_embedding_dim = 168
    device = "cuda:0"

    parent_dir = os.path.dirname(os.getcwd())

    # train and val set folders
    train_data_dir = os.path.join(parent_dir, "datasets", "train")
    val_data_dir = os.path.join(parent_dir, "datasets", "val")

    # test set folders
    ds_test_data_dir = os.path.join(parent_dir, "datasets", "test", "DS_test_set")
    mds_test_data_dir = os.path.join(parent_dir, "datasets", "test", "MDS_test_set")
    snr_test_data_dir = os.path.join(parent_dir, "datasets", "test", "SNR_test_set")

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
    test_dataloaders = {
        "DS": ds_test_dataloaders,
        "MDS": mds_test_dataloaders,
        "SNR": snr_test_dataloaders,
    }

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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9954)
    loss = torch.nn.MSELoss()

    for ep in range(epoch):
        train_loss = train_model(
            encoder, optimizer, loss, scheduler, train_dataloader,
            device, patcher, tokenizer, inverse_patcher)
        writer.add_scalar(tag='train loss',
                          scalar_value=train_loss,
                          global_step=ep + 1)

        val_loss = eval_model(
            encoder, val_dataloader, device,
            patcher, tokenizer, inverse_patcher, loss)
        writer.add_scalar(tag='val loss',
                          scalar_value=val_loss,
                          global_step=ep + 1)

    ds_stats, mds_stats, snr_stats = get_all_test_stats(
            encoder, patcher, inverse_patcher, tokenizer,
            test_dataloaders, device, loss)

    ds_ls_stats = get_mse_per_folder(ds_test_data_dir)
    mds_ls_stats = get_mse_per_folder(mds_test_data_dir)
    snr_ls_stats = get_mse_per_folder(snr_test_data_dir)

    writer.add_figure(tag='MSE vs. Doppler Spread',
                      figure=plot_test_stats(
                          x_name="Doppler Spread (ns)",
                          stats=[ds_stats, ds_ls_stats],
                          methods=["CE-ViT", "LS"]))
    writer.add_figure(tag='MSE vs. Max. Doppler Shift',
                      figure=plot_test_stats(
                          x_name="Max. Doppler Shift (Hz)",
                          stats=[mds_stats, mds_ls_stats],
                          methods=["CE-ViT", "LS"]))
    writer.add_figure(tag='MSE vs. SNR',
                      figure=plot_test_stats(
                          x_name="SNR (dB)",
                          stats=[snr_stats, snr_ls_stats],
                          methods=["CE-ViT", "LS"]))


if __name__ == "__main__":
    main()
