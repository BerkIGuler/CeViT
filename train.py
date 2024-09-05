import os
import torch
from dataloader import MatDataset, get_test_dataloaders
from torch.utils.data import DataLoader
from model import CeViT
from utils import EarlyStopping
from train_helpers import get_all_test_stats, train_model, eval_model
from plot_helpers import plot_test_stats
from torch.utils.tensorboard import SummaryWriter
from parser import parse_arguments
from utils import get_mse_per_folder, count_parameters


def main():
    args = parse_arguments()
    log_folder = "runs"

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    patience = args.patience
    test_every_n_epoch = args.test_every_n

    exp_name = f"{args.model_name}_ep-{max_epoch}_bs-{batch_size}_pt-{patience}"
    writer = SummaryWriter(os.path.join(log_folder, exp_name))

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

    cevit_model = CeViT(device, token_embedding_dim, input_dim, patch_dim, model_dim, n_head, dropout)
    count_parameters(cevit_model)
    early_stopper = EarlyStopping(patience)
    optimizer = torch.optim.Adam(cevit_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9954)
    loss = torch.nn.MSELoss()

    for ep in range(max_epoch):
        train_loss = train_model(cevit_model, optimizer, loss, scheduler, train_dataloader)
        writer.add_scalar(tag='train loss',
                          scalar_value=train_loss,
                          global_step=ep + 1)

        val_loss = eval_model(cevit_model, val_dataloader, loss)
        writer.add_scalar(tag='val loss',
                          scalar_value=val_loss,
                          global_step=ep + 1)
        if early_stopper.early_stop(val_loss):
            break

        if (ep + 1) % test_every_n_epoch == 0:
            ds_stats, mds_stats, snr_stats = get_all_test_stats(cevit_model, test_dataloaders, loss)

            ds_ls_stats = get_mse_per_folder(ds_test_data_dir)
            mds_ls_stats = get_mse_per_folder(mds_test_data_dir)
            snr_ls_stats = get_mse_per_folder(snr_test_data_dir)

            writer.add_figure(tag=f"MSE vs. Doppler Spread (Epoch:{ep + 1})",
                              figure=plot_test_stats(
                                  x_name="Doppler Spread (ns)",
                                  stats=[ds_stats, ds_ls_stats],
                                  methods=["CE-ViT", "LS"]))
            writer.add_figure(tag=f"MSE vs. Max. Doppler Shift (Epoch:{ep + 1})",
                              figure=plot_test_stats(
                                  x_name="Max. Doppler Shift (Hz)",
                                  stats=[mds_stats, mds_ls_stats],
                                  methods=["CE-ViT", "LS"]))
            writer.add_figure(tag=f"MSE vs. SNR (Epoch:{ep + 1})",
                              figure=plot_test_stats(
                                  x_name="SNR (dB)",
                                  stats=[snr_stats, snr_ls_stats],
                                  methods=["CE-ViT", "LS"]))


if __name__ == "__main__":
    main()
