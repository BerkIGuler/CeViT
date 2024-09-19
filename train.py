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
from utils import get_mse_per_folder, get_model_details


def main():
    args = parse_arguments()

    # generate exp log file
    log_dir = "runs"
    os.makedirs("runs", exist_ok=True)
    exp_no = 1
    while f"exp{exp_no}" in os.listdir(log_dir):
        exp_no += 1

    writer = SummaryWriter(os.path.join(log_dir, f"exp{exp_no}"))

    params_dict = {
        "bsize": args.batch_size,
        "max_epoch": args.max_epoch,
        "patience": args.patience,
        "test_every_n_epoch": args.test_every_n,
        "train_set_name": args.train_set,
        "val_set_name": args.val_set,
        "test_set_name": args.test_set,
        "lr": args.lr,
        "cuda": args.cuda
    }

    device = f"cuda:{params_dict['cuda']}"

    # Transformer Params
    model_dim = 128  # transformer linear projection dim
    n_head = 4
    patch_dim = 40  # patch embedding dim
    input_dim = 46
    dropout = 0.1
    token_embedding_dim = 168

    parent_dir = os.path.dirname(os.getcwd())

    # train and val set folders
    train_data_dir = os.path.join(parent_dir, "datasets", params_dict["train_set_name"])
    val_data_dir = os.path.join(parent_dir, "datasets", params_dict["val_set_name"])

    # test set folders
    ds_test_data_dir = os.path.join(parent_dir, "datasets", params_dict["test_set_name"], "DS_test_set")
    mds_test_data_dir = os.path.join(parent_dir, "datasets", params_dict["test_set_name"], "MDS_test_set")
    snr_test_data_dir = os.path.join(parent_dir, "datasets", params_dict["test_set_name"], "SNR_test_set")

    # train dataloader
    train_dataset = MatDataset(train_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=params_dict["bsize"], shuffle=True)

    # val dataloader
    val_dataset = MatDataset(val_data_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=params_dict["bsize"], shuffle=True)

    # test dataloaders
    ds_test_dataloaders = get_test_dataloaders(ds_test_data_dir, batch_size=params_dict["bsize"])
    mds_test_dataloaders = get_test_dataloaders(mds_test_data_dir, batch_size=params_dict["bsize"])
    snr_test_dataloaders = get_test_dataloaders(snr_test_data_dir, batch_size=params_dict["bsize"])

    test_dataloaders = {
        "DS": ds_test_dataloaders,
        "MDS": mds_test_dataloaders,
        "SNR": snr_test_dataloaders,
    }

    model = CeViT(device, token_embedding_dim, input_dim, patch_dim, model_dim, n_head, dropout)
    num_total_params, model_summary = get_model_details(model)
    writer.add_text("Number of Parameters", str(num_total_params))
    early_stopper = EarlyStopping(params_dict["patience"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params_dict["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9954)
    loss = torch.nn.MSELoss()

    for ep in range(params_dict["max_epoch"]):
        train_loss = train_model(model, optimizer, loss, scheduler, train_dataloader)
        writer.add_scalar(tag='Loss/Train',
                          scalar_value=train_loss,
                          global_step=ep + 1)

        val_loss = eval_model(model, val_dataloader, loss)
        writer.add_scalar(tag='Loss/Val',
                          scalar_value=val_loss,
                          global_step=ep + 1)
        if early_stopper.early_stop(val_loss):
            break

        if (ep + 1) % params_dict["test_every_n_epoch"] == 0:
            ds_stats, mds_stats, snr_stats = get_all_test_stats(model, test_dataloaders, loss)

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

    try:
        writer.add_hparams(
            hparam_dict=params_dict,
            metric_dict={"last_epoch": (ep + 1)},
            run_name=".")
    except NameError:
        writer.add_text("Error", "Parameter dictionary could not be logged.")

    try:
        for (ds_val, snr_val, mds_val) in zip(ds_stats.keys(), snr_stats.keys(), mds_stats.keys()):
            writer.add_scalars(
                "Delay Spread",
                {"LS": ds_ls_stats[ds_val],
                 "CeViT": ds_stats[ds_val]}, ds_val)
            writer.add_scalars(
                "SNR",
                {"LS": snr_ls_stats[snr_val],
                 "CeViT": snr_stats[snr_val]}, snr_val)
            writer.add_scalars(
                "Doppler Shift",
                {"LS": mds_ls_stats[mds_val],
                 "CeViT": mds_stats[mds_val]}, mds_val)
    except NameError:
        writer.add_text("Error", "Test results could not be logged")

    writer.close()


if __name__ == "__main__":
    main()
