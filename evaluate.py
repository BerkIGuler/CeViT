import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.data import get_in_distribution_test_datasets
from src.model.cevit import CeViT
from src.train import Trainer


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained CeViT checkpoint (NMSE dB).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    p.add_argument("--pilot_symbols", type=int, nargs="+", default=[2, 7, 11])
    p.add_argument("--pilot_every_n", type=int, default=2)
    p.add_argument("--num_subcarriers", type=int, default=120)
    p.add_argument("--num_symbols", type=int, default=14)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation (e.g. 'cuda' or 'cpu'). Defaults to CUDA if available, else CPU.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to save YAML results. Defaults to <checkpoint_dir>/eval_results_cevit.yaml",
    )
    return p


@torch.no_grad()
def main() -> None:
    args = _build_argparser().parse_args()

    data_root = Path(args.data_path)
    if not data_root.exists():
        alt = Path("/" + str(args.data_path).lstrip("/"))
        hint = f" Did you mean '{alt}'?" if alt.exists() else ""
        raise ValueError(f"--data_path does not exist: '{data_root}'.{hint}")
    if not data_root.is_dir():
        raise ValueError(f"--data_path is not a directory: '{data_root}'.")

    if args.device is not None:
        dev_name_str = str(args.device).lower()
        if dev_name_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"Requested device '{args.device}' but CUDA is not available; falling back to 'cpu'.")
            device = torch.device("cpu")
        else:
            device = torch.device(dev_name_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_pilot_subcarriers = args.num_subcarriers // args.pilot_every_n
    pilot_grid_shape = (num_pilot_subcarriers, len(args.pilot_symbols))

    model = CeViT(
        device=device,
        token_emb_dim=168,
        input_dim=46,
        patch_dim=40,
        model_dim=128,
        n_head=4,
        dropout=0.1,
        num_subcarriers=args.num_subcarriers,
        num_symbols=args.num_symbols,
        pilot_grid_shape=pilot_grid_shape,
    ).to(device)

    ckpt = Trainer.load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out is not None else (checkpoint_path.parent / "eval_results_cevit.yaml")

    results = {}

    for snr in args.snrs:
        print(f"=== SNR = {snr} dB ===")
        results[int(snr)] = {}
        for folder_name, dataset in get_in_distribution_test_datasets(
            Path(args.data_path),
            return_pilots_only=True,
            SNRs=[snr],
            pilot_symbols=args.pilot_symbols,
        ):
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            num_sum = torch.tensor(0.0, device=device)
            den_sum = torch.tensor(0.0, device=device)

            for batch in loader:
                ls_channel, h_true, stats = batch
                snr_t = stats["SNR"].to(device).float().unsqueeze(1)
                delay_spread = stats["delay_spread"].to(device).float().unsqueeze(1)
                max_dop_shift = stats["doppler_shift"].to(device).float().unsqueeze(1)
                meta = (None, snr_t, delay_spread, max_dop_shift, None, None)

                ls_channel = ls_channel.to(device)
                h_true = h_true.to(device)

                est_channel, ideal_channel = model((ls_channel, h_true, meta))

                err = est_channel - ideal_channel
                num_sum += (err * err.conj()).abs().sum()
                den_sum += (ideal_channel * ideal_channel.conj()).abs().sum()

            nmse = num_sum / den_sum
            nmse_db = 10.0 * torch.log10(nmse)
            nmse_db_f = float(nmse_db.detach().cpu())
            results[int(snr)][str(folder_name)] = {"nmse_mean_db": nmse_db_f}
            print(f"  folder={folder_name}: nmse_mean_db = {nmse_db_f:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=True)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

