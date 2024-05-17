import torch
import numpy as np


def get_all_test_stats(
        encoder, patcher, inverse_patcher, tokenizer,
        test_dataloaders, device):

    ds_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders[0],
        device=device, var_name="DS")

    mds_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders[1],
        device=device, var_name="MDS"
    )

    snr_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders[2],
        device=device, var_name="SNR"
    )

    mismatched_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders[3],
        device=device, var_name="Mismatched"
    )

    return ds_stats, mds_stats, snr_stats, mismatched_stats


def get_test_stats(
        encoder, patcher, inverse_patcher, tokenizer,
        test_dataloaders, device, var_name):
    """For a given test loaders evaluate performance of the encoder for each test set and report"""
    stats = {}
    for name, test_dataloader in test_dataloaders:
        test_set_size_per_point = len(test_dataloader.dataset)
        test_loss = 0.0
        encoder.eval()
        for batch in test_dataloader:
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
            test_loss += output.item() * batch[0].size(0)  # Accumulate batch loss

        test_loss /= test_set_size_per_point  # Calculate average loss
        db_error = 20 * np.log10(test_loss)
        print(f"{var_name}:{name} Test MSE: {db_error:.4f} dB")
        stats[int(name)] = db_error
    return stats


def forward_pass(batch, device, patcher, tokenizer, encoder, inverse_patcher):
    ls_channel, ideal_channel, meta_data = batch
    ls_channel, ideal_channel = ls_channel.to(device), ideal_channel.to(device)
    file_no, SNR, delay_spread, max_dop_shift, ch_type = meta_data
    SNR, delay_spread, max_dop_shift = SNR.to(device), delay_spread.to(device), max_dop_shift.to(device)

    ls_channel = patcher(ls_channel)
    token_encodings = tokenizer(SNR, delay_spread, max_dop_shift)
    model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

    encoder_output = encoder(model_input)
    estimated_channel = inverse_patcher(encoder_output)

    return estimated_channel, ideal_channel
