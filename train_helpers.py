import torch
from utils import to_db


def get_all_test_stats(
        encoder, patcher, inverse_patcher, tokenizer,
        test_dataloaders, device, loss):

    ds_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders["DS"],
        device=device, loss=loss)

    mds_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders["MDS"],
        device=device, loss=loss
    )

    snr_stats = get_test_stats(
        encoder=encoder, patcher=patcher,
        inverse_patcher=inverse_patcher, tokenizer=tokenizer,
        test_dataloaders=test_dataloaders["SNR"],
        device=device, loss=loss
    )

    return ds_stats, mds_stats, snr_stats


def get_test_stats(
        encoder, patcher, inverse_patcher, tokenizer,
        test_dataloaders, device, loss):
    """For a given test loaders evaluate performance of the encoder for each test set and report"""
    stats = {}
    test_dataloaders = sorted(test_dataloaders, key=lambda x: int(x[0].split("_")[1]))
    for name, test_dataloader in test_dataloaders:
        var, val = name.split("_")
        test_loss = eval_model(
            encoder, test_dataloader, device,
            patcher, tokenizer, inverse_patcher, loss)
        db_error = to_db(test_loss)
        print(f"{var}:{val} Test MSE: {db_error:.4f} dB")
        stats[int(val)] = db_error
    return stats


def eval_model(encoder, eval_dataloader, device, patcher, tokenizer, inverse_patcher, loss):
    eval_loss = 0.0
    encoder.eval()
    for batch in eval_dataloader:
        estimated_channel, ideal_channel = forward_pass(
            batch, device, patcher, tokenizer,
            encoder, inverse_patcher)
        output = loss(estimated_channel, ideal_channel)
        eval_loss += output.item() * batch[0].size(0)  # Accumulate batch loss
    eval_loss /= len(eval_dataloader.dataset)  # Calculate average loss
    return eval_loss


def train_model(
        encoder, optimizer, loss, scheduler, train_dataloader,
        device, patcher, tokenizer, inverse_patcher):
    train_loss = 0.0
    encoder.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        estimated_channel, ideal_channel = forward_pass(
            batch, device, patcher,
            tokenizer, encoder, inverse_patcher)

        output = loss(estimated_channel, ideal_channel)
        output.backward()

        optimizer.step()
        train_loss += output.item() * batch[0].size(0)  # Accumulate batch loss
    scheduler.step()
    train_loss /= len(train_dataloader.dataset)  # Calculate average epoch loss
    return train_loss


def forward_pass(batch, device, patcher, tokenizer, encoder, inverse_patcher):
    ls_channel, ideal_channel, meta_data = batch
    ls_channel, ideal_channel = ls_channel.to(device), ideal_channel.to(device)
    _, snr, delay_spread, max_dop_shift, _, _ = meta_data
    snr, delay_spread, max_dop_shift = snr.to(device), delay_spread.to(device), max_dop_shift.to(device)

    ls_channel = patcher(ls_channel)
    token_encodings = tokenizer(snr, delay_spread, max_dop_shift)
    model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

    encoder_output = encoder(model_input)
    estimated_channel = inverse_patcher(encoder_output)

    return estimated_channel, ideal_channel
