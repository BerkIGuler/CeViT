from utils import to_db


def get_all_test_stats(model, test_dataloaders, loss):

    ds_stats = get_test_stats(model, test_dataloaders=test_dataloaders["DS"], loss=loss)
    mds_stats = get_test_stats(model, test_dataloaders=test_dataloaders["MDS"], loss=loss)
    snr_stats = get_test_stats(model, test_dataloaders=test_dataloaders["SNR"], loss=loss)

    return ds_stats, mds_stats, snr_stats


def get_test_stats(model, test_dataloaders, loss):
    """For a given test loaders evaluate performance of the encoder for each test set and report"""
    stats = {}
    test_dataloaders = sorted(test_dataloaders, key=lambda x: int(x[0].split("_")[1]))
    for name, test_dataloader in test_dataloaders:
        var, val = name.split("_")
        test_loss = eval_model(model, test_dataloader, loss)
        # complex mse for N complex elements = 2 x mse for 2N real elements
        complex_mse_loss = 2 * test_loss
        db_error = to_db(complex_mse_loss)
        print(f"{var}:{val} Test MSE: {db_error:.4f} dB")
        stats[int(val)] = db_error
    return stats


def eval_model(model, eval_dataloader, loss):
    eval_loss = 0.0
    model.eval()
    for batch in eval_dataloader:
        estimated_channel, ideal_channel = model(batch)
        output = loss(estimated_channel, ideal_channel)
        eval_loss += output.item() * batch[0].size(0)  # Accumulate batch loss
    eval_loss /= len(eval_dataloader.dataset)  # Calculate average loss
    return eval_loss


def train_model(model, optimizer, loss, scheduler, train_dataloader):
    train_loss = 0.0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        estimated_channel, ideal_channel = model(batch)
        output = loss(estimated_channel, ideal_channel)
        output.backward()

        optimizer.step()
        train_loss += output.item() * batch[0].size(0)  # Accumulate batch loss
    scheduler.step()
    train_loss /= len(train_dataloader.dataset)  # Calculate average epoch loss
    return train_loss
