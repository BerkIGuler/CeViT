import torch
import numpy as np
from preprocess import PatchEmbedding, InversePatchEmbedding
from dataloader import MatDataset
from torch.utils.data import DataLoader
from token_module import TokenModule
from model import Encoder


batch_size = 32
epoch = 40
model_dim = 128  # transformer linear projection dim
n_head = 4
patch_dim = 40  # patch embedding dim
input_dim = 46
dropout = 0.1
token_embedding_dim = 168
device = "cuda:0"

train_data_dir = "train_data"
test_data_dir = "test_data"
val_data_dir = "val_data"

train_dataset = MatDataset(train_data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MatDataset(val_data_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MatDataset(test_data_dir)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

        train_loss += output.item() * batch_size  # Accumulate batch loss

    train_loss /= len(train_dataset)  # Calculate average epoch loss
    print(f"Train [{ep+1}/{epoch}], Loss: {20 * np.log10(train_loss):.4f}")

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
        val_loss += output.item() * batch_size  # Accumulate batch loss

    val_loss /= len(val_dataset)  # Calculate average epoch loss
    print(f"Val [{ep+1}/{epoch}], Loss: {20 * np.log10(val_loss):.4f}")
