import torch
from preprocess import PatchEmbedding, InversePatchEmbedding
from dataloader import MatDataset
from torch.utils.data import DataLoader
from token_module import TokenModule
from model import Encoder


batch_size = 64
epoch = 10
model_dim = 128  # transformer linear projection dim
n_head = 4
patch_dim = 40  # patch embedding dim
input_dim = 46
dropout = 0.1
token_embedding_dim = 168
device = "cuda:0"
data_dir = "data"
dataset = MatDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
    epoch_loss = 0.0
    for batch in dataloader:
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

        epoch_loss += output.item() * batch_size  # Accumulate batch loss

    epoch_loss /= len(dataset)  # Calculate average epoch loss
    print(f"Epoch [{ep+1}/{epoch}], Loss: {epoch_loss:.4f}")
