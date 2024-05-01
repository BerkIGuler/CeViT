import torch
from preprocess import (PatchEmbedding,
                        InversePatchEmbedding,
                        inverse_concat_complex_channel)
from dataloader import MatDataset
from torch.utils.data import DataLoader
from token_module import TokenModule
from model import Encoder


batch_size = 32
model_dim = 128
n_head = 4
patch_dim = 40
data_dir = 'data'
dataset = MatDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
patchifier = PatchEmbedding()
inverse_patchifier = InversePatchEmbedding()

# Usage example:
for batch in dataloader:
    batch_X, batch_Y, meta_data = batch
    file_no, SNR, delay_spread, max_dop_shift, ch_type = meta_data
    print(batch_X.shape, batch_Y.shape)
    batch_X = patchifier(batch_X)
    tokenizer = TokenModule(input_size=1, embedding_dim=168)
    token_encodings = tokenizer(SNR, delay_spread, max_dop_shift)
    print(batch_X.shape, token_encodings.shape)
    model_input = torch.cat((batch_X, token_encodings), dim=2)
    print(model_input.shape)

    input_dim = model_input.shape[-1]
    encoder = Encoder(
        input_dim=input_dim,
        output_dim=patch_dim,
        d_model=model_dim,
        nhead=n_head,
        activation="gelu",
        dropout=0.1)
    output = encoder(model_input)
    print(output.shape)
    output = inverse_patchifier(output)
    print(output.shape)
    result = inverse_concat_complex_channel(output)
    print(result.shape)
    break