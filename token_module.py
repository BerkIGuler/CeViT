import torch
from torch import nn


class LinearEncoder(nn.Module):
    """linear projection from 1D to N-D space"""
    def __init__(self, input_size, output_size):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class TokenModule(nn.Module):
    """Linearly projects snr, delay spread, and max. doppler shift
    to obtain token embeddings then concatenates those
    """
    def __init__(self, input_size, embedding_dim):
        super(TokenModule, self).__init__()
        self.snr_encoder = LinearEncoder(input_size, embedding_dim)
        self.ds_encoder = LinearEncoder(input_size, embedding_dim)
        self.dop_encoder = LinearEncoder(input_size, embedding_dim)

    def forward(self, snr, delay_spread, doppler_shift):
        bs = snr.shape[0]  # batch size
        snr_emb = torch.reshape(self.snr_encoder(snr), (bs, -1, 2))
        ds_emb = torch.reshape(self.ds_encoder(delay_spread), (bs, -1, 2))
        dop_emb = torch.reshape(self.dop_encoder(doppler_shift), (bs, -1, 2))
        token_emb = torch.cat((snr_emb, ds_emb, dop_emb), dim=2)
        return token_emb


if __name__ == "__main__":
    SNR = torch.rand(16, 1)
    DS = torch.rand(16, 1)
    DOP = torch.rand(16, 1)
    tokenizer = TokenModule(input_size=1, embedding_dim=168)
    token_encodings = tokenizer(SNR, DS, DOP)
    print(token_encodings.shape)
