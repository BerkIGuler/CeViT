import torch
import torch.nn as nn


class CeViT(nn.Module):
    def __init__(self, device, token_emb_dim, input_dim, patch_dim, model_dim, n_head, dropout):
        super(CeViT, self).__init__()
        self.device = device
        self.patcher = PatchEmbedding().to(device)
        self.inverse_patcher = InversePatchEmbedding().to(device)
        self.tokenizer = TokenModule(input_size=1, embedding_dim=token_emb_dim).to(device)
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=patch_dim,
            d_model=model_dim,
            nhead=n_head,
            activation="gelu",
            dropout=dropout).to(device)

    def forward(self, x):
        ls_channel, ideal_channel, meta_data = x
        ls_channel, ideal_channel = ls_channel.to(self.device), ideal_channel.to(self.device)
        _, snr, delay_spread, max_dop_shift, _, _ = meta_data
        snr, delay_spread, max_dop_shift = (snr.to(self.device),
                                            delay_spread.to(self.device),
                                            max_dop_shift.to(self.device))

        ls_channel = self.patcher(ls_channel)
        token_encodings = self.tokenizer(snr, delay_spread, max_dop_shift)
        model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

        encoder_output = self.encoder(model_input)
        estimated_channel = self.inverse_patcher(encoder_output)
        return estimated_channel, ideal_channel


class Encoder(nn.Module):
    """
    1) Linearly projects patch embeddings to transformer dimension.
    2) Passes through transformer encoder
    3) Linearly projects transformers output to final dimension
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            d_model=128,
            nhead=4,
            activation="gelu",
            dropout=0.1):

        super(Encoder, self).__init__()
        self.linear_1 = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            activation=activation,
            dropout=dropout)
        self.linear_2 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.transformer(x)
        x = self.linear_2(x)
        return x


class PatchEmbedding(nn.Module):
    """Reorganizes channel matrix using the same idea from ViT"""
    def __init__(self, patch_size=(10, 4)):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.unfold(torch.unsqueeze(x, dim=1))
        x = torch.permute(x, dims=(0, 2, 1))
        return x


class InversePatchEmbedding(nn.Module):
    """Recovers matrix from patch embeddings"""
    def __init__(self, output_size=(120, 28), patch_size=(10, 4)):
        super().__init__()
        self.fold = torch.nn.Fold(
            output_size=output_size,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        x = torch.permute(x, dims=(0, 2, 1))
        x = self.fold(x)
        x = torch.squeeze(x, dim=1)
        return x


class LinearProjection(nn.Module):
    """linear projection from 1D to N-D space"""
    def __init__(self, input_size, output_size):
        super(LinearProjection, self).__init__()
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
        self.snr_encoder = LinearProjection(input_size, embedding_dim)
        self.ds_encoder = LinearProjection(input_size, embedding_dim)
        self.dop_encoder = LinearProjection(input_size, embedding_dim)

    def forward(self, snr, delay_spread, doppler_shift):
        bs = snr.shape[0]  # batch size
        snr_emb = torch.reshape(self.snr_encoder(snr), (bs, -1, 2))
        ds_emb = torch.reshape(self.ds_encoder(delay_spread), (bs, -1, 2))
        dop_emb = torch.reshape(self.dop_encoder(doppler_shift), (bs, -1, 2))
        token_emb = torch.cat((snr_emb, ds_emb, dop_emb), dim=2)
        return token_emb
