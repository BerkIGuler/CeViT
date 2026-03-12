import torch
import torch.nn as nn


class CeViT(nn.Module):
    """
    CE-ViT: Channel Estimator Vision Transformer (paper Section III.A).

    forward(ls_channel, meta_data) -> estimated_channel
    - ls_channel: dense (interpolated) grid (B, Nf, Nt) complex.
    - meta_data: dict from TDLDataset with keys "SNR", "delay_spread", "doppler_shift".
    - estimated_channel: (B, Nf, Nt) complex.
    """
    def __init__(
        self,
        device,
        token_emb_dim,
        input_dim,
        patch_dim,
        model_dim,
        n_head,
        dropout,
        num_subcarriers=120,
        num_symbols=14,
        patch_size=(10, 4),
        activation="gelu",
    ):
        super(CeViT, self).__init__()
        self.device = device
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.patch_size = tuple(patch_size)

        num_patches = (num_subcarriers * (2 * num_symbols)) // (self.patch_size[0] * self.patch_size[1])

        self.real_imag_concat = RealImagConcat().to(device)
        self.patcher = PatchEmbedding(patch_size=self.patch_size).to(device)
        self.inverse_patcher = InversePatchEmbedding(
            output_size=(num_subcarriers, 2 * num_symbols),
            patch_size=self.patch_size,
        ).to(device)
        self.tokenizer = TokenModule(input_size=1, embedding_dim=token_emb_dim).to(device)
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=patch_dim,
            d_model=model_dim,
            nhead=n_head,
            num_patches=num_patches,
            activation=activation,
            dropout=dropout,
        ).to(device)

    def forward(self, ls_channel, meta_data):
        # Caller must move inputs to the model device (e.g. trainer does this once per batch).
        # meta_data: dict from TDLDataset with keys "SNR", "delay_spread", "doppler_shift"
        snr = meta_data["SNR"].float().unsqueeze(1)
        delay_spread = meta_data["delay_spread"].float().unsqueeze(1)
        max_dop_shift = meta_data["doppler_shift"].float().unsqueeze(1)

        # Input is already dense (120×14); paper: real/imag concat → R^{Nf×2Nt}
        real_2d = self.real_imag_concat(ls_channel)  # (B, Nf, 2*Nt)
        ls_channel = self.patcher(real_2d)
        token_encodings = self.tokenizer(snr, delay_spread, max_dop_shift)
        model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

        encoder_output = self.encoder(model_input)
        real_out = self.inverse_patcher(encoder_output)  # (B, Nf, 2*Nt)
        # Paper: "splicing together the real and imaginary parts" → H in C^{Nf×Nt}
        half = real_out.size(-1) // 2
        estimated_channel = real_out[..., :half] + 1j * real_out[..., half:]
        return estimated_channel


class RealImagConcat(nn.Module):
    """
    Concatenate real and imaginary parts along the last dimension.
    Input: (B, Nf, Nt) complex. Output: (B, Nf, 2*Nt) real.
    (Upsampling/interpolation is done in the dataset; input is already dense.)
    """

    def forward(self, x):
        return torch.cat([x.real, x.imag], dim=-1)


class Encoder(nn.Module):
    """
    Paper: linear 46→d_model, learnable positional encoding R^{num_patches × d_model},
    then Transformer (MHA + FFN 2*d_model, GELU), then linear d_model→40.
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            d_model=128,
            nhead=4,
            num_patches=84,
            activation="gelu",
            dropout=0.1):

        super(Encoder, self).__init__()
        self.num_patches = num_patches
        self.linear_1 = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)
        # Paper: "Layer normalization is applied before every block, and residual connects after every block" → Pre-LN
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            activation=activation,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.linear_2 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.linear_2(x)
        return x


class PatchEmbedding(nn.Module):
    """Paper: partition real matrix into 10×4 patches, flatten → R^{num_patches × 40}."""
    def __init__(self, patch_size=(10, 4)):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, Nf, 2*Nt) e.g. (B, 120, 28)
        x = torch.unsqueeze(x, dim=1)
        x = self.unfold(x)
        x = torch.permute(x, dims=(0, 2, 1))
        return x


class InversePatchEmbedding(nn.Module):
    """Paper: remap 40-dim sequence back to R^{Nf×2Nt}."""
    def __init__(self, output_size=(120, 28), patch_size=(10, 4)):
        super().__init__()
        self.fold = torch.nn.Fold(
            output_size=output_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

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
