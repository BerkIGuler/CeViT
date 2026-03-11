import torch
import torch.nn as nn
import torch.nn.functional as F


class CeViT(nn.Module):
    """
    CE-ViT: Channel Estimator Vision Transformer (paper Section III.A).
    Expects LS at pilots as (B, num_pilot_subcarriers, num_pilot_symbols) e.g. (B, 60, 3),
    or full grid (B, Nf, Nt) e.g. (B, 120, 14). Output is (B, Nf, Nt) complex.
    """
    def __init__(self, device, token_emb_dim, input_dim, patch_dim, model_dim, n_head, dropout,
                 num_subcarriers=120, num_symbols=14, pilot_grid_shape=(60, 3)):
        super(CeViT, self).__init__()
        self.device = device
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.pilot_grid_shape = pilot_grid_shape  # (H_pilot, W_pilot) when input is pilot-only

        self.upsampling = UpsamplingModule(
            num_subcarriers=num_subcarriers,
            num_symbols=num_symbols,
            pilot_grid_shape=pilot_grid_shape,
        ).to(device)
        self.patcher = PatchEmbedding().to(device)
        self.inverse_patcher = InversePatchEmbedding(
            output_size=(num_subcarriers, 2 * num_symbols),
            patch_size=(10, 4),
        ).to(device)
        self.tokenizer = TokenModule(input_size=1, embedding_dim=token_emb_dim).to(device)
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=patch_dim,
            d_model=model_dim,
            nhead=n_head,
            num_patches=(num_subcarriers * (2 * num_symbols)) // (10 * 4),  # 84 for 120×28
            activation="gelu",
            dropout=dropout,
        ).to(device)

    def forward(self, x):
        ls_channel, ideal_channel, meta_data = x
        ls_channel = ls_channel.to(self.device)
        ideal_channel = ideal_channel.to(self.device)
        _, snr, delay_spread, max_dop_shift, _, _ = meta_data
        snr = snr.to(self.device)
        delay_spread = delay_spread.to(self.device)
        max_dop_shift = max_dop_shift.to(self.device)

        # Paper: f_int(LS at pilots) then real/imag concat → R^{Nf×2Nt}
        real_2d = self.upsampling(ls_channel)  # (B, Nf, 2*Nt)
        ls_channel = self.patcher(real_2d)
        token_encodings = self.tokenizer(snr, delay_spread, max_dop_shift)
        model_input = torch.cat(tensors=(ls_channel, token_encodings), dim=2)

        encoder_output = self.encoder(model_input)
        real_out = self.inverse_patcher(encoder_output)  # (B, Nf, 2*Nt)
        # Paper: "splicing together the real and imaginary parts" → H in C^{Nf×Nt}
        half = real_out.size(-1) // 2
        estimated_channel = real_out[..., :half] + 1j * real_out[..., half:]
        return estimated_channel, ideal_channel


class UpsamplingModule(nn.Module):
    """
    Paper: bilinear interpolation of LS at pilots to full Nf×Nt, then concatenate
    real and imaginary along time → R^{Nf×2Nt}.
    Supports: (B, H_pilot, W_pilot) complex e.g. (B, 60, 3) or (B, Nf, Nt) complex e.g. (B, 120, 14).
    """
    def __init__(self, num_subcarriers=120, num_symbols=14, pilot_grid_shape=(60, 3)):
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.pilot_grid_shape = pilot_grid_shape

    def forward(self, ls_channel):
        # ls_channel: (B, ..., ) complex
        if ls_channel.shape[-2:] == self.pilot_grid_shape:
            # Pilot-only: (B, H_p, W_p) → interpolate to (B, Nf, Nt) then real concat
            re_im = torch.stack([ls_channel.real, ls_channel.imag], dim=1)  # (B, 2, H_p, W_p)
            re_im = F.interpolate(
                re_im,
                size=(self.num_subcarriers, self.num_symbols),
                mode="bilinear",
                align_corners=False,
            )
            real_part = re_im[:, 0]   # (B, Nf, Nt)
            imag_part = re_im[:, 1]
            return torch.cat([real_part, imag_part], dim=-1)  # (B, Nf, 2*Nt)
        else:
            # Full grid (B, Nf, Nt) complex → concat real and imag along time
            real_part = ls_channel.real
            imag_part = ls_channel.imag
            return torch.cat([real_part, imag_part], dim=-1)  # (B, Nf, 2*Nt)


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
