from torch.nn import Module, TransformerEncoderLayer, Linear


class Encoder(Module):
    """
    1) Linearly projects patch embeddings to transformer dimension.
    2) Passes through transformer encoder
    3) Linearly projects transformers output to final dimension
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model=128,
                 nhead=4,
                 activation="gelu",
                 dropout=0.1):
        super(Encoder, self).__init__()
        self.linear_1 = Linear(input_dim, d_model)
        self.transformer = TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=2 * d_model,
                                                   activation=activation,
                                                   dropout=dropout)
        self.linear_2 = Linear(d_model, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.transformer(x)
        x = self.linear_2(x)
        return x


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.remaining_patience = patience
        self.min_loss = None

    def early_stop(self, loss):
        return_val = False

        if self.min_loss is None:
            self.min_loss = loss
        elif loss < self.min_loss:
            self.min_loss = loss
            self.remaining_patience = self.patience
        else:
            self.remaining_patience -= 1

        if self.remaining_patience == 0:
            return_val = True

        return return_val
