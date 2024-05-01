from torch.nn import Module, TransformerEncoderLayer, Linear


class Encoder(Module):
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
