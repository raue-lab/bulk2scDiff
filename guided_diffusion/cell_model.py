import torch
import torch.nn as nn

from .nn import linear, timestep_embedding


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        return self.time_embed(timestep_embedding(t, self.hidden_dim).squeeze(1))


class PseudobulkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pseudobulk):
        return self.network(pseudobulk.float())


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        time_features,
        dropout=0.0,
        cond_features=0,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                time_features,
                out_features,
            ),
        )
        self.film_layer = None
        if cond_features > 0:
            self.film_layer = nn.Sequential(
                nn.SiLU(),
                linear(cond_features, out_features * 2),
            )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, emb, cond_emb=None):
        h = self.fc(x)
        h = h + self.emb_layer(emb)
        h = self.norm(h)
        if self.film_layer is not None and cond_emb is not None:
            gamma, beta = torch.chunk(self.film_layer(cond_emb), 2, dim=-1)
            h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.drop(h)
        return h


class Cell_Unet(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_num=(2000, 1000, 500, 500),
        dropout=0.1,
        cond_pseudobulk=False,
        pseudobulk_dim=0,
        cond_embed_dim=256,
        pseudobulk_hidden_dim=512,
    ):
        super().__init__()
        hidden_num = list(hidden_num)
        self.hidden_num = hidden_num
        self.cond_pseudobulk = cond_pseudobulk

        self.time_embedding = TimeEmbedding(hidden_num[0])

        cond_features = 0
        self.pseudobulk_encoder = None
        self.cond_to_time = None
        if self.cond_pseudobulk:
            if pseudobulk_dim <= 0:
                raise ValueError("pseudobulk_dim must be positive when cond_pseudobulk=True")
            cond_features = cond_embed_dim
            self.pseudobulk_encoder = PseudobulkEncoder(
                input_dim=pseudobulk_dim,
                hidden_dim=pseudobulk_hidden_dim,
                output_dim=cond_embed_dim,
            )
            self.cond_to_time = nn.Sequential(
                nn.SiLU(),
                linear(cond_embed_dim, hidden_num[0]),
            )

        self.layers = nn.ModuleList()
        self.layers.append(
            ResidualBlock(
                input_dim,
                hidden_num[0],
                hidden_num[0],
                dropout=dropout,
                cond_features=cond_features,
            )
        )
        for i in range(len(hidden_num) - 1):
            self.layers.append(
                ResidualBlock(
                    hidden_num[i],
                    hidden_num[i + 1],
                    hidden_num[0],
                    dropout=dropout,
                    cond_features=cond_features,
                )
            )

        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_num) - 1)):
            self.reverse_layers.append(
                ResidualBlock(
                    hidden_num[i + 1],
                    hidden_num[i],
                    hidden_num[0],
                    dropout=dropout,
                    cond_features=cond_features,
                )
            )

        self.out1 = nn.Linear(hidden_num[0], int(hidden_num[1] * 2))
        self.norm_out = nn.LayerNorm(int(hidden_num[1] * 2))
        self.out2 = nn.Linear(int(hidden_num[1] * 2), input_dim, bias=True)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def encode_condition(self, pseudobulk):
        if not self.cond_pseudobulk:
            raise ValueError("pseudobulk conditioning is disabled for this model")
        if pseudobulk is None:
            raise ValueError("pseudobulk conditioning enabled but no pseudobulk input was provided")
        return self.pseudobulk_encoder(pseudobulk.float())

    def forward(self, x_input, t, pseudobulk=None, **unused_kwargs):
        emb = self.time_embedding(t)
        cond_emb = None
        if self.cond_pseudobulk:
            cond_emb = self.encode_condition(pseudobulk)
            emb = emb + self.cond_to_time(cond_emb)

        x = x_input.float()

        history = []
        for layer in self.layers:
            x = layer(x, emb, cond_emb=cond_emb)
            history.append(x)

        history.pop()
        for layer in self.reverse_layers:
            x = layer(x, emb, cond_emb=cond_emb)
            x = x + history.pop()

        x = self.out1(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.out2(x)
        return x
