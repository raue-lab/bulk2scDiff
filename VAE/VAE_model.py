from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Sequence[int] = (1024, 1024),
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        super().__init__()
        self.network = nn.ModuleList()
        self.residual = residual

        hidden_dim = list(hidden_dim)
        if self.residual:
            assert len(set(hidden_dim)) == 1

        for idx, width in enumerate(hidden_dim):
            in_features = n_genes if idx == 0 else hidden_dim[idx - 1]
            layers = [nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.PReLU()]
            if idx == 0:
                layers.insert(0, nn.Dropout(p=input_dropout))
            else:
                layers.insert(0, nn.Dropout(p=dropout))
            self.network.append(nn.Sequential(*layers))

        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        for idx, layer in enumerate(self.network):
            if self.residual and 0 < idx < len(self.network) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def load_state(self, filename: str, use_gpu: bool = False):
        checkpoint = torch.load(
            filename,
            map_location=None if use_gpu else torch.device("cpu"),
        )
        state_dict = checkpoint["state_dict"]
        first_layer_keys = [
            "network.0.1.weight",
            "network.0.1.bias",
            "network.0.2.weight",
            "network.0.2.bias",
            "network.0.2.running_mean",
            "network.0.2.running_var",
            "network.0.2.num_batches_tracked",
            "network.0.3.weight",
        ]
        for key in first_layer_keys:
            state_dict.pop(key, None)
        self.load_state_dict(state_dict, strict=False)


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Sequence[int] = (1024, 1024),
        dropout: float = 0.5,
        residual: bool = False,
    ):
        super().__init__()
        self.network = nn.ModuleList()
        self.residual = residual

        hidden_dim = list(hidden_dim)
        if self.residual:
            assert len(set(hidden_dim)) == 1

        for idx, width in enumerate(hidden_dim):
            in_features = latent_dim if idx == 0 else hidden_dim[idx - 1]
            layers = [nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.PReLU()]
            if idx > 0:
                layers.insert(0, nn.Dropout(p=dropout))
            self.network.append(nn.Sequential(*layers))

        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for idx, layer in enumerate(self.network):
            if self.residual and 0 < idx < len(self.network) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def load_state(self, filename: str, use_gpu: bool = False):
        checkpoint = torch.load(
            filename,
            map_location=None if use_gpu else torch.device("cpu"),
        )
        state_dict = checkpoint["state_dict"]
        for key in ("network.3.weight", "network.3.bias"):
            state_dict.pop(key, None)
        self.load_state_dict(state_dict, strict=False)


class VAE(nn.Module):
    def __init__(
        self,
        num_genes,
        device="cuda",
        seed=0,
        hidden_dim=128,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        self.seed = seed
        self.latent_dim = hidden_dim
        self.hidden_dims = [1024, 1024, 1024]

        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dims,
            dropout=0.0,
            input_dropout=0.0,
            residual=False,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.latent_dim,
            hidden_dim=list(reversed(self.hidden_dims)),
            dropout=0.0,
            residual=False,
        )
        self.loss_autoencoder = nn.MSELoss(reduction="mean")
        self.optimizer_autoencoder = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=5e-4,
            weight_decay=0.01,
        )
        self.to(self.device)

    def forward(self, genes, return_latent=False, return_decoded=False):
        if return_decoded:
            return nn.ReLU()(self.decoder(genes))

        latent = self.encoder(genes)
        if return_latent:
            return latent
        return self.decoder(latent)

    def train_step(self, genes):
        genes = genes.to(self.device)
        gene_reconstructions = self.forward(genes)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)

        self.optimizer_autoencoder.zero_grad()
        reconstruction_loss.backward()
        self.optimizer_autoencoder.step()
        return {"loss_reconstruction": reconstruction_loss.item()}
