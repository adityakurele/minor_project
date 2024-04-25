import torch

class NoiseGenerator():
    def __init__(self, latent_dim, device):
        self.latent_dim = latent_dim
        self.device = device

    def generate(self, n_samples):
        return torch.randn((n_samples, self.latent_dim), device=self.device)
