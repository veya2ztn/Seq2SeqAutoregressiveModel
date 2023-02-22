import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np



class Codebook(nn.Module):
    """
    Codebook mapping: takes in an encoded image and maps each vector onto its closest codebook vector.
    Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super().__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        B, _, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()  # moving average instead of hard codebook remapping

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        min_encoding_indices = min_encoding_indices.reshape(B, H, W)

        return z_q, min_encoding_indices, loss

    def dictionary_lookup(self, index):
        embeddings = F.embedding(index, self.embedding.weight).permute(0, 3, 1, 2)  #(B,C,H,W)
        return embeddings


class Codebook_EAMupdate(nn.Module):
    def __init__(self, n_codes, embedding_dim, beta=0.25, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.training = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = z.permute(0, 2, 3, 1).contiguous().flatten(end_dim=-2)

        # flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
        # print("_init_embeddings")

    def forward(self, z):
        # z: [b, c, h, w]

        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = z.permute(0, 2, 3, 1).contiguous().flatten(end_dim=-2)
        # flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # [bhw, c]

        # print("flat_inputs", flat_inputs.shape)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bhw, c]
        # print("distances", distances.shape)
        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs) # [bhw, ncode]
        # print("encode_onehot", torch.sum(encode_onehot))
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:]) # [b, h, w, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings) # [b, h, w, c]
        # embeddings = shift_dim(embeddings, -1, 1) # [b, c,  h, w]
        embeddings = embeddings.permute(0, 3, 1, 2)
        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        # print('avg_probs', avg_probs)

        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))
        # print('perplexity', perplexity)
        return embeddings_st, encoding_indices, commitment_loss
        # return dict(embeddings=embeddings_st, encodings=encoding_indices,
        #             commitment_loss=commitment_loss, perplexity=perplexity)
    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings.weight)
        return embeddings
