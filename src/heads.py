"""Decision heads beyond the flat softmax (experiments B6, B7). All keep q = softmax(z) exactly normalized.

NBHead      q(c) from the flat head plus a per-gene inverse dispersion theta_g(c) = exp(b_g + V h); trained with the
            negative-binomial likelihood of the raw panel counts, mean mu_g = N q_g (N = the cell's panel library size).
LowRankHead z = W h + log b + U diag(a(h)) eps, eps ~ N(0, I_k): a logistic-normal distribution on the simplex whose logit
            covariance U diag(a)^2 U^T is low rank (k gene programs). Trained as a VAE: an encoder of the target cell gives
            q(eps | x) = N(m, diag(s^2)); loss = -E_q[log Mult(x | softmax(z))] + beta * KL(q || N(0, I)), one reparameterized
            sample. Generation draws eps from the prior, so one forward pass of the language model still gives the decoder.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def nb_nll(x, mu, theta):
    """-log NB(x | mu, theta) per entry; variance mu + mu^2 / theta."""
    eps = 1e-8
    return -(torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
             + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
             + x * (torch.log(mu + eps) - torch.log(theta + mu + eps)))


class NBHead(nn.Module):
    def __init__(self, hidden, G, log_theta_init=0.0):
        super().__init__()
        self.theta = nn.Linear(hidden, G)
        nn.init.zeros_(self.theta.weight)
        nn.init.constant_(self.theta.bias, log_theta_init)

    def log_theta(self, h):
        return self.theta(h).clamp(-6.0, 8.0)

    def loss(self, logits, h, X):
        """X raw counts [B, G]; returns NB NLL summed over genes per cell [B]."""
        q = torch.softmax(logits, -1)
        N = X.sum(1, keepdim=True)
        return nb_nll(X, N * q, torch.exp(self.log_theta(h))).sum(1)


class LowRankHead(nn.Module):
    def __init__(self, hidden, G, k=16, enc_hidden=256):
        super().__init__()
        self.k = k
        self.U = nn.Parameter(torch.randn(G, k) * 0.01)
        self.amp = nn.Linear(hidden, k)
        nn.init.zeros_(self.amp.weight)
        nn.init.constant_(self.amp.bias, math.log(math.e - 1.0))      # softplus(bias) = 1 at init
        self.enc = nn.Sequential(nn.Linear(G, enc_hidden), nn.GELU(), nn.Linear(enc_hidden, 2 * k))

    def scale(self, h):
        return F.softplus(self.amp(h))                                  # [B, k]

    def offset(self, h, e):
        return (self.U @ (self.scale(h) * e).T).T                      # [B, G]

    def loss(self, logits, h, X, beta=1.0):
        """VAE loss per cell [B] (nats, summed over UMIs) and diagnostics."""
        L = torch.log1p(1e4 * X / X.sum(1, keepdim=True).clamp(min=1))
        m, logs = self.enc(L).chunk(2, -1)
        logs = logs.clamp(-6, 3)
        e = m + torch.exp(logs) * torch.randn_like(m)
        z = logits + self.offset(h, e)
        rec = -(X * torch.log_softmax(z, -1)).sum(1)
        kl = 0.5 * (m ** 2 + torch.exp(2 * logs) - 1 - 2 * logs).sum(1)
        return rec + beta * kl, {"rec": float(rec.mean()), "kl": float(kl.mean()), "U_norm": float(self.U.norm())}

    @torch.no_grad()
    def sample_logits(self, logits, h, n, generator=None):
        """n prior draws per row of logits: returns [rows * n, G] in row-major order."""
        e = torch.randn(logits.shape[0], n, self.k, device=logits.device, generator=generator)
        off = torch.einsum("gk,bnk->bng", self.U, self.scale(h)[:, None, :] * e)
        return (logits[:, None, :] + off).reshape(-1, logits.shape[-1])
