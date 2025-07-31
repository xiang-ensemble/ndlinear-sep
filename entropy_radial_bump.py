#!/usr/bin/env python3
# Compare Dense vs NdLinear networks on a radial‑bump regression task,
# tracking feature entropy layer‑by‑layer.
# ---------------------------------------------------------------------
import argparse, itertools, math, warnings, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------- Data --------------------------------------------------
def radial_bump(x, mu=0.8, sigma=0.2):
    """y = exp(-(‖x‖-μ)^2 / (2σ²))"""
    r = torch.linalg.vector_norm(x, dim=1)
    return torch.exp(-((r - mu) ** 2) / (2 * sigma ** 2))

# --------------------- Entropy utils ----------------------------------------
def _von_neumann_entropy(cov: torch.Tensor, eps=1e-12):
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals.real, min=eps)
    p = eigvals / eigvals.sum()
    return -(p * torch.log(p)).sum().item()

def tensor_entropy(t: torch.Tensor, eps=1e-9):
    """von‑Neumann entropy, normalised to [0,1]."""
    z = t.reshape(t.shape[0], -1).float()
    z = (z - z.mean(0, keepdim=True))
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

    N, D = z.shape
    cov = (z.T @ z) if N > D else (z @ z.T)
    cov = cov / cov.trace()
    ent = _von_neumann_entropy(cov.double())

    denom = max(eps, min(math.log(max(2, N)), math.log(max(2, D))))
    return ent / denom              # ∈[0,1]

# --------------------- NdLinear ---------------------------------------------
class NdLinear(nn.Module):
    """
    Dense layer that maps (..., *in_dims) → (..., *out_dims)
    by flattening the feature axes, applying a Linear, then
    reshaping back.
    """
    def __init__(self, in_dims, out_dims, bias=True):
        super().__init__()
        self.in_dims  = tuple(in_dims)
        self.out_dims = tuple(out_dims)

        fan_in  = int(np.prod(self.in_dims))
        fan_out = int(np.prod(self.out_dims))
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias   = nn.Parameter(torch.empty(fan_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_dims = x.shape[:-len(self.in_dims)]     # <<< FIXED
        y = x.reshape(*batch_dims, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*batch_dims, *self.out_dims)

# --------------------- Network definitions ----------------------------------
class WithEntropy(nn.Module):
    """Mixin: stores layer activations & entropy automatically."""
    def _register_hooks(self):
        self._entropies = {}
        for idx, m in enumerate(self.modules()):
            if isinstance(m, (nn.Linear, NdLinear, nn.ReLU)):
                m.register_forward_hook(self._make_hook(idx, m))

    def _make_hook(self, idx, mod):
        def hook(_, __, out):
            self._entropies[f"{mod.__class__.__name__}_{idx}"] = tensor_entropy(out)
        return hook

    def get_entropies(self):
        return self._entropies

class DenseNet(WithEntropy):
    def __init__(self, width, depth, in_dim=10):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self._register_hooks()

    def forward(self, x):                       # (B, in_dim)
        return self.net(x).squeeze(-1)          # (B,)

class NdNet(WithEntropy):
    def __init__(self, width, depth, in_dims=(2,5)):
        super().__init__()
        hid = (width // len(in_dims),) * len(in_dims)
        layers = [NdLinear(in_dims, hid), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [NdLinear(hid, hid), nn.ReLU()]
        layers.append(NdLinear(hid, (1,1)))
        self.net = nn.Sequential(*layers)
        self.in_dims = in_dims
        self._register_hooks()

    def forward(self, x):                       # (B, prod(in_dims))
        b = x.size(0)
        x = x.view(b, *self.in_dims)
        y = self.net(x)                         # (B,1,1)
        return y.view(b)                        # (B,)

# --------------------- Experiment runner ------------------------------------
def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xtr = torch.rand(cfg.n_train, cfg.dim, device=device)*2 - 1
    Xte = torch.rand(cfg.n_test , cfg.dim, device=device)*2 - 1
    ytr = radial_bump(Xtr, sigma=cfg.sigma)
    yte = radial_bump(Xte, sigma=cfg.sigma)

    results = []
    grid = itertools.product(cfg.widths, cfg.depths, ["dense","nd"])
    for width, depth, arch in grid:
        model = (DenseNet(width, depth) if arch=="dense"
                 else NdNet(width, depth)).to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        for _ in tqdm(range(cfg.epochs), desc=f"{arch}-{width}x{depth}", leave=False):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr), ytr)
            loss.backward()
            opt.step()
            if loss.item() < 1e-4: break        # quick escape

        with torch.no_grad():
            test_mse = loss_fn(model(Xte), yte).item()
        ents = model.get_entropies()
        results.append(dict(kind=arch, width=width, depth=depth,
                            test_mse=test_mse,
                            avg_entropy=float(np.mean(list(ents.values())))))

    return pd.DataFrame(results)

# --------------------- CLI ---------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    cfg = argparse.Namespace(**vars(p.parse_args()) | dict(
        widths  = [16,32]       ,
        depths  = [2,3]         ,
        sigma   = 0.3           ,
        dim     = 10            ,
        n_train = 1200           ,
        n_test  = 500           ,
        lr      = 1e-3          ,
        epochs  = 4000 if not p.parse_args().quick else 120,
        seed    = 0
    ))

    df = run(cfg)
    print("\n=== RESULTS (σ=%.2f) ===" % cfg.sigma)
    print(df)

    if not cfg.no_plots:
        for arch in df.kind.unique():
            sub = df[df.kind==arch]
            plt.scatter(sub.avg_entropy, sub.test_mse, label=arch, s=90)
        plt.gca().invert_yaxis()
        plt.xlabel("Average layer entropy (↑)")
        plt.ylabel("Test MSE (↓)")
        plt.title("Dense vs NdLinear")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
