#!/usr/bin/env python3
"""
ndlinear_bias_experiment.py  ░  Quick benchmark for NdLinear networks with and without bias
-------------------------------------------------------------------------
• Compares test‑MSE and parameter counts for NdLinear models that include a
  bias term versus those that do not.
• Uses a tiny radial‑bump regression task (10‑D → scalar) so the run finishes
  in under 3 minutes on a laptop CPU.


Example
-------
    python ndlinear_bias_experiment.py \
            --widths 16 64 128 \
            --seeds 0 1 2 \
            --epochs 800

This prints a Markdown table summarising mean / std test‑MSE and trainable
parameter counts for each width × bias setting.
"""
import argparse, inspect, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from ndlinear import NdLinear

# ---------------------------------------------------------------------------
#  Back‑compat helper ─ make NdLinear bias‑aware even if the installed
#  version predates the "bias" keyword (older than v0.4.0‑alpha)
# ---------------------------------------------------------------------------

def make_ndlayer(in_dims, out_dims, *, bias: bool):
    """Instantiate NdLinear, falling back when the 'bias' kwarg is missing."""
    if 'bias' in inspect.signature(NdLinear).parameters:
        return NdLinear(in_dims, out_dims, bias=bias)
    # Fallback: old NdLinear has no bias parameter → emulate w/out bias only
    if bias:
        raise RuntimeError(
            "Installed NdLinear version lacks the 'bias' option.\n"
            "Upgrade with  `pip install --upgrade ndlinear`  or run the test with bias=False." )
    return NdLinear(in_dims, out_dims)

# ---------------------------------------------------------------------------
#  Synthetic regression target
# ---------------------------------------------------------------------------

def radial_bump(x: torch.Tensor, μ: float = 0.8, σ: float = 0.2) -> torch.Tensor:
    """Radially symmetric bump‑function on the d‑sphere."""
    r = torch.linalg.vector_norm(x, dim=1)
    return torch.exp(-((r - μ) ** 2) / (2 * σ ** 2))


# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def param_count(module: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class NdNet(nn.Module):
    """Two‑layer NdLinear network (optionally 3) with optional bias."""

    def __init__(self, width: int, depth: int = 2, *, bias: bool = False):
        super().__init__()
        in_dims = (2, 5)                   # 10‑dim input reshaped to 2×5
        hid     = (width // 4, width // 4) # balanced hidden factors

        layers = [make_ndlayer(in_dims, hid, bias=bias), nn.ReLU()]
        if depth == 3:
            layers += [make_ndlayer(hid, hid, bias=bias), nn.ReLU()]
        layers.append(make_ndlayer(hid, (1, 1), bias=bias))
        self.net = nn.Sequential(*layers)
        self.in_dims = in_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 10) → (B,)
        b = x.shape[0]
        return self.net(x.view(b, *self.in_dims)).view(b)


# ---------------------------------------------------------------------------
#  Single training / evaluation run
# ---------------------------------------------------------------------------

def run_once(width: int, bias: bool, seed: int, *, epochs: int, lr: float,
             n_trn: int, n_tst: int, dim: int, sigma: float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    Xtr = torch.rand(n_trn, dim) * 2 - 1
    Xte = torch.rand(n_tst, dim) * 2 - 1
    ytr = radial_bump(Xtr, σ=sigma)
    yte = radial_bump(Xte, σ=sigma)

    net   = NdNet(width, depth=2, bias=bias)
    opt   = optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = lossf(net(Xtr), ytr); loss.backward(); opt.step()

    with torch.no_grad():
        test_mse = lossf(net(Xte), yte).item()

    return {
        "width": width,
        "bias": bias,
        "seed": seed,
        "params": param_count(net),
        "test_mse": test_mse,
    }


# ---------------------------------------------------------------------------
#  Command‑line interface
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="NdLinear bias vs no‑bias quick experiment")
    p.add_argument("--widths", type=int, nargs="+", default=[16, 64],
                   help="hidden widths to test")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help="random seeds")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_trn", type=int, default=800)
    p.add_argument("--n_tst", type=int, default=200)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.2)
    args = p.parse_args()

    records = [
        run_once(w, b, s, epochs=args.epochs, lr=args.lr,
                  n_trn=args.n_trn, n_tst=args.n_tst, dim=args.dim, sigma=args.sigma)
        for w in args.widths for b in (True, False) for s in args.seeds
    ]

    df = pd.DataFrame(records)
    summary = (df.groupby(["width", "bias"])         # pivot
                 .agg(MSE_mean=("test_mse", "mean"),
                      MSE_std =("test_mse", "std"),
                      params  =("params",   "first"))
                 .reset_index())
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("\n[ERROR]", e)
