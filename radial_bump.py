import itertools, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from ndlinear import NdLinear
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- helpers --------------------------------------------------
def param_count(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def flops_training(params, n, epochs):
    return 3 * params * n * epochs          # fwd + bwd

def radial_bump(X, μ=0.8, σ=0.2):
    r = torch.linalg.vector_norm(X, dim=1)
    return torch.exp(-((r-μ)**2)/(2*σ**2))

# ---------- model defs -----------------------------------------------
class DenseNet(nn.Module):
    def __init__(self, width, depth, in_dim=10):
        super().__init__()
        layers=[nn.Linear(in_dim,width), nn.ReLU()]
        if depth==3:
            layers += [nn.Linear(width,width), nn.ReLU()]
        layers.append(nn.Linear(width,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

class NdNet(nn.Module):
    def __init__(self, width, depth, in_dims=(2,5)):
        super().__init__()
        h = width//4; hid=(h,h)
        layers=[NdLinear(in_dims,hid), nn.ReLU()]
        if depth==3: layers += [NdLinear(hid,hid), nn.ReLU()]
        layers.append(NdLinear(hid,(1,1)))
        self.net=nn.Sequential(*layers); self.in_dims=in_dims; self.hid=hid
    def forward(self,x):
        b=x.size(0); return self.net(x.view(b,*self.in_dims)).view(b)

# ---------- experiment settings --------------------------------------
WIDTHS  = [16,32,64,128]
DEPTHS  = [2,3]
SIGMAS  = [0.30,0.20,0.10]   # easy → hard
SEEDS   = [0,1,2,3,4]        # 5 repeats for statistics
EPOCHS  = 4_000
LR      = 1e-3
N_TRN,N_TST,DIM = 1_200,500,10
criterion = nn.MSELoss()

# ---------- main loop -------------------------------------------------
records=[]
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr = torch.rand(N_TRN,DIM)*2-1
    X_te = torch.rand(N_TST,DIM)*2-1

    for sigma in SIGMAS:
        y_tr = radial_bump(X_tr, σ=sigma)
        y_te = radial_bump(X_te, σ=sigma)

        for width,depth in itertools.product(WIDTHS,DEPTHS):
            dense = DenseNet(width,depth)
            nd    = NdNet(width,depth)

            for kind,net in [("dense",dense),("nd",nd)]:
                params = param_count(net)
                opt=optim.Adam(net.parameters(),lr=LR)
                for _ in range(EPOCHS):
                    opt.zero_grad(set_to_none=True)
                    loss=criterion(net(X_tr),y_tr); loss.backward(); opt.step()
                with torch.no_grad():
                    records.append(dict(seed=seed,sigma=sigma,width=width,
                                        depth=depth,kind=kind,
                                        params=params,
                                        test_MSE=criterion(net(X_te),y_te).item(),
                                        FLOPs=flops_training(params,N_TRN,EPOCHS)))

df = pd.DataFrame(records)

# ---------- aggregate mean ± std -------------------------------------
summary = (df.groupby(["sigma","width","kind","depth"])
             .agg(MSE_mean=("test_MSE","mean"),
                  MSE_std =("test_MSE","std"),
                  params  =("params","first"),
                  FLOPs   =("FLOPs","first"))
             .reset_index())

pd.set_option("display.float_format","{:,.6g}".format)
print("\nEldan–Shamir radial bump – 5 seeds")
print(summary.set_index(["sigma","width","kind","depth"]))

# ---------- plots -----------------------------------------------------
parser=argparse.ArgumentParser(); parser.add_argument('--no_plots',action='store_true')
args=parser.parse_args()
if not args.no_plots:
    styles={('dense',2):'o-',
            ('dense',3):'o--',
            ('nd',2):'s-',
            ('nd',3):'s--'}
    for sigma in SIGMAS:
        plt.figure(figsize=(6.5,4))
        sub=summary[summary['sigma']==sigma]
        for (kind,depth),g in sub.groupby(['kind','depth']):
            plt.errorbar(g['width'], g['MSE_mean'], yerr=g['MSE_std'],
                         fmt=styles[(kind,depth)], capsize=3,
                         label=f'{kind} depth={depth}')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Hidden width w'); plt.ylabel('Test MSE')
        plt.title(f'Radial bump  σ={sigma}')
        plt.grid(True, which='both', ls=':')
        plt.legend()
    plt.show()

