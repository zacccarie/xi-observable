"""C-STAB — les 2 cellules manquantes de la grille archis × tâches.

1. transformer × parité  (la famille attention sur la tâche du MLP)
2. MLP × modadd          (la famille MLP sur la tâche du transformer)
   (modadd en entrée plate one-hot 2×97 → MLP groke avec wd, Liu et al.)

Pilote auto par cellule (bande à plateau propre à chaque couple), puis
2 seeds frais, signature = (largeur de germination, taux médian) — machinerie
Γ inchangée. Complète la grille pour la décomposition de variance C-STAB.
"""
import os
import sys
import json
import time
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv.append("--v1")
import xi_gamma_germination as xg
from gamma_universality import excess_rates

RESULTS = os.path.join(HERE, "gamma_cstab_cells")
DEVICE = torch.device("cpu")
PILOT_SEED, FRESH = 5, (30, 31)
DELAYED = (1500, 11000)


# ---------------------------------------------------------------- cellule 1
N_BITS, K_PARITY = 30, 3


class TinyTransformer(nn.Module):
    """2 couches, d=64, 2 têtes — GrokGPT réduit, classification parité."""
    def __init__(self, d=64, heads=2):
        super().__init__()
        self.tok = nn.Embedding(2, d)
        self.pos = nn.Embedding(N_BITS, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                         batch_first=True)
        self.blocks = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(d, 2)
    def forward(self, x, want_act=False):
        h = self.tok(x.long()) + self.pos(torch.arange(N_BITS, device=x.device))
        h = self.blocks(h)
        a = h.mean(1)
        return (self.head(a), a) if want_act else self.head(a)


def run_parity_transformer(n_train, seed, steps=12000, every=100, wd=1e-2):
    outdir = os.path.join(xg.CKPT_ROOT, f"run_tfparityN{n_train}_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pb = rng.choice(N_BITS, size=K_PARITY, replace=False)
    mk = lambda n: (lambda X: (torch.tensor(X, dtype=torch.float32),
                               torch.tensor((X[:, pb].sum(1) % 2).astype(np.int64))))(
        rng.integers(0, 2, size=(n, N_BITS)))
    Xtr, Ytr = mk(n_train); Xte, Yte = mk(2000)
    probe = Xte[:384]
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(Xtr), Ytr).backward(); opt.step()
        if step % every == 0:
            model.eval()
            with torch.no_grad():
                tr_a = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
                te_a = (model(Xte).argmax(-1) == Yte).float().mean().item()
                _, h = model(probe, want_act=True)
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=h.cpu().numpy().astype(np.float16))
            traj.append({"step": step, "train_acc": tr_a, "val_acc": te_a})
            model.train()
            if (step > 1500 and len(traj) >= 3
                    and all(t["val_acc"] > 0.9 for t in traj[-3:])):
                break
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "arch": "transformer", "n_train": n_train,
            "t_grok": t_grok, "traj": traj, "elapsed_s": time.time() - t0}
    json.dump(meta, open(os.path.join(outdir, "traj.json"), "w"), indent=1)
    print(f"  tf-parity N={n_train} s{seed} : t_grok={t_grok} "
          f"({meta['elapsed_s']:.0f}s)", flush=True)
    return meta


# ---------------------------------------------------------------- cellule 2
P_MOD = 97


class MlpModAdd(nn.Module):
    def __init__(self, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * P_MOD, w), nn.ReLU(),
                                 nn.Linear(w, w), nn.ReLU())
        self.head = nn.Linear(w, P_MOD)
    def forward(self, x, want_act=False):
        a = self.net(x)
        return (self.head(a), a) if want_act else self.head(a)


def run_modadd_mlp(frac, seed, steps=12000, every=100, wd=1.0):
    tag = str(frac).replace(".", "p")
    outdir = os.path.join(xg.CKPT_ROOT, f"run_mlpmodF{tag}_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P_MOD) for b in range(P_MOD)])
    perm = rng.permutation(len(pairs))
    n_tr = int(frac * len(pairs))
    X = np.zeros((len(pairs), 2 * P_MOD), dtype=np.float32)
    X[np.arange(len(pairs)), pairs[:, 0]] = 1
    X[np.arange(len(pairs)), P_MOD + pairs[:, 1]] = 1
    Y = (pairs[:, 0] + pairs[:, 1]) % P_MOD
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr = torch.tensor(X[tr]); Ytr = torch.tensor(Y[tr])
    Xte = torch.tensor(X[te][:2000]); Yte = torch.tensor(Y[te][:2000])
    probe = Xte[:384]
    model = MlpModAdd().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(Xtr), Ytr).backward(); opt.step()
        if step % every == 0:
            model.eval()
            with torch.no_grad():
                tr_a = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
                te_a = (model(Xte).argmax(-1) == Yte).float().mean().item()
                _, h = model(probe, want_act=True)
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=h.cpu().numpy().astype(np.float16))
            traj.append({"step": step, "train_acc": tr_a, "val_acc": te_a})
            model.train()
            if (step > 1500 and len(traj) >= 3
                    and all(t["val_acc"] > 0.9 for t in traj[-3:])):
                break
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "arch": "mlp", "frac": frac, "t_grok": t_grok,
            "traj": traj, "elapsed_s": time.time() - t0}
    json.dump(meta, open(os.path.join(outdir, "traj.json"), "w"), indent=1)
    print(f"  mlp-modadd f={frac} s{seed} : t_grok={t_grok} "
          f"({meta['elapsed_s']:.0f}s)", flush=True)
    return meta


def signature(dirname, every):
    rd = os.path.join(xg.CKPT_ROOT, dirname)
    meta = json.load(open(os.path.join(rd, "traj.json")))
    t_grok = meta["t_grok"]
    t_mem = next((t["step"] for t in meta["traj"] if t["train_acc"] > 0.99), None)
    lo = (t_mem + 50) if t_mem else 200
    hi = (t_grok - 100) if t_grok else 20000
    e = excess_rates(rd, lo, hi, every)
    return {"dir": dirname, "t_grok": t_grok, "n_germes": int(len(e)),
            "taux_median": float(np.median(e)) if len(e) else None}


def main():
    t0 = time.time()
    sigs = []

    print("[cellule transformer × parité]", flush=True)
    chosen = None
    for n in (800, 1300, 2000, 3000, 5000):
        m = run_parity_transformer(n, PILOT_SEED)
        if m["t_grok"] and DELAYED[0] <= m["t_grok"] <= DELAYED[1]:
            chosen = n; break
    if chosen:
        for s in FRESH:
            run_parity_transformer(chosen, s)
            sigs.append({**signature(f"run_tfparityN{chosen}_s{s}", 100),
                         "cell": "transformer×parity"})
    else:
        sigs.append({"cell": "transformer×parity", "note": "pas de plateau"})

    print("[cellule MLP × modadd]", flush=True)
    chosen = None
    for f_ in (0.5, 0.6, 0.7, 0.8):
        m = run_modadd_mlp(f_, PILOT_SEED)
        if m["t_grok"] and DELAYED[0] <= m["t_grok"] <= DELAYED[1]:
            chosen = f_; break
    if chosen:
        for s in FRESH:
            run_modadd_mlp(chosen, s)
            tag = str(chosen).replace(".", "p")
            sigs.append({**signature(f"run_mlpmodF{tag}_s{s}", 100),
                         "cell": "mlp×modadd"})
    else:
        sigs.append({"cell": "mlp×modadd", "note": "pas de plateau"})

    json.dump({"signatures": sigs, "runtime_s": time.time() - t0},
              open(RESULTS + ".json", "w"), indent=1)
    print("\nCELLULES :", flush=True)
    for s in sigs:
        print(" ", s, flush=True)
    print(f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
