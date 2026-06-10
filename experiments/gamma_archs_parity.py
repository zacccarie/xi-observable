"""Γ-signatures de conv1d / deep-sets / ode-lite — MÊME testbed que le MLP.

Remplit la colonne « germination » de la table (THEORY_ARCHITECTURAL_CLASSES).
Testbed identique L4 : (30,3)-sparse parity, wd=1e-2, plateau-band à calibrer
PAR architecture (le seuil d'échantillons dépend du biais inductif — leçon
L4 v1). Signature = (largeur de germination = n_germes/run ;
taux médian d'excès ; t_grok). Référence MLP : aiguille 3-12 germes.

PILOTE par arch (1 seed, hors preuve) : N ∈ {800, 1300, 2000, 3000, 5000}
jusqu'à delayed-grok (1500 ≤ t_grok ≤ 18000). Puis 2 seeds frais à ce N,
Γ machinery inchangée (ranks 4-32, fenêtre 6, θ q95 null, Frobenius).
Si une arch ne déléye jamais → signature « pas de plateau mesurable sur ce
testbed » (info en soi).
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv.append("--v1")
import xi_gamma_germination as xg
from gamma_universality import excess_rates

RESULTS = os.path.join(HERE, "gamma_archs_parity")
DEVICE = torch.device("cpu")
N_BITS, K_PARITY = 30, 3
WD, STEPS, EVERY = 1e-2, 20000, 100
PILOT_SEED, FRESH = 5, (30, 31)
DELAYED = (1500, 18000)


class Conv1dNet(nn.Module):
    """Conv1d sur la séquence de 30 bits — CG local."""
    def __init__(self, w=64):
        super().__init__()
        self.c1 = nn.Conv1d(1, w, 5, padding=2)
        self.c2 = nn.Conv1d(w, w, 5, padding=2)
        self.head = nn.Linear(w, 2)
    def forward(self, x, want_act=False):
        h = F.relu(self.c1(x.unsqueeze(1)))
        h = F.relu(self.c2(h))
        a = h.mean(-1)                                   # pool global
        return (self.head(a), a) if want_act else self.head(a)


class DeepSetsNet(nn.Module):
    """φ par (position one-hot-lite, bit) puis somme — CG permutationnel
    sur les paires (index, valeur)."""
    def __init__(self, w=96):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, w), nn.ReLU(),
                                 nn.Linear(w, w), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(w, w), nn.ReLU())
        self.head = nn.Linear(w, 2)
        pos = torch.arange(N_BITS, dtype=torch.float32) / N_BITS
        self.register_buffer("pos", pos)
    def forward(self, x, want_act=False):
        toks = torch.stack([self.pos.expand_as(x), x], -1)   # (B, 30, 2)
        s = self.phi(toks).sum(1)
        a = self.rho(s)
        return (self.head(a), a) if want_act else self.head(a)


class OdeLiteNet(nn.Module):
    """Intégrateur d'Euler sur la séquence de bits — CG temporel intégratif."""
    def __init__(self, w=72, dt=0.25):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(w + 1, w), nn.Tanh())
        self.head = nn.Linear(w, 2)
        self.w, self.dt = w, dt
    def forward(self, x, want_act=False):
        h = torch.zeros(x.shape[0], self.w, device=x.device)
        for t in range(N_BITS):
            h = h + self.dt * self.f(torch.cat([h, x[:, t:t+1]], -1))
        return (self.head(h), h) if want_act else self.head(h)


ARCHS = {"conv1d": Conv1dNet, "deepsets": DeepSetsNet, "odelite": OdeLiteNet}


def run_arch(arch, n_train, seed):
    outdir = os.path.join(xg.CKPT_ROOT, f"run_{arch}N{n_train}_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parity_bits = rng.choice(N_BITS, size=K_PARITY, replace=False)

    def make(n):
        X = rng.integers(0, 2, size=(n, N_BITS)).astype(np.float32)
        Y = (X[:, parity_bits].sum(axis=1) % 2).astype(np.int64)
        return torch.tensor(X), torch.tensor(Y)

    Xtr, Ytr = make(n_train)
    Xte, Yte = make(2000)
    probe = Xte[:384]
    model = ARCHS[arch]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD)

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def pacts():
        model.eval()
        _, h = model(probe, want_act=True)
        model.train()
        return h.cpu().numpy().astype(np.float16)

    traj = []
    t0 = time.time()
    for step in range(STEPS + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=pacts())
            traj.append({"step": step, "train_acc": acc(Xtr, Ytr),
                         "val_acc": acc(Xte, Yte)})
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "arch": arch, "n_train": n_train, "t_grok": t_grok,
            "traj": traj, "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, "traj.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  {arch:9s} N={n_train} s{seed} : t_grok={t_grok} "
          f"({meta['elapsed_s']:.0f}s)")
    return meta


def signature(arch, n_train, seed):
    rd = os.path.join(xg.CKPT_ROOT, f"run_{arch}N{n_train}_s{seed}")
    meta = json.load(open(os.path.join(rd, "traj.json")))
    t_grok = meta["t_grok"]
    t_mem = next((t["step"] for t in meta["traj"] if t["train_acc"] > 0.99), None)
    lo = (t_mem + 50) if t_mem else 200
    hi = (t_grok - 100) if t_grok else STEPS
    e = excess_rates(rd, lo, hi, EVERY)
    return {"arch": arch, "n_train": n_train, "seed": seed, "t_grok": t_grok,
            "t_mem": t_mem, "n_germes": int(len(e)),
            "taux_median": float(np.median(e)) if len(e) else None,
            "tau": hi - lo}


def main():
    t0 = time.time()
    sigs, pilots = [], {}
    for arch in ARCHS:
        print(f"[pilote {arch}]")
        chosen = None
        for n in (800, 1300, 2000, 3000, 5000):
            m = run_arch(arch, n, PILOT_SEED)
            if m["t_grok"] and DELAYED[0] <= m["t_grok"] <= DELAYED[1]:
                chosen = n
                break
        pilots[arch] = chosen
        if chosen is None:
            print(f"  {arch} : PAS de plateau mesurable (instant ou jamais)")
            continue
        print(f"[frais {arch} N={chosen}]")
        for seed in FRESH:
            run_arch(arch, chosen, seed)
            sigs.append(signature(arch, chosen, seed))
            s = sigs[-1]
            print(f"    s{seed}: germes={s['n_germes']} "
                  f"taux_méd={s['taux_median']} grok={s['t_grok']}")

    out = {"pilot_N": pilots, "signatures": sigs,
           "reference_mlp": {"n_germes": "3-12 (aiguille)",
                             "source": "gamma_universality 7 seeds"},
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nVERDICT — colonnes germination :")
    for arch in ARCHS:
        ss = [s for s in sigs if s["arch"] == arch]
        if ss:
            print(f"  {arch:9s}: germes={[s['n_germes'] for s in ss]} "
                  f"(N={pilots[arch]})")
        else:
            print(f"  {arch:9s}: pas de plateau sur ce testbed")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
