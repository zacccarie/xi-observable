"""Γ L3 — ABLATION CAUSALE : couper le germe doit retarder le grok.

L1 (existence, v2 fresh) et L2 (dose-réponse, v3 frac) acquis. Ici L3 :
si les directions Γ-positives sont LE germe du circuit (cause) et pas un
symptôme, les projeter hors des activations pendant le plateau doit retarder
la généralisation. Si le grok arrive à l'heure malgré l'ablation → directions
épiphénoménales → Γ retombe à corrélat.

MÉCANISME (online, adaptatif — « chasse au germe ») :
  - tous les 50 steps : probe test (512) à travers le réseau ABLATÉ, buffer
    des 6 derniers probes (centrés, Frobenius-normalisés — identique Γ v1).
  - refresh U : SVD de l'ancre (plus vieux du buffer), candidates rangs 4..32,
    pentes par direction figée sur le buffer, θ = q95 de directions aléatoires
    (null), germe = candidates pente > θ, top K_ABL=8 par pente.
  - forward : x_eq ← x_eq − U Uᵀ x_eq (avant LayerNorm final + head),
    pendant TRAIN et EVAL — le réseau ne peut pas utiliser ces directions.

CONDITIONS (frac=0.5, wd=1.0, seeds 20-21, steps=4500) :
  germ     : ablate les directions Γ-positives (k(t) adaptatif ≤ 8)
  random   : ablate k(t) directions aléatoires (même cadence, k matché au
             k détecté par le détecteur du run lui-même)
  lowslope : ablate k(t) directions du MÊME pool (rangs 4..32) mais aux
             pentes les PLUS BASSES (non-croissantes)
  baseline : t_grok v3 sans ablation (s20: 1250, s21: 1150 — mêmes seeds/data)

PRÉ-ENREGISTREMENT :
  Δ_cond(seed) = t_grok(cond) − t_grok(baseline). t_grok=None → Δ censuré
  à 4500−baseline (borne basse).
  H-L3  : Δ_germ ≥ Δ_random + 300 ET Δ_germ ≥ Δ_lowslope + 300, dans 2/2 seeds.
  KILL  : Δ_germ ≤ max(Δ_random, Δ_lowslope) + 100 dans ≥ 1/2 seeds
          → épiphénomène (ou re-routage plus rapide que la cadence de chasse).
  Entre les deux : causalité faible (whack-a-mole partiel) — NUANCE.
  Anti-symétrique : si Δ_germ < 0 (ACCÉLÈRE) → inversion, à rapporter telle quelle.
"""
import os
import sys
import json
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")

from grok import Block, DEVICE
from gamma_v3_datafraction import make_data_frac

sys.argv.append("--v1")
import xi_gamma_germination as xg

RESULTS = os.path.join(HERE, "gamma_L3_ablation")
RANK_LO, RANK_HI = 3, 32
WINDOW = 6
K_ABL = 8
EVERY = 50
STEPS = 4500
SEEDS = [20, 21]
BASELINE_TGROK = {20: 1250, 21: 1150}      # v3 frac=0.5 mêmes seeds/data


class AblatedGrokGPT(nn.Module):
    """GrokGPT + projection hors du sous-espace U à la position eq (pre-LN)."""

    def __init__(self, vocab, p, d_model=128, n_head=4, n_layer=2, ff_mult=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, ff_mult) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)
        self.register_buffer("U", torch.zeros(d_model, 0))   # (d, k)

    def set_subspace(self, U_np):
        U = torch.tensor(np.ascontiguousarray(U_np.T), dtype=torch.float32,
                         device=self.tok.weight.device)       # (d, k)
        self.U = U

    def forward(self, idx, want_last_act=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        la = x[:, -1, :]
        if self.U.shape[1] > 0:
            la = la - (la @ self.U) @ self.U.t()
        logits = self.head(self.ln(la))
        return (logits, la) if want_last_act else logits


class OnlineGermDetector:
    """Réplique Γ v1 en ligne : buffer 6 probes, ancre = plus vieux."""

    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.buffer = deque(maxlen=WINDOW)
        self.k_history = []

    def update(self, acts):
        a = acts.astype(np.float64)
        a = a - a.mean(axis=0, keepdims=True)
        a = a / (np.linalg.norm(a) + 1e-12)
        self.buffer.append(a)
        if len(self.buffer) < WINDOW:
            self.k_history.append(0)
            return np.zeros((0, a.shape[1]))
        anchor = self.buffer[0]
        _, _, Vt = np.linalg.svd(anchor, full_matrices=False)
        cand = Vt[RANK_LO:RANK_HI]
        null = self.rng.standard_normal(cand.shape)
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        win = list(self.buffer)
        s_cand = xg.slopes_for_dirs(win, cand)
        s_null = xg.slopes_for_dirs(win, null)
        theta = np.percentile(s_null, 95)
        order = np.argsort(s_cand)[::-1]
        growing = [i for i in order if s_cand[i] > theta][:K_ABL]
        k = len(growing)
        self.k_history.append(k)
        if k == 0:
            return np.zeros((0, cand.shape[1]))
        if self.mode == "germ":
            U = cand[growing]
        elif self.mode == "lowslope":
            U = cand[np.argsort(s_cand)[:k]]
        elif self.mode == "random":
            R = self.rng.standard_normal((k, cand.shape[1]))
            U, _ = np.linalg.qr(R.T)
            U = U.T[:k]
        else:
            raise ValueError(self.mode)
        # orthonormalise (sécurité)
        Uq, _ = np.linalg.qr(U.T)
        return Uq.T[:k]


def run_condition(seed, mode, p=97, frac=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xte, Yte, vocab = make_data_frac(p, seed, frac)
    model = AblatedGrokGPT(vocab, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                            betas=(0.9, 0.98))
    probe_te = Xte[:512]
    det = OnlineGermDetector(mode, seed=seed * 101)

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def probe_acts():
        model.eval()
        _, la = model(probe_te, want_last_act=True)
        model.train()
        return la.cpu().numpy()

    traj = []
    t0 = time.time()
    for step in range(STEPS + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            U = det.update(probe_acts())
            model.set_subspace(U)
            traj.append({"step": step, "train_acc": acc(Xtr, Ytr),
                         "val_acc": acc(Xte, Yte), "k": int(U.shape[0])})
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    print(f"  {mode:9s} s{seed} : t_grok={t_grok} "
          f"k_mean={np.mean([t['k'] for t in traj]):.1f} "
          f"({time.time() - t0:.0f}s)")
    return {"seed": seed, "mode": mode, "t_grok": t_grok, "traj": traj,
            "k_history": det.k_history}


def main():
    t0 = time.time()
    print("[L3 ablation — frac=0.5 wd=1.0, steps=4500]")
    runs = []
    for seed in SEEDS:
        for mode in ("germ", "random", "lowslope"):
            runs.append(run_condition(seed, mode))

    # -------- verdict pré-enregistré
    def delta(r):
        base = BASELINE_TGROK[r["seed"]]
        tg = r["t_grok"] if r["t_grok"] is not None else STEPS  # censuré
        return tg - base

    table = {}
    for r in runs:
        table.setdefault(r["seed"], {})[r["mode"]] = {
            "t_grok": r["t_grok"], "delta": delta(r),
            "censored": r["t_grok"] is None}

    per_seed_pass, per_seed_kill = [], []
    for seed in SEEDS:
        row = table[seed]
        dg, dr, dl = (row["germ"]["delta"], row["random"]["delta"],
                      row["lowslope"]["delta"])
        per_seed_pass.append(dg >= dr + 300 and dg >= dl + 300)
        per_seed_kill.append(dg <= max(dr, dl) + 100)

    if all(per_seed_pass):
        verdict = "KEEP L3 (le germe est causal — l'ablater retarde le grok)"
    elif any(per_seed_kill):
        verdict = "KILL (épiphénomène ou re-routage plus rapide que la chasse)"
    else:
        verdict = "NUANCE (causalité faible / whack-a-mole partiel)"

    out = {"preregistered": {
              "baseline_tgrok": BASELINE_TGROK,
              "table": table,
              "per_seed_pass": per_seed_pass,
              "per_seed_kill": per_seed_kill,
              "verdict": verdict},
           "runs": [{k: v for k, v in r.items() if k != "traj"} for r in runs],
           "trajs": [{"seed": r["seed"], "mode": r["mode"],
                      "traj": r["traj"]} for r in runs],
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    for seed in SEEDS:
        row = table[seed]
        print(f"seed {seed} (baseline {BASELINE_TGROK[seed]}) : " + "  ".join(
            f"{m}: t={row[m]['t_grok']} Δ={row[m]['delta']:+d}"
            f"{'(cens.)' if row[m]['censored'] else ''}"
            for m in ("germ", "random", "lowslope")))
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
