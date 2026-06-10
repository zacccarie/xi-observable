"""Γ L4 — GÉNÉRALITÉ : sparse parity + MLP (2e tâche, 2e architecture).

Γ forecaster validé L1+L2 sur modular-add/transformer. Ici : le même
instrument, paramètres IDENTIQUES (ranks 4..32, fenêtre 6 checkpoints,
θ = q95 null, Frobenius-norm), sur (30,3)-sparse parity / MLP 256.
Config tâche validée : metamorphic-systems/grokking_beta/modal_sparse_parity.py
(n_train=1000 groke < 20k steps, wd=1e-2).

DEUX PHASES :
  1. PILOTE (calibration, 1 seed, n'entre PAS dans la preuve) :
     n_train ∈ {250, 500, 1000, 2000} → frontière grok/no-grok dans 20k steps
     + cap de fenêtre no-grok = 0.8 × min(t_grok pilote), arrondi à 100.
  2. TEST (seeds FRAIS 30-31, jamais vus) : 2 n_train grokkers + 2 no-grok
     issus du pilote, 2 seeds chacun = 8 runs.

PRÉ-ENREGISTREMENT (avant les runs frais ; pilote = calibration documentée) :
  Γ_window = mean Γ sur [t_mem+50, (t_grok−100) si grok sinon cap).
  H-L4 : min(Γ_window | grok) > max(Γ_window | no-grok) sur les 8 runs frais
         (séparation binaire par run, réplique v3 sur tâche+archi nouvelles).
  KILL : chevauchement → Γ forecaster spécifique au couple
         modular-add × transformer → généralité tombe.

Activations : h2 (avant-dernière couche, 256-d — celle du β-reg validé),
probe test fixe 384, checkpoint tous les 100 steps.
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

RESULTS = os.path.join(HERE, "gamma_L4_parity")
DEVICE = torch.device("cpu")
N_BITS, K_PARITY = 30, 3
WD = 1e-2
STEPS = 20000
EVERY = 100
PILOT_NTRAIN = [250, 500, 1000, 2000]
PILOT_SEED = 5
FRESH_SEEDS = [30, 31]


class ParityMLP(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.fc1 = nn.Linear(N_BITS, width)
        self.fc2 = nn.Linear(width, width)
        self.head = nn.Linear(width, 2)

    def forward(self, x, want_act=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.head(h2)
        return (logits, h2) if want_act else logits


def run_parity(n_train, seed):
    outdir = os.path.join(xg.CKPT_ROOT, f"run_parityN{n_train}_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    parity_bits = rng.choice(N_BITS, size=K_PARITY, replace=False)

    def make(n):
        X = rng.integers(0, 2, size=(n, N_BITS)).astype(np.float32)
        Y = (X[:, parity_bits].sum(axis=1) % 2).astype(np.int64)
        return torch.tensor(X, device=DEVICE), torch.tensor(Y, device=DEVICE)

    Xtr, Ytr = make(n_train)
    Xte, Yte = make(2000)
    probe = Xte[:384]
    model = ParityMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD)

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def probe_acts():
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
                                acts_test=probe_acts())
            traj.append({"step": step, "train_acc": acc(Xtr, Ytr),
                         "val_acc": acc(Xte, Yte)})
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "n_train": n_train, "wd": WD, "t_grok": t_grok,
            "traj": traj, "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, "traj.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  N={n_train} s{seed} : t_grok={t_grok} ({meta['elapsed_s']:.0f}s)")
    return meta


def gamma_window(rundir, cap):
    steps, acts, meta = xg.load_run(rundir)
    anchors, gamma, theta = xg.gamma_series(steps, acts, seed=meta["seed"])
    t_mem = next((t["step"] for t in meta["traj"] if t["train_acc"] > 0.99), None)
    t_grok = meta["t_grok"]
    lo = (t_mem + 50) if t_mem is not None else 200
    hi = (t_grok - 100) if t_grok else cap
    vals = [g for a, g in zip(anchors, gamma) if lo <= a < hi]
    return {"dir": os.path.basename(rundir), "seed": meta["seed"],
            "n_train": meta["n_train"], "t_mem": t_mem, "t_grok": t_grok,
            "window": [lo, hi], "theta": theta,
            "too_short": (hi - lo) < 3 * EVERY or len(vals) < 3,
            "gamma_window": float(np.mean(vals)) if vals else float("nan")}


def main():
    t0 = time.time()
    print("[PHASE 1 — pilote calibration (hors preuve)]")
    pilot = [run_parity(n, PILOT_SEED) for n in PILOT_NTRAIN]
    grok_ns = sorted([p["n_train"] for p in pilot if p["t_grok"] is not None])
    nogrok_ns = sorted([p["n_train"] for p in pilot if p["t_grok"] is None])
    print(f"  pilote : grok={grok_ns} no-grok={nogrok_ns}")
    if len(grok_ns) < 2 or len(nogrok_ns) < 2:
        extra = []
        if len(nogrok_ns) < 2:
            extra = [125, 60]
        elif len(grok_ns) < 2:
            extra = [4000, 8000]
        for n in extra:
            p = run_parity(n, PILOT_SEED)
            (grok_ns if p["t_grok"] else nogrok_ns).append(n)
            grok_ns.sort(); nogrok_ns.sort()
            if len(grok_ns) >= 2 and len(nogrok_ns) >= 2:
                break
    test_grok_ns = grok_ns[:2]                 # 2 plus petits grokkers
    test_nogrok_ns = nogrok_ns[-2:]            # 2 plus grands no-grok (frontière)
    min_grok_t = min(p["t_grok"] for p in pilot if p["t_grok"] is not None)
    cap = int(round(0.8 * min_grok_t, -2))
    print(f"  conditions test : grok_ns={test_grok_ns} nogrok_ns={test_nogrok_ns} "
          f"cap={cap}")

    print("[PHASE 2 — test seeds frais (pré-enregistré)]")
    for n in test_grok_ns + test_nogrok_ns:
        for seed in FRESH_SEEDS:
            run_parity(n, seed)

    rows = []
    for n in test_grok_ns + test_nogrok_ns:
        for seed in FRESH_SEEDS:
            rd = os.path.join(xg.CKPT_ROOT, f"run_parityN{n}_s{seed}")
            rows.append(gamma_window(rd, cap))
    for r in rows:
        print(f"  {r['dir']:24s} t_grok={r['t_grok']} win={r['window']} "
              f"Γ_win={r['gamma_window']:.4f}"
              f"{'  [EXCLU]' if r['too_short'] else ''}")

    valid = [r for r in rows if not r["too_short"]]
    grokkers = [r for r in valid if r["t_grok"] is not None]
    nogrok = [r for r in valid if r["t_grok"] is None]
    binary_sep = (min(r["gamma_window"] for r in grokkers) >
                  max(r["gamma_window"] for r in nogrok)) \
        if (grokkers and nogrok) else None

    if binary_sep is True:
        verdict = "KEEP L4 (Γ forecaster généralise : 2e tâche + 2e archi)"
    elif binary_sep is False:
        verdict = "KILL (Γ spécifique modular-add × transformer)"
    else:
        verdict = "NUANCE (contraste destin absent sur seeds frais)"

    out = {"preregistered": {
              "pilot_grok_ns": grok_ns, "pilot_nogrok_ns": nogrok_ns,
              "test_conditions": {"grok": test_grok_ns, "nogrok": test_nogrok_ns},
              "window_cap": cap,
              "gamma_window_grokkers": [r["gamma_window"] for r in grokkers],
              "gamma_window_nogrok": [r["gamma_window"] for r in nogrok],
              "binary_separation": binary_sep,
              "excluded": [r["dir"] for r in rows if r["too_short"]],
              "verdict": verdict},
           "rows": rows,
           "pilot": [{k: p[k] for k in ("seed", "n_train", "t_grok")}
                     for p in pilot],
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"grokkers Γ_win={[round(r['gamma_window'],4) for r in grokkers]}")
    print(f"no-grok  Γ_win={[round(r['gamma_window'],4) for r in nogrok]}")
    print(f"binaire={binary_sep}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
