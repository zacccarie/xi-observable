"""Γ v3 — TEST DÉCISIF : data-fraction à wd FIXE (lève le confound wd ↔ destin).

v2 a confirmé la vague pré-grok sur seeds frais, MAIS contrôles = wd=0 →
Γ pourrait signer « wd actif » et non « germe de circuit ». Ici : wd=1.0
PARTOUT, on varie la fraction de données d'entraînement — c'est elle qui
décide du destin (grok tôt / tard / pas dans le budget 3000 steps).

PRÉ-ENREGISTREMENT (avant tout run) :
  Fenêtre principielle (généralise v2 hors frac=0.5) :
    t_mem = 1er step train_acc > 0.99
    window = [t_mem + 50,  (t_grok − 100) si grok sinon 1100)
    exclusion si fenêtre < 150 steps (3 ancres) — exclu + rapporté.
  Γ_window = mean Γ_v1(t) sur la fenêtre (Γ_v1 = Frobenius-normalisé,
    ranks 4..32, θ = q95 null aléatoire/run — identique v1/v2).

  H-Γ3a (primaire, binaire) : min(Γ_window | grok ≤ 3000) >
    max(Γ_window | pas de grok). Séparation par run totale.
  H-Γ3b (secondaire, si ≥ 4 grokkers) : Spearman(Γ_window, t_grok) ≤ −0.6
    (vague forte = grok tôt).
  Fallback si TOUTES les fractions grokent : H-Γ3b devient primaire.
  KILL : chevauchement binaire OU Spearman > 0 → Γ signe l'effet wd,
    pas le destin — le claim « germe » retombe à « détecteur de wd ».

Conditions : wd=1.0, frac ∈ {0.70, 0.50, 0.30, 0.25} × seeds {20, 21}.
Attendu (lit. grokking) : 0.70 groke tôt, 0.50 ~1200-1400, 0.30 tard/jamais,
0.25 jamais dans 3000.
"""
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")

from grok import GrokGPT, DEVICE

sys.argv.append("--v1")
import xi_gamma_germination as xg
assert xg.FROB_NORM

RESULTS = os.path.join(HERE, "gamma_v3_datafraction")
FRACS = [0.70, 0.50, 0.30, 0.25]
SEEDS = [20, 21]
STEPS = 3000
EVERY = 50


def make_data_frac(p, seed, frac):
    """make_data de grok.py avec fraction d'entraînement paramétrable."""
    eq_token, vocab = p, p + 1
    pairs = [(a, b) for a in range(p) for b in range(p)]
    X = np.array([[a, b, eq_token] for a, b in pairs], dtype=np.int64)
    Y = np.array([(a + b) % p for a, b in pairs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(frac * len(X))
    t = lambda a: torch.tensor(a, device=DEVICE)
    return t(X[:n_train]), t(Y[:n_train]), t(X[n_train:]), t(Y[n_train:]), vocab


def run_and_save(frac, seed, p=97):
    outdir = os.path.join(xg.CKPT_ROOT, f"run_frac{frac}_wd1.0_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xte, Yte, vocab = make_data_frac(p, seed, frac)
    model = GrokGPT(vocab, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                            betas=(0.9, 0.98))
    probe_te = Xte[:512]

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def acts(Xs):
        model.eval()
        _, la = model(Xs, want_last_act=True)
        model.train()
        return la.cpu().numpy().astype(np.float16)

    traj = []
    t0 = time.time()
    for step in range(STEPS + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=acts(probe_te))
            traj.append({"step": step, "train_acc": acc(Xtr, Ytr),
                         "val_acc": acc(Xte, Yte)})
    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    meta = {"seed": seed, "wd": 1.0, "frac": frac, "t_grok": t_grok,
            "traj": traj, "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, "traj.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  frac={frac} s{seed} : t_grok={t_grok} ({meta['elapsed_s']:.0f}s)")
    return meta


def gamma_window_principled(rundir):
    steps, acts, meta = xg.load_run(rundir)
    anchors, gamma, theta = xg.gamma_series(steps, acts, seed=meta["seed"])
    t_mem = next((t["step"] for t in meta["traj"] if t["train_acc"] > 0.99), None)
    t_grok = meta["t_grok"]
    lo = (t_mem + 50) if t_mem is not None else 100
    hi = (t_grok - 100) if t_grok else 1100
    vals = [g for a, g in zip(anchors, gamma) if lo <= a < hi]
    too_short = (hi - lo) < 150 or len(vals) < 3
    return {
        "dir": os.path.basename(rundir), "frac": meta.get("frac", 0.5),
        "seed": meta["seed"], "t_mem": t_mem, "t_grok": t_grok,
        "window": [lo, hi], "too_short": bool(too_short), "theta": theta,
        "gamma_window": float(np.mean(vals)) if vals else float("nan"),
        "anchors": anchors, "gamma": gamma,
    }


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    t0 = time.time()
    print("[v3 runs — wd=1.0 fixe, frac sweep]")
    for frac in FRACS:
        for seed in SEEDS:
            run_and_save(frac, seed)

    rows = []
    for frac in FRACS:
        for seed in SEEDS:
            rd = os.path.join(xg.CKPT_ROOT, f"run_frac{frac}_wd1.0_s{seed}")
            rows.append(gamma_window_principled(rd))
    for r in rows:
        print(f"  {r['dir']:24s} t_mem={r['t_mem']} t_grok={r['t_grok']} "
              f"win={r['window']} Γ_win={r['gamma_window']:.4f}"
              f"{'  [EXCLU fenêtre courte]' if r['too_short'] else ''}")

    valid = [r for r in rows if not r["too_short"]]
    grokkers = [r for r in valid if r["t_grok"] is not None]
    nogrok = [r for r in valid if r["t_grok"] is None]

    # -------- verdict pré-enregistré
    binary_sep = None
    if grokkers and nogrok:
        binary_sep = min(r["gamma_window"] for r in grokkers) > \
                     max(r["gamma_window"] for r in nogrok)
    rho = spearman([r["gamma_window"] for r in grokkers],
                   [r["t_grok"] for r in grokkers]) if len(grokkers) >= 4 else None

    if binary_sep is True and (rho is None or rho <= -0.6):
        verdict = "KEEP (Γ prédit le destin à wd fixe — confound levé)"
    elif binary_sep is True:
        verdict = "KEEP binaire / NUANCE timing (ρ Spearman faible)"
    elif binary_sep is False:
        verdict = "KILL (chevauchement — Γ signait l'effet wd)"
    elif binary_sep is None and rho is not None and rho <= -0.6:
        verdict = "KEEP timing (toutes les fractions grokent ; fallback H-Γ3b)"
    else:
        verdict = "NUANCE (contraste insuffisant)"

    out = {"preregistered": {
              "binary_separation": binary_sep,
              "gamma_window_grokkers": [r["gamma_window"] for r in grokkers],
              "t_grok_grokkers": [r["t_grok"] for r in grokkers],
              "gamma_window_nogrok": [r["gamma_window"] for r in nogrok],
              "spearman_gamma_tgrok": rho,
              "excluded_too_short": [r["dir"] for r in rows if r["too_short"]],
              "verdict": verdict},
           "rows": [{k: v for k, v in r.items() if k not in ("anchors", "gamma")}
                    for r in rows],
           "series": [{k: r[k] for k in ("dir", "anchors", "gamma")} for r in rows],
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"grokkers Γ_win={[round(r['gamma_window'],4) for r in grokkers]} "
          f"t_grok={[r['t_grok'] for r in grokkers]}")
    print(f"no-grok  Γ_win={[round(r['gamma_window'],4) for r in nogrok]}")
    print(f"binaire={binary_sep}  Spearman(Γ,t_grok)={rho}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
