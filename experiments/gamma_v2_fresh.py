"""Γ v2 — confirmation sur SEEDS FRAIS (anti-HARKing).

La statistique de fenêtre (vague 2) a été découverte en EXPLORANT les séries
v1 des seeds 0-2. Elle est ici PRÉ-ENREGISTRÉE puis testée sur des seeds
jamais vus (10-12 wd=1.0, 10-11 wd=0.0).

PRÉ-ENREGISTREMENT (avant tout run frais) :
  Γ_window(run) = mean Γ_v1(t) pour t ∈ [500, min(1100, t_grok−100))
                  (post-mémorisation, pré-grok ; Γ_v1 = Frobenius-normalisé,
                   θ = q95 null aléatoire par run, identique à v1)
  H-Γ2 : mean Γ_window(wd=1.0 frais) ≥ 5 × mean Γ_window(wd=0.0 frais),
         ET chaque run wd=1.0 frais > max(run wd=0.0 frais).
  KILL : ratio < 5 OU chevauchement par seed → la vague 2 était du
         cherry-picking de fenêtre sur 5 runs.
  Référence exploratoire seeds 0-2 : ratio ≈ 17× sur [600,1200).
"""
import os
import sys
import json
import glob
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grok_checkpoint_runs as gcr

# réutilise gamma_series v1 : importe le module en forçant le mode Frobenius
sys.argv.append("--v1")
import xi_gamma_germination as xg
assert xg.FROB_NORM, "mode v1 (Frobenius) requis"

RESULTS = os.path.join(HERE, "gamma_v2_fresh")
FRESH_GROK = [10, 11, 12]
FRESH_CTRL = [10, 11]


def gamma_window(rundir):
    steps, acts, meta = xg.load_run(rundir)
    anchors, gamma, theta = xg.gamma_series(steps, acts, seed=meta["seed"])
    t_grok = meta["t_grok"]
    hi = min(1100, (t_grok - 100) if t_grok else 1100)
    vals = [g for a, g in zip(anchors, gamma) if 500 <= a < hi]
    return {
        "dir": os.path.basename(rundir), "wd": meta["wd"], "seed": meta["seed"],
        "t_grok": t_grok, "theta": theta, "window_hi": hi,
        "gamma_window": float(np.mean(vals)) if vals else float("nan"),
        "anchors": anchors, "gamma": gamma,
    }


def main():
    t0 = time.time()
    print("[fresh runs]")
    for seed in FRESH_GROK:
        if not os.path.exists(os.path.join(xg.CKPT_ROOT, f"run_wd1.0_s{seed}", "traj.json")):
            gcr.run_and_save(seed, weight_decay=1.0)
    for seed in FRESH_CTRL:
        if not os.path.exists(os.path.join(xg.CKPT_ROOT, f"run_wd0.0_s{seed}", "traj.json")):
            gcr.run_and_save(seed, weight_decay=0.0)

    rows = []
    for seed in FRESH_GROK:
        rows.append(gamma_window(os.path.join(xg.CKPT_ROOT, f"run_wd1.0_s{seed}")))
    for seed in FRESH_CTRL:
        rows.append(gamma_window(os.path.join(xg.CKPT_ROOT, f"run_wd0.0_s{seed}")))
    for r in rows:
        print(f"  {r['dir']:16s} t_grok={r['t_grok']} "
              f"Γ_window={r['gamma_window']:.4f} (hi={r['window_hi']})")

    g1 = [r["gamma_window"] for r in rows if r["wd"] == 1.0]
    g0 = [r["gamma_window"] for r in rows if r["wd"] == 0.0]
    ratio = float(np.mean(g1) / (np.mean(g0) + 1e-12))
    per_seed_sep = min(g1) > max(g0)
    verdict = ("KEEP (vague 2 confirmée sur seeds frais)"
               if (ratio >= 5.0 and per_seed_sep)
               else "KILL (cherry-picking de fenêtre)")

    out = {"preregistered": {"gamma_window_wd1": g1, "gamma_window_wd0": g0,
                             "ratio": ratio, "per_seed_separation": per_seed_sep,
                             "verdict": verdict},
           "rows": [{k: v for k, v in r.items() if k not in ("anchors", "gamma")}
                    for r in rows],
           "series": [{k: r[k] for k in ("dir", "anchors", "gamma")} for r in rows],
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"Γ_window wd1={np.round(g1,4)} wd0={np.round(g0,4)}")
    print(f"ratio={ratio:.1f}×  per-seed sep={per_seed_sep}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
