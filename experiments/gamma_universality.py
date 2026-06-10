"""Γ UNIVERSALITÉ — la phase préparatoire a-t-elle des invariants cross-système ?

L'idée (physique stat) : aux transitions de phase, des systèmes différents
partagent les mêmes exposants critiques (universalité). Ici : si la
germination est une dynamique universelle, les DISTRIBUTIONS de taux de
croissance des germes — rendues sans dimension — doivent se SUPERPOSER
entre testbeds (collapse), comme des courbes de finite-size scaling.

DONNÉES (déjà sur disque, zéro re-run) : checkpoints d'activations des 3
testbeds from-scratch où Γ est validé :
  - modadd  : GrokGPT transformer, EVERY=50  (runs wd1.0 s0-2,10-12 + frac v3 grokkers)
  - parity  : MLP 256, EVERY=100             (runs N1300 s30-31)
  - induct  : attn-only 2L, EVERY=25         (runs inductv3_2L s30-37)

STATISTIQUE (pré-enregistrée) :
  Par run grokker, fenêtre pré-grok principielle [lo, hi) (identique aux
  analyses antérieures). Par ancre : SVD (Frobenius-norm), candidates rangs
  4..32, pentes sur fenêtre 6 ckpts, θ = q95 du null aléatoire poolé/run.
  GERME = pentes > θ. On collecte les excès SANS DIMENSION :
      ẽ = (pente − θ) / EVERY × τ        avec τ = hi − lo (durée fenêtre)
  (pente/EVERY = taux par step ; × τ = croissance par durée-de-plateau.)
  Pool par testbed → distributions cumulatives (ECDF).

VERDICT (pré-enregistré, avant run) :
  baseline de bruit = KS intra-testbed (split-half des runs, moitiés de seeds)
  KEEP (collapse)  : max KS cross-testbed (3 paires, rescalé) ≤
                     2 × max KS intra-testbed ET < 0.30 absolu.
  KILL             : une paire cross ≥ 0.40 → pas d'invariant, la
                     « universalité de la phase préparatoire » meurt.
  Entre les deux   : NUANCE.
  Contrôle de pertinence du rescaling : rapporter aussi le KS NON-rescalé
  (taux par step bruts) — si rescalé ≈ non-rescalé, le rescaling est
  décoratif et le claim d'« exposants » s'affaiblit (à rapporter honnêtement).
"""
import os
import sys
import json
import glob
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.argv.append("--v1")
import xi_gamma_germination as xg

RESULTS = os.path.join(HERE, "gamma_universality")
RANK_LO, RANK_HI = 3, 32
WINDOW = 6
NULL_Q = 95


def excess_rates(rundir, lo, hi, every):
    """Excès de pente (germe) par step, dans la fenêtre [lo, hi)."""
    steps, acts, meta = xg.load_run(rundir)   # centré + Frobenius (mode --v1)
    rng = np.random.default_rng(meta["seed"])
    cand_slopes, null_pool, anchors = [], [], []
    for i in range(len(acts) - WINDOW + 1):
        _, _, Vt = np.linalg.svd(acts[i], full_matrices=False)
        cand = Vt[RANK_LO:RANK_HI]
        null = rng.standard_normal(cand.shape)
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        win = acts[i:i + WINDOW]
        cand_slopes.append(xg.slopes_for_dirs(win, cand))
        null_pool.append(xg.slopes_for_dirs(win, null))
        anchors.append(steps[i])
    theta = np.percentile(np.concatenate(null_pool), NULL_Q)
    out = []
    for a, s in zip(anchors, cand_slopes):
        if lo <= a < hi:
            out.extend((v - theta) / every for v in s if v > theta)
    return np.array(out)


def window_for(meta, every, kind):
    traj = meta["traj"]
    t_grok = meta["t_grok"]
    if kind == "induct":
        lo = 200
    else:
        t_mem = next((t["step"] for t in traj if t["train_acc"] > 0.99), None)
        lo = (t_mem + 50) if t_mem is not None else 200
    hi = t_grok - 100
    return lo, hi


def ks_distance(x, y):
    """Kolmogorov-Smirnov bilatéral, sans scipy."""
    x, y = np.sort(x), np.sort(y)
    allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / len(x)
    cy = np.searchsorted(y, allv, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


def main():
    t0 = time.time()
    testbeds = {
        "modadd": {"every": 50, "kind": "grok", "dirs":
                   [f"run_wd1.0_s{s}" for s in (0, 1, 2, 10, 11, 12)] +
                   ["run_frac0.7_wd1.0_s20", "run_frac0.7_wd1.0_s21",
                    "run_frac0.5_wd1.0_s20", "run_frac0.5_wd1.0_s21",
                    "run_frac0.4_wd1.0_s20", "run_frac0.4_wd1.0_s21"]},
        "parity": {"every": 100, "kind": "grok", "dirs":
                   [f"run_parityN1300_s{s}" for s in (30, 31, 32, 33, 34, 35, 36)]},
        "induct": {"every": 25, "kind": "induct", "dirs":
                   [f"run_inductv3_2L_s{s}" for s in range(30, 38)]},
    }

    pooled_resc, pooled_raw, per_run = {}, {}, {}
    for name, tb in testbeds.items():
        resc, raw, runs = [], [], []
        for d in tb["dirs"]:
            rd = os.path.join(xg.CKPT_ROOT, d)
            if not os.path.exists(os.path.join(rd, "traj.json")):
                continue
            meta = json.load(open(os.path.join(rd, "traj.json")))
            if meta.get("t_grok") is None:
                continue
            lo, hi = window_for(meta, tb["every"], tb["kind"])
            if hi - lo < 3 * tb["every"]:
                continue
            e = excess_rates(rd, lo, hi, tb["every"])
            if len(e) == 0:
                runs.append({"dir": d, "n": 0})
                continue
            tau = hi - lo
            resc.append(e * tau)
            raw.append(e)
            runs.append({"dir": d, "n": len(e), "tau": tau,
                         "median_raw": float(np.median(e)),
                         "median_resc": float(np.median(e * tau))})
            print(f"  {name:7s} {d:24s} n_germes={len(e):4d} τ={tau:5d} "
                  f"méd(ẽ)={np.median(e * tau):.3f}")
        pooled_resc[name] = np.concatenate(resc) if resc else np.array([])
        pooled_raw[name] = np.concatenate(raw) if raw else np.array([])
        per_run[name] = runs

    # -------- intra-testbed (split-half par runs) = plancher de bruit
    intra = {}
    for name, tb in testbeds.items():
        halves = [[], []]
        i = 0
        for d in tb["dirs"]:
            rd = os.path.join(xg.CKPT_ROOT, d)
            if not os.path.exists(os.path.join(rd, "traj.json")):
                continue
            meta = json.load(open(os.path.join(rd, "traj.json")))
            if meta.get("t_grok") is None:
                continue
            lo, hi = window_for(meta, tb["every"], tb["kind"])
            if hi - lo < 3 * tb["every"]:
                continue
            e = excess_rates(rd, lo, hi, tb["every"])
            if len(e):
                halves[i % 2].append(e * (hi - lo))
                i += 1
        if halves[0] and halves[1]:
            intra[name] = ks_distance(np.concatenate(halves[0]),
                                      np.concatenate(halves[1]))
    intra_max = max(intra.values())

    # -------- cross-testbed
    names = [n for n in testbeds if len(pooled_resc[n])]
    cross_resc, cross_raw = {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cross_resc[f"{a}↔{b}"] = ks_distance(pooled_resc[a], pooled_resc[b])
            cross_raw[f"{a}↔{b}"] = ks_distance(pooled_raw[a], pooled_raw[b])
    cross_max = max(cross_resc.values())

    if cross_max <= max(2 * intra_max, 1e-9) and cross_max < 0.30:
        verdict = "KEEP (COLLAPSE — invariants de germination cross-système)"
    elif cross_max >= 0.40:
        verdict = "KILL (pas d'invariant universel de la phase préparatoire)"
    else:
        verdict = "NUANCE (collapse partiel)"

    out = {"preregistered": {
              "cross_ks_rescaled": cross_resc, "cross_ks_raw": cross_raw,
              "intra_ks": intra, "intra_max": intra_max,
              "cross_max": cross_max, "verdict": verdict},
           "n_pooled": {n: int(len(pooled_resc[n])) for n in names},
           "per_run": per_run, "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    # -------- figure : ECDF brut vs rescalé
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"modadd": "#00e5ff", "parity": "#b388ff", "induct": "#ff6d00"}
    for ax, pool, title in [(axes[0], pooled_raw, "taux bruts (par step)"),
                            (axes[1], pooled_resc, "sans dimension (× τ)")]:
        for n in names:
            x = np.sort(pool[n])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=colors[n],
                    lw=1.8, label=f"{n} (n={len(x)})")
        ax.set_xscale("log")
        ax.set_title(title, color="white")
        ax.set_xlabel("excès de croissance du germe")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("ECDF")
    fig.suptitle(f"Universalité de la germination — {verdict}\n"
                 f"KS cross max={cross_max:.3f} | intra max={intra_max:.3f} "
                 f"| brut max={max(cross_raw.values()):.3f}", color="white",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS + ".png", dpi=130, facecolor="black")

    print("\n" + "=" * 60)
    print(f"intra (plancher bruit) : {intra}")
    print(f"cross rescalé          : {cross_resc}")
    print(f"cross brut             : {cross_raw}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
