"""Γ (germination) — l'exposant-mètre. Analyse offline sur grok_ckpts/.

PRÉ-ENREGISTRÉ (theory/germination.md §7, opérationnalisé) :
  Γ(t) = Σ_directions max(0, slope_i − θ_null)
    directions : rangs 4..32 de la SVD des activations test à l'ancre t
                 (top-3 exclues = dominance acquise)
    slope_i    : pente OLS de log-amplitude RMS(A_t' @ v_i frozen) sur fenêtre
                 de 6 checkpoints (250 steps), v_i FIGÉE à l'ancre
    θ_null     : q95 des pentes de directions ALÉATOIRES (même nombre, même
                 fenêtres, poolées sur le run) — débiaisage contre
                 E[max(0,bruit)] (leçon killswitch v2.2)
  t_Γ = 1er t où Γ > 0 soutenu 3 checkpoints consécutifs.

  H-Γ  : lead = t_grok − t_Γ ≥ 300 steps dans ≥ 2/3 seeds wd=1.0,
         ET contrôles wd=0 non soutenus.
  KILL : Γ plat jusqu'au grok OU monte dans contrôles → le germe ne vit pas
         dans les activations (version forte thèse B sur ce substrat).

Baselines :
  - weight_norm : t_wn = 1er step de déclin soutenu 3 checkpoints (signature
    grokking connue). Γ doit faire mieux (avance ou spécificité).
  - TC plateau (pré-enregistré, exploratoire d'hier) : mean TC test
    steps 300-1200, prédiction wd=1.0 > wd=0.0 au-delà des stds.

Logique du contraste : le bruit est fort mais infidèle (direction figée →
pente ≈ 0) ; le germe est faible mais têtu (pente > 0 sur direction figée).
La pertinence est un exposant, pas un coefficient.
"""
import os
import sys
import json
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from xi.interaction import total_correlation

CKPT_ROOT = os.path.join(HERE, "grok_ckpts")

# v1 : normalisation Frobenius par checkpoint — la "croissance" devient une
# RÉALLOCATION de part de variance vers la direction figée, insensible au
# gonflement global d'activations (artefact v0 : le contrôle mémorisant
# gonfle aussi, sharpening softmax sans wd).
FROB_NORM = "--v1" in sys.argv
RESULTS = os.path.join(HERE, "xi_gamma_germination" + ("_v1" if FROB_NORM else ""))

RANK_LO, RANK_HI = 3, 32        # candidates = rangs 4..32 (0-indexé 3..31)
WINDOW = 6                      # checkpoints par fenêtre de pente (250 steps)
SUSTAIN = 3
NULL_Q = 95


def load_run(rundir):
    files = sorted(glob.glob(os.path.join(rundir, "step_*.npz")))
    steps = [int(os.path.basename(f)[5:10]) for f in files]
    acts = [np.load(f)["acts_test"].astype(np.float64) for f in files]
    acts = [a - a.mean(axis=0, keepdims=True) for a in acts]
    if FROB_NORM:
        acts = [a / (np.linalg.norm(a) + 1e-12) for a in acts]
    meta = json.load(open(os.path.join(rundir, "traj.json")))
    return steps, acts, meta


def slopes_for_dirs(acts_window, dirs):
    """Pentes OLS de log-RMS-amplitude par direction figée, sur la fenêtre."""
    xs = np.arange(len(acts_window), dtype=np.float64)
    amps = np.stack([np.sqrt(np.mean((A @ dirs.T) ** 2, axis=0))
                     for A in acts_window])          # (W, n_dirs)
    log_a = np.log(amps + 1e-12)
    xc = xs - xs.mean()
    return (xc[:, None] * (log_a - log_a.mean(axis=0))).sum(0) / (xc ** 2).sum()


def gamma_series(steps, acts, seed=0):
    n = len(acts)
    d = acts[0].shape[1]
    rng = np.random.default_rng(seed)
    anchors, gamma_raw, null_slopes_pool, cand_slopes_per_anchor = [], [], [], []
    for i in range(n - WINDOW + 1):
        A = acts[i]
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        cand = Vt[RANK_LO:RANK_HI]                       # (29, d)
        null = rng.standard_normal((cand.shape[0], d))
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        win = acts[i:i + WINDOW]
        s_cand = slopes_for_dirs(win, cand)
        s_null = slopes_for_dirs(win, null)
        anchors.append(steps[i])
        cand_slopes_per_anchor.append(s_cand)
        null_slopes_pool.append(s_null)
    theta = np.percentile(np.concatenate(null_slopes_pool), NULL_Q)
    gamma = [float(np.sum(np.clip(s - theta, 0, None)))
             for s in cand_slopes_per_anchor]
    return anchors, gamma, float(theta)


def sustained_onset(anchors, series, threshold=0.0):
    for i in range(len(series) - SUSTAIN + 1):
        if all(series[i + j] > threshold for j in range(SUSTAIN)):
            return anchors[i]
    return None


def weight_norm_onset(meta):
    """1er step de déclin soutenu 3 checkpoints de la norme totale."""
    tr = meta["traj"]
    wn = [t["weight_norm"] for t in tr]
    for i in range(1, len(wn) - SUSTAIN + 1):
        if all(wn[i + j] < wn[i + j - 1] for j in range(SUSTAIN)):
            return tr[i]["step"]
    return None


def tc_series(acts, n_subsets=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for A in acts:
        binar = (A > np.median(A, axis=0, keepdims=True)).astype(np.int8)
        tcs = []
        for _ in range(n_subsets):
            idx = rng.choice(A.shape[1], 4, replace=False)
            sub = binar[:, idx]
            if sub.std(axis=0).min() < 1e-9:
                continue
            tcs.append(total_correlation(sub))
        out.append(float(np.mean(tcs)) if tcs else 0.0)
    return out


def main():
    t0 = time.time()
    rundirs = sorted(glob.glob(os.path.join(CKPT_ROOT, "run_*")))
    runs = []
    for rd in rundirs:
        steps, acts, meta = load_run(rd)
        anchors, gamma, theta = gamma_series(steps, acts, seed=meta["seed"])
        t_gamma = sustained_onset(anchors, gamma)
        t_wn = weight_norm_onset(meta)
        tc = tc_series(acts, seed=meta["seed"])
        tc_plateau = float(np.mean([v for s, v in zip(steps, tc)
                                    if 300 <= s <= 1200]))
        runs.append({"dir": os.path.basename(rd), "wd": meta["wd"],
                     "seed": meta["seed"], "t_grok": meta["t_grok"],
                     "t_gamma": t_gamma, "t_wn": t_wn, "theta": theta,
                     "anchors": anchors, "gamma": gamma, "steps": steps,
                     "tc": tc, "tc_plateau": tc_plateau,
                     "val_acc": [t["val_acc"] for t in meta["traj"]],
                     "weight_norm": [t["weight_norm"] for t in meta["traj"]]})
        print(f"{os.path.basename(rd)} : t_grok={meta['t_grok']} "
              f"t_Γ={t_gamma} t_wn={t_wn} θ={theta:.5f} "
              f"TC_plateau={tc_plateau:.4f}")

    grok_runs = [r for r in runs if r["wd"] == 1.0]
    ctrl_runs = [r for r in runs if r["wd"] == 0.0]

    # -------- verdict pré-enregistré
    leads = [(r["t_grok"] - r["t_gamma"])
             if (r["t_grok"] is not None and r["t_gamma"] is not None) else None
             for r in grok_runs]
    n_early = sum(1 for l in leads if l is not None and l >= 300)
    ctrl_sustained = [r["t_gamma"] is not None for r in ctrl_runs]
    leads_wn = [(r["t_grok"] - r["t_wn"])
                if (r["t_grok"] is not None and r["t_wn"] is not None) else None
                for r in grok_runs]
    tc1 = [r["tc_plateau"] for r in grok_runs]
    tc0 = [r["tc_plateau"] for r in ctrl_runs]
    tc_sep = (np.mean(tc1) - np.mean(tc0)) > (np.std(tc1) + np.std(tc0))

    if n_early >= 2 and not any(ctrl_sustained):
        verdict = "KEEP (germe audible dans les activations)"
    elif any(ctrl_sustained):
        verdict = "ARTEFACT (contrôle monte aussi)"
    elif all(l is None or l < 0 for l in leads):
        verdict = "KILL (Γ retardé — le germe ne vit pas dans les activations)"
    else:
        verdict = "NUANCE"

    out = {
        "preregistered": {
            "leads_gamma": leads, "n_early_ge300": n_early,
            "ctrl_sustained": ctrl_sustained,
            "leads_weight_norm_baseline": leads_wn,
            "tc_plateau_wd1": tc1, "tc_plateau_wd0": tc0,
            "tc_separation_beyond_stds": bool(tc_sep),
            "verdict": verdict,
        },
        "runs": [{k: v for k, v in r.items()
                  if k not in ("anchors", "gamma", "steps", "tc",
                               "val_acc", "weight_norm")} for r in runs],
        "series": [{k: r[k] for k in ("dir", "anchors", "gamma", "steps",
                                      "tc", "val_acc", "weight_norm")}
                   for r in runs],
        "runtime_s": time.time() - t0,
    }
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    # -------- figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.4))
    for ax, r in zip(axes, runs):
        ax2 = ax.twinx()
        ax.plot(r["steps"], r["val_acc"], color="#00e5ff", lw=1.6, label="val_acc")
        wn = np.array(r["weight_norm"])
        ax.plot(r["steps"], (wn - wn.min()) / (wn.max() - wn.min() + 1e-9),
                color="#888888", lw=0.9, alpha=0.7, label="‖W‖ (norm.)")
        ax2.plot(r["anchors"], r["gamma"], color="#ff6d00", lw=1.7, label="Γ")
        if r["t_grok"]:
            ax.axvline(r["t_grok"], color="#00e5ff", ls="--", lw=0.9, alpha=0.7)
        if r["t_gamma"]:
            ax.axvline(r["t_gamma"], color="#ff6d00", ls="--", lw=0.9, alpha=0.8)
        if r["t_wn"]:
            ax.axvline(r["t_wn"], color="#888888", ls=":", lw=0.9, alpha=0.8)
        ax.set_title(f"wd={r['wd']} s{r['seed']}\n"
                     f"t_Γ={r['t_gamma']} t_grok={r['t_grok']} t_wn={r['t_wn']}",
                     fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("step")
    axes[0].set_ylabel("val_acc / ‖W‖")
    fig.suptitle(f"Γ germination — {verdict}", color="white")
    plt.tight_layout()
    plt.savefig(RESULTS + ".png", dpi=130, facecolor="black")

    print("\n" + "=" * 60)
    print(f"leads Γ      : {leads}  (≥300 dans {n_early}/3)")
    print(f"leads ‖W‖    : {leads_wn}  (baseline)")
    print(f"ctrl soutenu : {ctrl_sustained}")
    print(f"TC plateau   : wd1={np.round(tc1,4)} vs wd0={np.round(tc0,4)} "
          f"sep={tc_sep}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
