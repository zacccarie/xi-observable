"""Γ-POIDS v0 — validation jouets (pivot post-test3 : le germe vit dans les poids).

test3 a montré des ROUTES DE FORMATION SILENCIEUSES en activations (s40 :
0.0003 sur 28 canaux, même destin que s41 à 0.626). L3 disait déjà : le germe
vit dans les poids. Ici : la même machinerie d'exposants (gamma_series v1,
INCHANGÉE) appliquée aux MATRICES DE POIDS — lignes = échantillons
(tokens/neurones), colonnes = dims. La Frobenius-normalisation par checkpoint
absorbe le decay global du weight decay → on mesure la réallocation interne.

Données : poids fp16 déjà sur disque tous les 50 steps —
grok wd=1.0 s{0,1,2,10,11,12} (6) + contrôles wd=0.0 s{0,1,10,11} (4).

Canaux (6 matrices) : tok.weight, head.weight, in_proj attn × 2 blocs,
mlp.0.weight × 2 blocs. Γw_window(run) = max canaux de mean Γ sur la fenêtre
v2 [500, min(1100, t_grok−100)) — IDENTIQUE à gamma_v2_fresh pour
comparabilité directe (mêmes runs).

PRÉ-ENREGISTREMENT :
  H-w1 (réplication poids) : séparation binaire min(Γw grokkers) >
        max(Γw contrôles) — comme v2 l'a fait en activations.
  KILL : chevauchement → machinerie poids cassée → STOP avant de payer
         la phase pythia.
  Exploratoire : timing Γ-poids vs Γ-activations (le poids précède-t-il ?).
"""
import os
import sys
import glob
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.argv.append("--v1")
sys.path.insert(0, HERE)
import xi_gamma_germination as xg

CKPT_ROOT = os.path.join(HERE, "grok_ckpts")
RESULTS = os.path.join(HERE, "gamma_Aw_toys")
CHANNELS = ["tok.weight", "head.weight",
            "blocks.0.attn.in_proj_weight", "blocks.1.attn.in_proj_weight",
            "blocks.0.mlp.0.weight", "blocks.1.mlp.0.weight"]
RUNS = [("wd1.0", s) for s in (0, 1, 2, 10, 11, 12)] + \
       [("wd0.0", s) for s in (0, 1, 10, 11)]


def load_weight_series(rundir, channel):
    files = sorted(glob.glob(os.path.join(rundir, "weights_*.pt")))
    steps = [int(os.path.basename(f)[8:13]) for f in files]
    mats = []
    for f in files:
        sd = torch.load(f, map_location="cpu")
        mats.append(sd[channel].float().numpy().astype(np.float64))
    return steps, mats


def main():
    rows = []
    for wd_tag, seed in RUNS:
        rundir = os.path.join(CKPT_ROOT, f"run_{wd_tag}_s{seed}")
        meta = json.load(open(os.path.join(rundir, "traj.json")))
        t_grok = meta["t_grok"]
        hi = min(1100, (t_grok - 100) if t_grok else 1100)
        per_ch = {}
        for ch in CHANNELS:
            steps, mats = load_weight_series(rundir, ch)
            anchors, gamma, theta = xg.gamma_series(steps, mats, seed=seed)
            vals = [g for a, g in zip(anchors, gamma) if 500 <= a < hi]
            if len(vals) >= 3:
                per_ch[ch] = float(np.mean(vals))
        best_ch = max(per_ch, key=per_ch.get) if per_ch else None
        gw = per_ch.get(best_ch, float("nan"))
        rows.append({"wd": wd_tag, "seed": seed, "t_grok": t_grok,
                     "window_hi": hi, "gw": gw, "best_ch": best_ch,
                     "per_ch": per_ch})
        print(f"{wd_tag} s{seed:2d} t_grok={t_grok} : Γw={gw:.4f} "
              f"[{best_ch}]  " +
              " ".join(f"{c.split('.')[0]}{c[7] if c.startswith('blocks') else ''}"
                       f"={v:.3f}" for c, v in per_ch.items()))

    grok = [r for r in rows if r["wd"] == "wd1.0"]
    ctrl = [r for r in rows if r["wd"] == "wd0.0"]
    sep = min(g["gw"] for g in grok) > max(c["gw"] for c in ctrl)
    verdict = ("KEEP Γ-poids (réplication — go phase pythia)" if sep
               else "KILL (machinerie poids ne sépare pas — stop)")

    out = {"preregistered": {"gw_grok": [g["gw"] for g in grok],
                             "gw_ctrl": [c["gw"] for c in ctrl],
                             "binary_separation": bool(sep),
                             "verdict": verdict},
           "rows": rows}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"grokkers Γw : {sorted(round(g['gw'],4) for g in grok)}")
    print(f"contrôles Γw: {sorted(round(c['gw'],4) for c in ctrl)}")
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
