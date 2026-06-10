"""Γ L4 v2 — sparse parity : chasse au RÉGIME À PLATEAU.

v1 : échec de testbed (NUANCE vide) — N≥2000 généralise instantanément
(t_grok=100-200, aucun plateau à mesurer), N≤1000 jamais (20k). La parité
full-batch a un seuil d'échantillons abrupt (Barak 2022) ; le grokking
(plateau long puis bascule) vit dans une bande étroite près du seuil.

PHASE A (calibration fine, seed 5, hors preuve) :
  n_train ∈ {1100, 1300, 1500, 1700} — cherche « delayed grok » :
  t_grok ≥ 1500 (plateau mesurable). Si aucun → PHASE B : width=128
  (capacité réduite ralentit), n_train ∈ {1500, 2000}.
  Si toujours aucun → verdict TESTBED INAPPLICABLE (documenté, pas de preuve).

PRÉ-ENREGISTREMENT du test frais (AVANT seeds 30-31) :
  Classification PAR RUN (pas par condition — variance de seuil énorme) :
    delayed grok : 1500 ≤ t_grok        → mesurable
    instant      : t_grok < 600         → EXCLU (pas de plateau, documenté)
    no-grok      : t_grok = None
  Γ_window = mean Γ sur [t_mem+50, (t_grok−100) | cap),
  cap = 0.8 × min(t_grok des delayed du pilote), arrondi à 100.
  H-L4' : min(Γ_window | delayed) > max(Γ_window | no-grok).
  KILL : chevauchement. NUANCE : < 2 delayed OU < 2 no-grok valides.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gamma_L4_parity as L4
from gamma_L4_parity import run_parity   # cache par dossier réutilisé
import xi_gamma_germination as xg

RESULTS = os.path.join(HERE, "gamma_L4_parity_v2")
FRESH_SEEDS = [30, 31]
PILOT_SEED = 5
DELAYED_MIN = 1500
INSTANT_MAX = 600


def run_parity_width(n_train, seed, width):
    """Variante largeur — dossier suffixé pour ne pas collisionner le cache."""
    if width == 256:
        return run_parity(n_train, seed)
    orig_mlp = L4.ParityMLP
    orig_root = xg.CKPT_ROOT

    class NarrowMLP(orig_mlp):
        def __init__(self):
            super().__init__(width=width)

    L4.ParityMLP = NarrowMLP
    xg.CKPT_ROOT = os.path.join(orig_root, f"w{width}")
    os.makedirs(xg.CKPT_ROOT, exist_ok=True)
    try:
        return run_parity(n_train, seed)
    finally:
        L4.ParityMLP = orig_mlp
        xg.CKPT_ROOT = orig_root


def main():
    t0 = time.time()
    print("[PHASE A — calibration fine n_train, width=256]")
    pilotA = [run_parity(n, PILOT_SEED) for n in (1100, 1300, 1500, 1700)]
    delayed = [p for p in pilotA if p["t_grok"] and p["t_grok"] >= DELAYED_MIN]
    width = 256
    pilot = pilotA

    if not delayed:
        print("[PHASE B — fallback width=128]")
        width = 128
        pilotB = [run_parity_width(n, PILOT_SEED, width) for n in (1500, 2000)]
        delayed = [p for p in pilotB if p["t_grok"] and p["t_grok"] >= DELAYED_MIN]
        pilot = pilotA + pilotB

    if not delayed:
        out = {"verdict": "TESTBED INAPPLICABLE (aucun régime à plateau trouvé "
                          "— parité full-batch : transition instantané/jamais)",
               "pilot": [{k: p[k] for k in ("seed", "n_train", "t_grok")}
                         for p in pilot],
               "runtime_s": time.time() - t0}
        with open(RESULTS + ".json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"VERDICT : {out['verdict']}")
        return

    delayed_ns = sorted({p["n_train"] for p in delayed})[:2]
    nogrok_ns = [500, 1000]                       # connus v1 (20k, jamais)
    cap = int(round(0.8 * min(p["t_grok"] for p in delayed), -2))
    print(f"  delayed_ns={delayed_ns} (width={width}) nogrok_ns={nogrok_ns} cap={cap}")

    print("[PHASE TEST — seeds frais (pré-enregistré)]")
    rows = []
    for n in delayed_ns + nogrok_ns:
        for seed in FRESH_SEEDS:
            run_parity_width(n, seed, width)
            sub = f"w{width}/" if width != 256 else ""
            rd = os.path.join(HERE, "grok_ckpts", sub.rstrip("/"),
                              f"run_parityN{n}_s{seed}") if width != 256 else \
                 os.path.join(xg.CKPT_ROOT, f"run_parityN{n}_s{seed}")
            orig_root = xg.CKPT_ROOT
            if width != 256:
                xg.CKPT_ROOT = os.path.join(orig_root, f"w{width}")
            try:
                rows.append(L4.gamma_window(rd, cap))
            finally:
                xg.CKPT_ROOT = orig_root

    for r in rows:
        cls = ("instant" if (r["t_grok"] is not None and r["t_grok"] < INSTANT_MAX)
               else "delayed" if r["t_grok"] is not None else "nogrok")
        r["cls"] = cls
        print(f"  {r['dir']:24s} t_grok={r['t_grok']} [{cls}] win={r['window']} "
              f"Γ_win={r['gamma_window']:.4f}{'  [EXCLU]' if r['too_short'] else ''}")

    valid = [r for r in rows if not r["too_short"] and r["cls"] != "instant"]
    grokkers = [r for r in valid if r["cls"] == "delayed"]
    nogrok = [r for r in valid if r["cls"] == "nogrok"]

    if len(grokkers) >= 2 and len(nogrok) >= 2:
        binary_sep = (min(r["gamma_window"] for r in grokkers) >
                      max(r["gamma_window"] for r in nogrok))
        verdict = ("KEEP L4 (Γ généralise : parité + MLP)" if binary_sep
                   else "KILL (Γ spécifique modular-add × transformer)")
    else:
        binary_sep = None
        verdict = (f"NUANCE (contraste insuffisant : {len(grokkers)} delayed, "
                   f"{len(nogrok)} no-grok valides — variance de seuil)")

    out = {"preregistered": {
              "width": width, "delayed_ns": delayed_ns, "nogrok_ns": nogrok_ns,
              "cap": cap,
              "gamma_window_delayed": [r["gamma_window"] for r in grokkers],
              "gamma_window_nogrok": [r["gamma_window"] for r in nogrok],
              "binary_separation": binary_sep,
              "excluded_instant": [r["dir"] for r in rows if r["cls"] == "instant"],
              "verdict": verdict},
           "rows": rows,
           "pilot": [{k: p[k] for k in ("seed", "n_train", "t_grok")}
                     for p in pilot],
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"delayed Γ_win={[round(r['gamma_window'],4) for r in grokkers]}")
    print(f"no-grok Γ_win={[round(r['gamma_window'],4) for r in nogrok]}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
