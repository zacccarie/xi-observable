"""C-STAB — décomposition de variance des signatures Γ (LE verrou).

Γ mesure-t-il l'ARCHITECTURE ou le COUPLAGE architecture×tâche ?

Grille disponible :
  parity   : mlp (×7), conv1d/deepsets/odelite (×2), transformer (×2)
  modadd   : transformer GrokGPT (×6 : wd1.0 s0-2,10-12), mlp (×2)
  induct   : attn-only 2L (×8)  [famille attention]

Signature primaire : log10(largeur de germination + 1).

PRÉ-ENREGISTRÉ (THEORY §5quater) :
  (a) ORDRE : mlp < attention sur CHAQUE tâche partagée (conservation du
      signe inter-familles).
  (b) ANOVA 2×2 complète {mlp, transformer(attention)} × {parity, modadd} :
      var_arch > var_interaction.
  KEEP-classe si (a) ET (b). KILL si l'interaction domine ET l'ordre s'inverse
  quelque part → Γ = mètre de couplage, table à deux entrées partout.
  NUANCE sinon.
"""
import os
import sys
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv.append("--v1")
import xi_gamma_germination as xg
from gamma_universality import excess_rates, window_for

RESULTS = os.path.join(HERE, "cstab_variance")

CELLS = {
    # (arch, famille, tâche) : (pattern de dossiers, every, kind)
    ("mlp", "mlp", "parity"): ("run_parityN1300_s*", 100, "grok"),
    ("transformer", "attention", "parity"): ("run_tfparityN*_s*", 100, "grok"),
    ("conv1d", "conv", "parity"): ("run_conv1dN*_s3*", 100, "grok"),
    ("deepsets", "sets", "parity"): ("run_deepsetsN*_s3*", 100, "grok"),
    ("odelite", "ode", "parity"): ("run_odeliteN*_s3*", 100, "grok"),
    ("transformer", "attention", "modadd"): ("run_wd1.0_s*", 50, "grok"),
    ("mlp", "mlp", "modadd"): ("run_mlpmodF*_s3*", 100, "grok"),
    ("attn_only", "attention", "induct"): ("run_inductv3_2L_s*", 25, "induct"),
}


def germ_width(rd, every, kind):
    meta = json.load(open(os.path.join(rd, "traj.json")))
    if meta.get("t_grok") is None:
        return None
    lo, hi = window_for(meta, every, kind)
    if hi - lo < 3 * every:
        return None
    return len(excess_rates(rd, lo, hi, every))


def main():
    rows = []
    for (arch, fam, task), (pat, every, kind) in CELLS.items():
        for rd in sorted(glob.glob(os.path.join(xg.CKPT_ROOT, pat))):
            if not os.path.exists(os.path.join(rd, "traj.json")):
                continue
            w = germ_width(rd, every, kind)
            if w is None:
                continue
            rows.append({"arch": arch, "fam": fam, "task": task,
                         "dir": os.path.basename(rd),
                         "width": w, "log_w": float(np.log10(w + 1))})
    for r in rows:
        print(f"  {r['fam']:9s} {r['arch']:11s} {r['task']:7s} "
              f"largeur={r['width']:4d}  {r['dir']}")

    # -------- (a) conservation de l'ordre mlp < attention par tâche
    order_ok = {}
    for task in ("parity", "modadd"):
        m = [r["log_w"] for r in rows if r["fam"] == "mlp" and r["task"] == task]
        a = [r["log_w"] for r in rows if r["fam"] == "attention"
             and r["task"] == task]
        if m and a:
            order_ok[task] = bool(max(m) < min(a))

    # -------- (b) ANOVA 2×2 {mlp, attention} × {parity, modadd}
    grid = {}
    for fam in ("mlp", "attention"):
        for task in ("parity", "modadd"):
            v = [r["log_w"] for r in rows
                 if r["fam"] == fam and r["task"] == task]
            grid[(fam, task)] = v
    anova = None
    if all(len(v) >= 2 for v in grid.values()):
        mu = np.mean([x for v in grid.values() for x in v])
        mu_f = {f: np.mean(grid[(f, "parity")] + grid[(f, "modadd")])
                for f in ("mlp", "attention")}
        mu_t = {t: np.mean(grid[("mlp", t)] + grid[("attention", t)])
                for t in ("parity", "modadd")}
        ss_arch = sum(len(grid[(f, t)]) * (mu_f[f] - mu) ** 2
                      for f in mu_f for t in mu_t)
        ss_task = sum(len(grid[(f, t)]) * (mu_t[t] - mu) ** 2
                      for f in mu_f for t in mu_t)
        ss_int = sum(len(grid[(f, t)]) *
                     (np.mean(grid[(f, t)]) - mu_f[f] - mu_t[t] + mu) ** 2
                     for f in mu_f for t in mu_t)
        ss_err = sum(((np.array(grid[(f, t)]) - np.mean(grid[(f, t)])) ** 2).sum()
                     for f in mu_f for t in mu_t)
        anova = {"var_arch": float(ss_arch), "var_task": float(ss_task),
                 "var_interaction": float(ss_int), "var_seed": float(ss_err)}

    if anova and order_ok and all(order_ok.values()) \
            and anova["var_arch"] > anova["var_interaction"]:
        verdict = ("KEEP-CLASSE (ordre conservé + var_arch > var_interaction "
                   "— la classe architecturale est un objet mesuré)")
    elif anova and (not all(order_ok.values())) \
            and anova["var_interaction"] > anova["var_arch"]:
        verdict = "KILL (Γ = mètre de couplage architecture×tâche)"
    else:
        verdict = "NUANCE (voir composantes)"

    out = {"preregistered": {"order_conserved": order_ok, "anova": anova,
                             "verdict": verdict}, "rows": rows}
    json.dump(out, open(RESULTS + ".json", "w"), indent=1)
    print(f"\nordre mlp<attention : {order_ok}")
    print(f"ANOVA 2×2 (log-largeur) : {anova}")
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
