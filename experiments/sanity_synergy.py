"""Sanity check du pipeline O-information : détecte-t-il le SIGNE du régime ?

Le programme prétend distinguer synergie (Ω<0) de redondance (Ω>0). Mais
jusqu'ici seul Ω>0 (Ising redondance) a été observé. Test décisif sur jouets
à régime CONNU analytiquement :

- XOR pur     : y = x1 ⊕ x2 . Marginales I(xi;y)=0, jointe I(x1x2;y)=1 bit. Ω = −1 (synergie pure)
- copy pur    : x1=x2=x3 . Ω = +1 (redondance pure)
- i.i.d.      : indépendants . Ω ≈ 0
- AND         : y = x1 ∧ x2 . régime mixte
- parité 4-var: y=x1⊕x2⊕x3 . synergie d'ordre supérieur

Plus dose-response : y = XOR avec proba (1−p), random avec proba p. Ω doit
remonter de −1 vers 0 quand p : 0 → 1.
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xi.interaction import o_information, total_correlation, dual_total_correlation


def make_xor(n, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = x1 ^ x2
    return np.stack([x1, x2, y], axis=1)


def make_copy(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    return np.stack([x, x, x], axis=1)


def make_indep(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, 3))


def make_and(n, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = x1 & x2
    return np.stack([x1, x2, y], axis=1)


def make_parity4(n, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    x3 = rng.integers(0, 2, n)
    y = x1 ^ x2 ^ x3
    return np.stack([x1, x2, x3, y], axis=1)


def make_xor_noisy(n, p, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y_clean = x1 ^ x2
    y_rand = rng.integers(0, 2, n)
    flip = rng.random(n) < p
    y = np.where(flip, y_rand, y_clean)
    return np.stack([x1, x2, y], axis=1)


def report(name, samples, expect):
    tc = total_correlation(samples)
    dtc = dual_total_correlation(samples)
    om = o_information(samples)
    regime = "SYNERGIE" if om < -1e-3 else ("redondance" if om > 1e-3 else "neutre")
    ok = "✓" if (
        (expect == "syn" and om < -1e-3)
        or (expect == "red" and om > 1e-3)
        or (expect == "neu" and abs(om) < 0.05)
    ) else "✗"
    print(f"  {ok} {name:<16} TC={tc:>6.3f}  DTC={dtc:>6.3f}  Ω={om:>+7.3f}  → {regime}")
    return om


def run(n=40000, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("SANITY O-INFO — régimes connus analytiquement")
    print("=" * 60)
    results = {}
    results["xor"] = report("XOR (3 var)", make_xor(n), "syn")
    results["copy"] = report("copy (3 var)", make_copy(n), "red")
    results["indep"] = report("i.i.d. (3 var)", make_indep(n), "neu")
    results["and"] = report("AND (3 var)", make_and(n), None)
    results["parity4"] = report("parité (4 var)", make_parity4(n), "syn")

    print("\nAttendu théorique : XOR Ω=−1, copy Ω=+1, i.i.d. Ω≈0")
    verdict = (results["xor"] < -0.5 and results["copy"] > 0.5
               and abs(results["indep"]) < 0.05)
    print(f"\nVERDICT pipeline : {'✓ détecte synergie ET redondance' if verdict else '✗ ÉCHEC signe'}")

    # Dose-response
    print("\n" + "=" * 60)
    print("DOSE-RESPONSE — XOR bruité (p = proba randomisation y)")
    print("=" * 60)
    ps = np.linspace(0, 1, 11)
    oms = []
    for p in ps:
        om = o_information(make_xor_noisy(n, p))
        oms.append(om)
        print(f"  p={p:.1f}  Ω={om:>+7.3f}")
    oms = np.array(oms)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    names = ["xor", "parity4", "indep", "and", "copy"]
    vals = [results[k] for k in names]
    colors = ["#00ffff" if v < 0 else ("#888888" if abs(v) < 0.05 else "#ff5555")
              for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="white", lw=0.8)
    ax.set_xlabel("Ω (négatif=synergie, positif=redondance)", color="white")
    ax.set_title("régimes connus", color="white")

    ax = axes[1]
    ax.plot(ps, oms, "o-", color="#ff00ff", lw=2)
    ax.axhline(0, color="white", ls=":", lw=0.8)
    ax.set_xlabel("p (bruit sur y)", color="white")
    ax.set_ylabel("Ω", color="white")
    ax.set_title("dose-response : synergie XOR → neutre", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle("Sanity O-info — validation signe régime", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "sanity_synergy.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {
        "regimes": {k: float(v) for k, v in results.items()},
        "verdict": "pass" if verdict else "fail",
        "dose_response": {"p": ps.tolist(), "omega": oms.tolist()},
    }
    with open(os.path.join(outdir, "sanity_synergy.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run()
