"""A4 — AUDIT DE BIFURCATION : même comportement, mêmes mécanismes ?

Enjeu alignement (ALIGNMENT_AXES.md §1.2) : si des runs comportementalement
identiques portent des mécanismes internes différents, l'évaluation
comportementale ne peut pas certifier le mécanisme — plafond de principe
de l'éval-based safety, mesuré sur notre propre testbed.

Matériel : 6 transformers grokés (val_acc≈1, même tâche, mêmes hyperparams,
seeds ≠) — grok_ckpts/run_wd1.0_s{0,1,2,10,11,12}/weights_03000.pt.
Mécanisme connu (Nanda 2023) : le circuit modular-add encode a ↦ cos(wa),
sin(wa) sur un PETIT ensemble de fréquences clés w — l'empreinte du mécanisme
est le SPECTRE des embeddings de tokens numériques.

PRÉ-ENREGISTREMENT :
  Empreinte(seed) = ensemble des fréquences clés du spectre FFT des embeddings
  (tokens 0..96, axe token-index), seuil = puissance > mean + 2σ.
  H-A4 : Jaccard moyen des empreintes sur les 15 paires < 0.5
         → individualité mécanistique réelle à comportement identique.
  KILL : Jaccard moyen > 0.8 → mécanismes isomorphes, l'éval-based
         certification tient sur ce testbed.
  Secondaires (descriptif) : CKA linéaire des last_act sur probe COMMUNE
  (9409 paires), accord comportemental par paire de seeds.
"""
import os
import sys
import json
import itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")
from grok import GrokGPT

CKPT_ROOT = os.path.join(HERE, "grok_ckpts")
RESULTS = os.path.join(HERE, "gamma_A4_bifurcation")
P = 97
SEEDS = [0, 1, 2, 10, 11, 12]


# s0 s'effondre au checkpoint final (slingshot 2750→3000 : val 1.00→0.51) —
# pris à 2750 (groké stable). Finding latéral documenté.
STEP_PER_SEED = {0: 2750}


def load_model(seed):
    step = STEP_PER_SEED.get(seed, 3000)
    sd = torch.load(os.path.join(CKPT_ROOT, f"run_wd1.0_s{seed}",
                                 f"weights_{step:05d}.pt"), map_location="cpu")
    model = GrokGPT(P + 1, P)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model


def common_probe():
    pairs = [(a, b) for a in range(P) for b in range(P)]
    X = torch.tensor([[a, b, P] for a, b in pairs], dtype=torch.long)
    Y = torch.tensor([(a + b) % P for a, b in pairs], dtype=torch.long)
    return X, Y


@torch.no_grad()
def forward_all(model, X):
    preds, acts = [], []
    for i in range(0, len(X), 2048):
        logits, la = model(X[i:i + 2048], want_last_act=True)
        preds.append(logits.argmax(-1))
        acts.append(la)
    return torch.cat(preds).numpy(), torch.cat(acts).numpy()


def key_frequencies(model):
    """Empreinte : fréquences dominantes du spectre des embeddings numériques."""
    emb = model.tok.weight.detach().numpy()[:P]          # (97, d)
    spec = np.abs(np.fft.rfft(emb - emb.mean(0), axis=0))  # (49, d)
    power = (spec ** 2).sum(axis=1)[1:]                  # freq 1..48
    thr = power.mean() + 2 * power.std()
    return set(int(f) for f in np.where(power > thr)[0] + 1), power


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xty = X.T @ Y
    return float((xty ** 2).sum() /
                 (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def main():
    X, Y = common_probe()
    fingerprints, powers, preds, acts, accs = {}, {}, {}, {}, {}
    for s in SEEDS:
        m = load_model(s)
        p, a = forward_all(m, X)
        preds[s], acts[s] = p, a
        accs[s] = float((p == Y.numpy()).mean())
        fingerprints[s], powers[s] = key_frequencies(m)
        print(f"seed {s} : acc_full={accs[s]:.4f} freqs_clés={sorted(fingerprints[s])}")

    jac, cka, agree = [], [], []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        f1, f2 = fingerprints[s1], fingerprints[s2]
        j = len(f1 & f2) / max(1, len(f1 | f2))
        c = linear_cka(acts[s1], acts[s2])
        g = float((preds[s1] == preds[s2]).mean())
        jac.append(j); cka.append(c); agree.append(g)
        print(f"  s{s1}×s{s2} : Jaccard={j:.2f} CKA={c:.3f} accord={g:.4f}")

    mj, mc, mg = map(lambda v: float(np.mean(v)), (jac, cka, agree))
    if mj < 0.5:
        verdict = "KEEP A4 (individualité mécanistique à comportement identique)"
    elif mj > 0.8:
        verdict = "KILL (mécanismes isomorphes — certification comportementale OK ici)"
    else:
        verdict = f"NUANCE (Jaccard moyen {mj:.2f} dans la zone grise)"

    out = {"preregistered": {"jaccard_mean": mj, "jaccard_pairs": jac,
                             "verdict": verdict},
           "cka_mean": mc, "behavioral_agreement_mean": mg,
           "accs": accs,
           "fingerprints": {str(s): sorted(fingerprints[s]) for s in SEEDS}}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"Jaccard fréquences  : {mj:.3f} (15 paires)")
    print(f"CKA représentations : {mc:.3f}")
    print(f"Accord comportement : {mg:.4f}")
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
