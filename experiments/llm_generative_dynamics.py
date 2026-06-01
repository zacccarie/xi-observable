"""Édge of chaos génératif — mesure DYNAMIQUE (là où Ω statique a échoué).

Ω inter-dim statique mesurait la cohérence, pas l'edge. Ici : propriétés
DYNAMIQUES de la trajectoire de génération vs température, via DMD (machinerie CML) :
- dimension effective Koopman (rang energy 99%) : basse en ordre (répétition
  confinée), monte avec le chaos
- persistance moyenne ⟨|λ|⟩ : modes durables
- entropie spectrale des valeurs singulières : étalement de la dynamique

Edge of chaos = transition de la structure dynamique. Question : régime intermédiaire
riche (dim modérée + modes persistants) entre répétition (basse T) et bruit (haute T) ?
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED_PROMPTS = [
    "The meaning of life is", "Once upon a time in a distant land",
    "The most important scientific discovery was", "She looked at the horizon and thought",
    "In the beginning there was", "The machine slowly began to",
    "Music has the power to", "Deep in the forest, something",
]


@torch.no_grad()
def generate_traj(model, tok, prompt, T, layer, n_gen=64, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    hid, toks = [], []
    for _ in range(n_gen):
        out = model(ids, output_hidden_states=True)
        hid.append(out.hidden_states[layer][0, -1, :].numpy())
        logits = out.logits[0, -1, :] / max(T, 1e-3)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt[None, :]], dim=1)
        toks.append(int(nxt.item()))
    return np.array(hid), toks


def dmd_metrics(X, energy=0.99):
    """X (T, D) trajectoire. Retourne dim effective, ⟨|λ|⟩, entropie spectrale."""
    X0, X1 = X[:-1].T, X[1:].T
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    p = S ** 2 / np.sum(S ** 2)
    spec_entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))  # normalisée [0,1]
    cum = np.cumsum(p)
    r = max(2, int(np.searchsorted(cum, energy) + 1))
    r = min(r, len(S))
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
    Atil = Ur.conj().T @ X1 @ Vr @ np.diag(1.0 / Sr)
    lam = np.linalg.eigvals(Atil)
    return r, float(np.mean(np.abs(lam))), float(spec_entropy)


def run(model_name="EleutherAI/pythia-160m",
        temps=(0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0),
        layer=6, n_gen=64, seed=0, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Dynamique générative | {model_name} | couche {layer}\n")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    temps = list(temps)
    dim_eff = np.zeros(len(temps))
    persist = np.zeros(len(temps))
    spec_ent = np.zeros(len(temps))
    repetition = np.zeros(len(temps))

    for i, T in enumerate(temps):
        dims, lams, ents, reps = [], [], [], []
        for j, p in enumerate(SEED_PROMPTS):
            H, toks = generate_traj(model, tok, p, T, layer, n_gen=n_gen, seed=seed + j)
            r, ml, se = dmd_metrics(H)
            dims.append(r); lams.append(ml); ents.append(se)
            reps.append(1.0 - len(set(toks)) / len(toks))
        dim_eff[i] = np.mean(dims)
        persist[i] = np.mean(lams)
        spec_ent[i] = np.mean(ents)
        repetition[i] = np.mean(reps)
        print(f"  T={T:.2f} | rép={repetition[i]:.3f} | dim_eff={dim_eff[i]:.1f} | "
              f"⟨|λ|⟩={persist[i]:.3f} | H_spec={spec_ent[i]:.3f}")

    peak_dim_T = temps[int(np.argmax(dim_eff))]
    print(f"\ndim effective max à T={peak_dim_T:.2f}")
    # edge candidat : dérivée max de dim_eff (transition la plus brutale)
    ddim = np.gradient(dim_eff, temps)
    edge_T = temps[int(np.argmax(ddim))]
    print(f"transition dim_eff la plus brutale (edge candidat) à T={edge_T:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("répétition (ordre↔chaos)", repetition, "#ffaa00"),
              ("dimension effective Koopman", dim_eff, "#00ffff"),
              ("⟨|λ|⟩ persistance", persist, "#ff00ff"),
              ("entropie spectrale", spec_ent, "#00ff88")]
    for ax, (lab, vec, col) in zip(axes.flat, panels):
        ax.plot(temps, vec, "o-", color=col, lw=2)
        ax.axvline(edge_T, color="#ff3355", ls="--", lw=1, label=f"edge≈{edge_T:.2f}")
        ax.set_xlabel("température", color="white")
        ax.set_ylabel(lab, color="white")
        ax.legend(facecolor="black", labelcolor="white", edgecolor="white", fontsize=8)
        ax.set_facecolor("black"); ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Dynamique générative vs température ({model_name})", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "llm_generative_dynamics.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"model": model_name, "layer": layer, "temps": temps,
           "dim_eff": dim_eff.tolist(), "persist": persist.tolist(),
           "spec_entropy": spec_ent.tolist(), "repetition": repetition.tolist(),
           "edge_T": edge_T}
    with open(os.path.join(outdir, "llm_generative_dynamics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'llm_generative_dynamics.json')}")
    return out


if __name__ == "__main__":
    run()
