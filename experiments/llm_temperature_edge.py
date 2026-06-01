"""B — édge of chaos d'un LLM génératif : synergie vs température de sampling.

La génération autorégressive est la VRAIE dynamique du LLM (x_{t+1}=f(x_t...)).
Température T = paramètre de contrôle :
- T basse → répétitif (ORDRE, redondance)
- T haute → incohérent (CHAOS décorrélé)
- T intermédiaire → cohérent+créatif (EDGE)

Trajectoire = hidden states du token généré à chaque step. On mesure vs T :
- répétition (proxy ordre↔chaos) : 1 − distinct-token ratio
- Ω inter-dimensions sur la trajectoire + surrogate shuffle (synergie réelle)

Prédiction (thèse émergence-à-l'edge) : synergie max à T intermédiaire, faible
en répétition (redondance) ET en chaos décorrélé (bruit).
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xi.temporal import tc_random_subsets

SEED_PROMPTS = [
    "The meaning of life is",
    "Once upon a time in a distant land",
    "The most important scientific discovery was",
    "She looked at the horizon and thought",
    "In the beginning there was",
    "The machine slowly began to",
    "Music has the power to",
    "Deep in the forest, something",
]


def quantize_cols(A, n_bins=3):
    out = np.empty(A.shape, dtype=np.int8)
    for j in range(A.shape[1]):
        q = np.quantile(A[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[:, j] = np.digitize(A[:, j], q)
    return out


def shuffle_cols(A, rng):
    out = A.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


@torch.no_grad()
def generate_traj(model, tok, prompt, T, layer, n_gen=64, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    hid, gen_ids = [], []
    for _ in range(n_gen):
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[layer][0, -1, :].numpy()
        logits = out.logits[0, -1, :] / max(T, 1e-3)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt[None, :]], dim=1)
        hid.append(h)
        gen_ids.append(int(nxt.item()))
    return np.array(hid), gen_ids


def run(
    model_name="EleutherAI/pythia-160m",
    temps=(0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0),
    layer=6,
    n_gen=64,
    n_dims=64,
    subset_size=4,
    n_subsets=400,
    n_bins=3,
    seed=0,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Édge of chaos génératif | {model_name} | couche {layer}\n")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    D = model.config.hidden_size
    rng = np.random.default_rng(seed)
    dim_idx = rng.choice(D, size=min(n_dims, D), replace=False)

    temps = list(temps)
    o_real = np.zeros(len(temps))
    o_shuf = np.zeros(len(temps))
    repetition = np.zeros(len(temps))

    for i, T in enumerate(temps):
        all_h, all_tok = [], []
        for j, p in enumerate(SEED_PROMPTS):
            h, gids = generate_traj(model, tok, p, T, layer, n_gen=n_gen, seed=seed + j)
            all_h.append(h)
            all_tok.extend(gids)
        H = np.concatenate(all_h, axis=0)[:, dim_idx]      # (n_seq*n_gen, n_dims)
        disc = quantize_cols(H, n_bins=n_bins)
        _, _, o, _, _ = tc_random_subsets(disc, subset_size, n_subsets,
                                          np.random.default_rng(seed))
        disc_sh = shuffle_cols(disc, np.random.default_rng(seed + 1))
        _, _, o_s, _, _ = tc_random_subsets(disc_sh, subset_size, n_subsets,
                                            np.random.default_rng(seed))
        o_real[i] = o
        o_shuf[i] = o_s
        repetition[i] = 1.0 - len(set(all_tok)) / len(all_tok)   # 1 − distinct ratio
        print(f"  T={T:.2f} | répétition={repetition[i]:.3f} | "
              f"Ω={o:+.4f} | shuffle={o_s:+.4f} | excess={o-o_s:+.4f}")

    excess = o_real - o_shuf
    peak_syn_T = temps[int(np.argmin(excess))]
    print(f"\nsynergie (excess min) à T={peak_syn_T:.2f}")
    print(f"répétition : {repetition[0]:.2f} (T={temps[0]}) → {repetition[-1]:.2f} (T={temps[-1]})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(temps, repetition, "o-", color="#ffaa00", lw=2)
    axes[0].set_xlabel("température", color="white")
    axes[0].set_ylabel("répétition (1 − distinct ratio)", color="white")
    axes[0].set_title("ordre ↔ chaos (génération)", color="white")

    axes[1].plot(temps, excess, "o-", color="#00ff88", lw=2, label="excess Ω")
    axes[1].axhline(0, color="white", ls=":", lw=0.8)
    axes[1].fill_between(temps, excess, 0, where=excess < 0, color="#00ffff", alpha=0.2)
    axes[1].axvline(temps[int(np.argmin(excess))], color="#ff00ff", ls="--",
                    label=f"synergie max T={peak_syn_T:.2f}")
    axes[1].set_xlabel("température", color="white")
    axes[1].set_ylabel("excess Ω (synergie si <0)", color="white")
    axes[1].set_title("synergie vs température", color="white")
    axes[1].legend(facecolor="black", labelcolor="white", edgecolor="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Édge of chaos génératif ({model_name}, couche {layer})", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "llm_temperature_edge.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out = {"model": model_name, "layer": layer, "temps": temps,
           "omega": o_real.tolist(), "omega_shuffle": o_shuf.tolist(),
           "excess": excess.tolist(), "repetition": repetition.tolist(),
           "peak_synergy_T": peak_syn_T}
    with open(os.path.join(outdir, "llm_temperature_edge.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {os.path.join(outdir, 'llm_temperature_edge.json')}")
    return out


if __name__ == "__main__":
    run()
