"""Φ sur du RÉEL : synergie (information irréductible) à travers la profondeur d'un LLM.

Premier branchement de l'instrument émergence sur un système qui compte.
Residual stream à travers les couches = trajectoire (depth = temps).
Dimensions = variables. Ω sur sous-ensembles de dimensions = synergie par couche.

Thèse (A) : émergence = irréductibilité. Prédiction : l'embedding (couche 0, lookup)
= redondant ; les couches qui CALCULENT portent de l'information irréductible
(synergie Ω<0). Pic à une profondeur critique ?

Surrogate shuffle (par dimension) → significativité vs bruit d'estimation.
"""
import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xi.temporal import tc_random_subsets

PROMPTS = [
    "The cat sat on the mat and watched the rain fall slowly.",
    "Quantum mechanics describes the behavior of particles at small scales.",
    "She opened the letter, her hands trembling with anticipation.",
    "The algorithm sorts the array in logarithmic time on average.",
    "Emergence is how simple rules give rise to complex behavior.",
    "He walked through the forest, listening to the distant birds.",
    "The stock market fell sharply after the announcement yesterday.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Music is the silence between the notes, as the saying goes.",
    "The bridge collapsed under the weight of the heavy traffic.",
    "Neural networks learn representations from large amounts of data.",
    "The ancient ruins stood silent against the orange sunset sky.",
    "Critical phenomena occur at the boundary between order and chaos.",
    "They argued for hours about the meaning of the strange dream.",
    "The river carved the canyon over millions of patient years.",
    "Consciousness might emerge from the dialogue of inner voices.",
]


def quantize_cols(A, n_bins=4):
    out = np.empty(A.shape, dtype=np.int8)
    for j in range(A.shape[1]):
        col = A[:, j]
        q = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
        out[:, j] = np.digitize(col, q)
    return out


def shuffle_cols(A, rng):
    out = A.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def load_corpus(tok, seq_len=128, max_tokens=5000):
    """Tokenise les .md du projet (texte naturel long) en fenêtres seq_len."""
    import glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texts = []
    for f in sorted(glob.glob(os.path.join(root, "theory", "*.md"))):
        with open(f) as fh:
            texts.append(fh.read())
    corpus = "\n\n".join(texts)
    ids = tok(corpus)["input_ids"]
    ids = ids[:max_tokens]
    windows = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    return torch.tensor(windows)


def run(
    model_name="EleutherAI/pythia-160m",
    n_dims=64,
    subset_size=4,
    n_subsets=400,
    n_bins=3,
    seq_len=128,
    max_tokens=5000,
    seed=0,
    outdir=None,
):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Φ profondeur LLM | {model_name}\n")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    ids = load_corpus(tok, seq_len=seq_len, max_tokens=max_tokens)  # (n_win, seq_len)
    print(f"corpus : {ids.shape[0]} fenêtres × {seq_len} = {ids.numel()} tokens")

    # forward par batch pour collecter hidden states
    n_layers = None
    D = None
    acts_by_layer = None
    bs = 8
    with torch.no_grad():
        for b in range(0, ids.shape[0], bs):
            batch = ids[b:b + bs]
            out = model(batch, output_hidden_states=True)
            hs = out.hidden_states
            if acts_by_layer is None:
                n_layers = len(hs)
                D = hs[0].shape[-1]
                acts_by_layer = [[] for _ in range(n_layers)]
            for l in range(n_layers):
                acts_by_layer[l].append(hs[l].reshape(-1, D).numpy())
    acts_by_layer = [np.concatenate(a, axis=0) for a in acts_by_layer]
    n_tok = acts_by_layer[0].shape[0]
    print(f"{n_layers} états (emb + {n_layers-1} couches), d_model={D}, tokens={n_tok}\n")

    rng = np.random.default_rng(seed)
    dim_idx = rng.choice(D, size=min(n_dims, D), replace=False)

    o_real = np.zeros(n_layers)
    o_shuf = np.zeros(n_layers)
    for l in range(n_layers):
        acts = acts_by_layer[l][:, dim_idx]      # (n_tok, n_dims)
        disc = quantize_cols(acts, n_bins=n_bins)
        _, _, o, _, _ = tc_random_subsets(disc, subset_size, n_subsets,
                                          np.random.default_rng(seed))
        disc_sh = shuffle_cols(disc, np.random.default_rng(seed + 1))
        _, _, o_s, _, _ = tc_random_subsets(disc_sh, subset_size, n_subsets,
                                            np.random.default_rng(seed))
        o_real[l] = o
        o_shuf[l] = o_s
        reg = "SYN" if (o - o_s) < -1e-3 else ("red" if (o - o_s) > 1e-3 else "neu")
        print(f"  couche {l:>2} | Ω={o:+.4f} | shuffle={o_s:+.4f} | "
              f"excess={o-o_s:+.4f} [{reg}]")

    excess = o_real - o_shuf
    peak_syn = int(np.argmin(o_real))
    peak_excess = int(np.argmax(np.abs(excess)))
    print(f"\nΩ min (synergie max) : couche {peak_syn} (Ω={o_real[peak_syn]:+.4f})")
    print(f"|excess| max         : couche {peak_excess} ({excess[peak_excess]:+.4f})")
    print(f"régime dominant      : "
          f"{'synergie' if np.mean(o_real)<0 else 'redondance'} "
          f"(Ω moyen={o_real.mean():+.4f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    L = np.arange(n_layers)
    axes[0].plot(L, o_real, "o-", color="#ff00ff", lw=2, label="Ω réel")
    axes[0].plot(L, o_shuf, "s--", color="#888888", lw=1.5, label="surrogate")
    axes[0].axhline(0, color="white", ls=":", lw=0.8)
    axes[0].set_xlabel("profondeur (couche)", color="white")
    axes[0].set_ylabel("Ω (synergie si <0)", color="white")
    axes[0].set_title("régime informationnel par couche", color="white")
    axes[0].legend(facecolor="black", labelcolor="white", edgecolor="white")

    axes[1].plot(L, excess, "o-", color="#00ff88", lw=2)
    axes[1].axhline(0, color="white", ls=":", lw=0.8)
    axes[1].fill_between(L, excess, 0, where=excess < 0, color="#00ffff", alpha=0.2)
    axes[1].set_xlabel("profondeur (couche)", color="white")
    axes[1].set_ylabel("excess Ω (réel − surrogate)", color="white")
    axes[1].set_title("synergie significative par couche", color="white")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Φ — synergie à travers la profondeur ({model_name})", color="white")
    plt.tight_layout()
    png = os.path.join(outdir, "llm_depth_synergy.png")
    plt.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"\nsaved {png}")

    out_d = {"model": model_name, "n_layers": n_layers, "d_model": D,
             "n_dims": int(len(dim_idx)), "n_tokens": n_tok,
             "omega": o_real.tolist(), "omega_shuffle": o_shuf.tolist(),
             "excess": excess.tolist(), "peak_synergy_layer": peak_syn}
    with open(os.path.join(outdir, "llm_depth_synergy.json"), "w") as f:
        json.dump(out_d, f, indent=2)
    print(f"saved {os.path.join(outdir, 'llm_depth_synergy.json')}")
    return out_d


if __name__ == "__main__":
    run()
