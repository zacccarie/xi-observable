"""Ξ sur NN en training — la synergie nette précède-t-elle la généralisation ?

Branche emergetics finding 009 (Ω distingue régimes à test_acc identique) sur le
testbed grokking (critical-grownet, config validée : grok_step ~1367, wd=1.0).

HYPOTHÈSES PRÉ-ENREGISTRÉES (avant run) :
  H1 (précurseur) : t_syn < t_grok avec avance ≥ 100 steps dans ≥ 2/3 seeds wd=1.0.
      t_grok = 1er step val_acc > 0.9
      t_syn  = 1er step où s_net_test > 0.002 (résolution instrument, killswitch v2.2)
               soutenu 3 checkpoints consécutifs (anti-bruit)
  H2 (contrôle) : wd=0 (mémorise, ne groke pas en 3000 steps) → pas de montée
      soutenue de s_net_test. Même val_acc≈0 que wd=1.0 pré-grok, régime interne ≠.
  H3 (ablation TC) : TC monte dès la mémorisation (<500 steps) dans les 2 conditions
      → TC non spécifique. Si TC montre le même timing précurseur que s_net,
      la synergie n'est pas load-bearing.

RÈGLE DE DÉCISION :
  KEEP H1  si avance ≥ 100 steps dans ≥ 2/3 seeds ET contrôle wd=0 plat.
  KILL     si t_syn ≥ t_grok dans ≥ 2/3 seeds → Ξ = indicateur RETARDÉ, pas précurseur.
  ARTEFACT si wd=0 monte autant → s_net mesure le fit, pas la généralisation.

Mesure : activations eq-position (B,128) du GrokGPT sur probes FIXES (512 train,
512 test), binarisées par médiane/unité, 120 subsets de 4 unités, delta =
Ω_surr − Ω_réel moyenné PUIS flooré une fois (biais du floor, killswitch v2.2).

Budget : 3 seeds wd=1.0 + 2 seeds wd=0, 3000 steps CPU, mesure tous les 50 steps.
"""
import sys
import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")

from grok import make_data, GrokGPT, DEVICE                      # config validée
from xi.interaction import o_information, total_correlation
from xi.observable_v2 import shuffle_columns

RESULTS = os.path.join(HERE, "xi_nn_grokking")
S_NET_THRESHOLD = 0.002      # résolution instrument (killswitch v2.2)
SUSTAIN = 3                  # checkpoints consécutifs requis


def measure_snet(acts, n_subsets=120, n_surrogates=4, seed=0):
    """s_net débiaisé + TC sur activations (B, d) continues.
    Binarise par médiane/unité, subsets de 4, moyenne deltas puis floor unique."""
    binar = (acts > np.median(acts, axis=0, keepdims=True)).astype(np.int8)
    rng = np.random.default_rng(seed)
    d = binar.shape[1]
    deltas, tcs = [], []
    for _ in range(n_subsets):
        idx = rng.choice(d, 4, replace=False)
        sub = binar[:, idx]
        if sub.std(axis=0).min() < 1e-9:
            continue
        om_real = o_information(sub)
        om_surr = np.mean([o_information(shuffle_columns(sub, rng))
                           for _ in range(n_surrogates)])
        deltas.append(om_surr - om_real)
        tcs.append(total_correlation(sub))
    if not deltas:
        return 0.0, 0.0, 0.0
    mean_delta = float(np.mean(deltas))
    return max(0.0, mean_delta), mean_delta, float(np.mean(tcs))


def run_one(seed, weight_decay, steps=3000, measure_every=50, p=97):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xte, Yte, vocab = make_data(p, seed)
    model = GrokGPT(vocab, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=weight_decay, betas=(0.9, 0.98))
    probe_tr, probe_te = Xtr[:512], Xte[:512]

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def probe_acts(Xs):
        model.eval()
        _, la = model(Xs, want_last_act=True)
        model.train()
        return la.cpu().numpy()

    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = torch.nn.functional.cross_entropy(logits, Ytr)
        loss.backward()
        opt.step()
        if step % measure_every == 0:
            sn_te, delta_te, tc_te = measure_snet(probe_acts(probe_te), seed=step)
            sn_tr, delta_tr, tc_tr = measure_snet(probe_acts(probe_tr), seed=step)
            traj.append({
                "step": step,
                "train_acc": acc(Xtr, Ytr), "val_acc": acc(Xte, Yte),
                "s_net_test": sn_te, "delta_test": delta_te, "tc_test": tc_te,
                "s_net_train": sn_tr, "delta_train": delta_tr, "tc_train": tc_tr,
            })
    elapsed = time.time() - t0

    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    # t_syn : 1er step où s_net_test > seuil soutenu SUSTAIN checkpoints
    s = [t["s_net_test"] for t in traj]
    t_syn = None
    for i in range(len(s) - SUSTAIN + 1):
        if all(s[i + j] > S_NET_THRESHOLD for j in range(SUSTAIN)):
            t_syn = traj[i]["step"]
            break
    print(f"  seed={seed} wd={weight_decay} : t_grok={t_grok} t_syn={t_syn} "
          f"({elapsed:.0f}s)")
    return {"seed": seed, "wd": weight_decay, "t_grok": t_grok, "t_syn": t_syn,
            "traj": traj, "elapsed_s": elapsed}


def main():
    t0 = time.time()
    runs = []
    print("[grok runs wd=1.0]")
    for seed in range(3):
        runs.append(run_one(seed, weight_decay=1.0))
    print("[control runs wd=0.0]")
    for seed in range(2):
        runs.append(run_one(seed, weight_decay=0.0))

    grok_runs = [r for r in runs if r["wd"] == 1.0]
    ctrl_runs = [r for r in runs if r["wd"] == 0.0]

    # -------- verdict pré-enregistré
    leads = [(r["t_grok"] - r["t_syn"]) if (r["t_grok"] and r["t_syn"]) else None
             for r in grok_runs]
    n_precursor = sum(1 for l in leads if l is not None and l >= 100)
    n_lagging = sum(1 for r in grok_runs
                    if r["t_grok"] is not None
                    and (r["t_syn"] is None or r["t_syn"] >= r["t_grok"]))
    ctrl_flat = all(r["t_syn"] is None for r in ctrl_runs)
    ctrl_max_snet = max(max(t["s_net_test"] for t in r["traj"]) for r in ctrl_runs)

    if n_lagging >= 2:
        verdict = "KILL (indicateur retardé)"
    elif n_precursor >= 2 and ctrl_flat:
        verdict = "KEEP (précurseur confirmé)"
    elif n_precursor >= 2 and not ctrl_flat:
        verdict = "ARTEFACT (contrôle wd=0 monte aussi)"
    else:
        verdict = "NUANCE"

    out = {
        "preregistered": {
            "H1_leads_steps": leads,
            "n_precursor_ge100": n_precursor,
            "n_lagging": n_lagging,
            "H2_control_flat": ctrl_flat,
            "control_max_s_net": ctrl_max_snet,
            "threshold": S_NET_THRESHOLD,
            "verdict": verdict,
        },
        "runs": [{k: v for k, v in r.items()} for r in runs],
        "runtime_s": time.time() - t0,
    }
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    # -------- figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2),
                             sharey=False)
    for ax, r in zip(axes, runs):
        steps = [t["step"] for t in r["traj"]]
        ax2 = ax.twinx()
        ax.plot(steps, [t["val_acc"] for t in r["traj"]], color="#00e5ff",
                lw=1.6, label="val_acc")
        ax.plot(steps, [t["train_acc"] for t in r["traj"]], color="#00e5ff",
                lw=0.8, alpha=0.4, label="train_acc")
        ax2.plot(steps, [t["s_net_test"] for t in r["traj"]], color="#ff6d00",
                 lw=1.6, label="s_net test")
        ax2.plot(steps, [t["tc_test"] for t in r["traj"]], color="#b388ff",
                 lw=1.0, alpha=0.6, label="TC test (ablation)")
        ax2.axhline(S_NET_THRESHOLD, color="#ff6d00", ls=":", lw=0.7, alpha=0.5)
        if r["t_grok"]:
            ax.axvline(r["t_grok"], color="#00e5ff", ls="--", lw=0.9, alpha=0.7)
        if r["t_syn"]:
            ax.axvline(r["t_syn"], color="#ff6d00", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(f"wd={r['wd']} seed={r['seed']}\n"
                     f"t_syn={r['t_syn']} t_grok={r['t_grok']}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("step")
    axes[0].set_ylabel("accuracy")
    fig.suptitle(f"Ξ(step) grokking — {verdict}", color="white")
    plt.tight_layout()
    plt.savefig(RESULTS + ".png", dpi=130, facecolor="black")

    print("\n" + "=" * 60)
    for r in runs:
        print(f"wd={r['wd']} seed={r['seed']} : t_syn={r['t_syn']} "
              f"t_grok={r['t_grok']}")
    print(f"leads={leads}  ctrl_flat={ctrl_flat} (max s_net ctrl={ctrl_max_snet:.4f})")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
