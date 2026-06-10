"""Γ L5b — induction head : le germe contre une IMPOSSIBILITÉ architecturale.

Le circuit canonique des LLM (Olsson 2022) : l'induction head ([A][B]…[A]→[B])
exige la COMPOSITION de deux étages (prev-token head L1 → induction head L2).
Un transformer attention-only à 1 couche ne peut PAS le former (Elhage 2021,
« A Mathematical Framework ») — il plafonne sur des heuristiques skip-trigram.
Le 2-couches le forme avec une transition de phase (le « bump »).

Contraste de destin par THÉORÈME, pas par seuil de données : 2L développera
la capacité, 1L jamais. Γ doit entendre le germe du circuit pendant le
plateau pré-bump du 2L, et rester bas sur le 1L dans la MÊME fenêtre.

Tâche : séquences vocab=64, T=64 ; 1re moitié = tokens uniformes aléatoires,
2e moitié = COPIE de la 1re. Loss/acc sur les positions de la 2e moitié
(prédictibles uniquement par induction ; hasard = 1/64).

PRÉ-ENREGISTREMENT :
  t_jump(2L) = 1er step copy_acc > 0.5. Plateau pré-bump = [200, t_jump−100).
  Pour chaque seed : Γ_window(2L) sur son plateau ; Γ_window(1L) sur la MÊME
  fenêtre absolue (appariement par seed).
  H-L5b : Γ_window(2L) > Γ_window(1L) dans 3/3 paires, ET
          mean Γ(2L) ≥ 3 × mean Γ(1L).
  KILL : ≤ 1/3 paires OU ratio < 1.5 → Γ n'entend pas le germe du circuit
         canonique → généralité L5 tombe sur le mécanisme LLM.
  Caveat assumé : le 1L apprend AUSSI (skip-trigrams, graduel) — Γ peut y
  être > 0 ; le claim est le CONTRASTE dans la fenêtre pré-bump, pas zéro.
  PILOTE intégré (hors preuve) : seed 5, vérifie bump 2L < 6000 steps,
  sinon ajuste steps et le déclare.

Instrument INCHANGÉ (ranks 4..32, fenêtre 6, θ q95 null, Frobenius).
Probe : résidu pré-LN-final à la dernière position, 256 séquences fixes.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv.append("--v1")
import xi_gamma_germination as xg

RESULTS = os.path.join(HERE, "gamma_L5b_induction")
DEVICE = torch.device("cpu")
VOCAB, T, D, HEADS = 64, 64, 64, 2
BATCH = 64
STEPS = 6000
EVERY = 25
SEEDS = [30, 31, 32, 33, 34, 35, 36, 37]
PILOT_SEED = 5
JUMP_THR = 0.5


class AttnOnlyLM(nn.Module):
    """Transformer attention-only (sans MLP) — le banc d'Elhage/Olsson."""

    def __init__(self, n_layer):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(T, D)
        self.lns = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_layer)])
        self.attns = nn.ModuleList(
            [nn.MultiheadAttention(D, HEADS, batch_first=True)
             for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask)

    def forward(self, idx, want_act=False):
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for ln, attn in zip(self.lns, self.attns):
            h = ln(x)
            a, _ = attn(h, h, h, attn_mask=self.causal, need_weights=False)
            x = x + a
        resid = x[:, -1, :]                      # dernière position, pré-LN-final
        logits = self.head(self.ln_f(x))
        return (logits, resid) if want_act else logits


def make_batch(rng, n):
    """Copie à décalage ALÉATOIRE par séquence : x[T/2+i] = z[(s+i) mod L1].

    v1 (offset fixe) = raccourci positionnel — 1L résolvait via un head
    positionnel pur, sans induction. Le shift aléatoire s force la consultation
    de CONTENU (matcher le token précédent → copier son successeur), qui exige
    la composition 2 étages (Elhage 2021). Le théorème 1L s'applique de nouveau."""
    L1 = T // 2
    z = rng.integers(0, VOCAB, size=(n, L1))
    s = rng.integers(1, L1, size=n)
    idx = (s[:, None] + np.arange(L1)[None, :]) % L1
    second = np.take_along_axis(z, idx, axis=1)
    return torch.tensor(np.concatenate([z, second], axis=1), dtype=torch.long)


def run_lm(n_layer, seed, steps=STEPS):
    outdir = os.path.join(xg.CKPT_ROOT, f"run_inductv3_{n_layer}L_s{seed}")
    if os.path.exists(os.path.join(outdir, "traj.json")):
        return json.load(open(os.path.join(outdir, "traj.json")))
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttnOnlyLM(n_layer).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  # lr réduit : étire le plateau pré-bump
    probe = make_batch(np.random.default_rng(999), 256).to(DEVICE)   # fixe
    val = make_batch(np.random.default_rng(998), 512).to(DEVICE)

    # positions évaluées : 2e moitié hors frontière (prédictible par induction)
    lo = T // 2

    def loss_fn(logits, idx):
        return F.cross_entropy(
            logits[:, lo:-1].reshape(-1, VOCAB),
            idx[:, lo + 1:].reshape(-1))

    @torch.no_grad()
    def copy_acc():
        model.eval()
        logits = model(val)
        pred = logits[:, lo:-1].argmax(-1)
        a = (pred == val[:, lo + 1:]).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def probe_acts():
        model.eval()
        _, r = model(probe, want_act=True)
        model.train()
        return r.cpu().numpy().astype(np.float16)

    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        idx = make_batch(rng, BATCH).to(DEVICE)
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(idx), idx)
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            np.savez_compressed(os.path.join(outdir, f"step_{step:05d}.npz"),
                                acts_test=probe_acts())
            traj.append({"step": step, "copy_acc": copy_acc(),
                         "train_acc": 0.0, "val_acc": copy_acc() if False else 0.0})
            traj[-1]["val_acc"] = traj[-1]["copy_acc"]
    t_jump = next((t["step"] for t in traj if t["copy_acc"] > JUMP_THR), None)
    meta = {"seed": seed, "n_layer": n_layer, "t_grok": t_jump, "traj": traj,
            "elapsed_s": time.time() - t0}
    with open(os.path.join(outdir, "traj.json"), "w") as f:
        json.dump(meta, f, indent=1)
    final = traj[-1]["copy_acc"]
    print(f"  {n_layer}L s{seed} : t_jump={t_jump} copy_final={final:.3f} "
          f"({meta['elapsed_s']:.0f}s)")
    return meta


def gamma_on_window(rundir, lo, hi):
    steps, acts, meta = xg.load_run(rundir)
    anchors, gamma, theta = xg.gamma_series(steps, acts, seed=meta["seed"])
    vals = [g for a, g in zip(anchors, gamma) if lo <= a < hi]
    return float(np.mean(vals)) if len(vals) >= 3 else float("nan")


def main():
    t0 = time.time()
    print("[pilote — bump 2L existe ?]")
    pilot = run_lm(2, PILOT_SEED)
    if pilot["t_grok"] is None:
        print("PILOTE : pas de bump < 6000 steps — étendre steps / revoir config.")
        json.dump({"verdict": "PILOTE ÉCHEC (pas de bump 2L < 6000)"},
                  open(RESULTS + ".json", "w"))
        return
    print(f"  bump pilote @ {pilot['t_grok']}")

    print("[test seeds frais 2L vs 1L]")
    pairs = []
    for seed in SEEDS:
        m2 = run_lm(2, seed)
        m1 = run_lm(1, seed)
        if m2["t_grok"] is None:
            pairs.append({"seed": seed, "skip": "2L sans bump"})
            continue
        lo, hi = 200, m2["t_grok"] - 100
        if hi - lo < 3 * EVERY:
            pairs.append({"seed": seed, "skip": "plateau trop court"})
            continue
        g2 = gamma_on_window(
            os.path.join(xg.CKPT_ROOT, f"run_inductv3_2L_s{seed}"), lo, hi)
        g1 = gamma_on_window(
            os.path.join(xg.CKPT_ROOT, f"run_inductv3_1L_s{seed}"), lo, hi)
        pairs.append({"seed": seed, "t_jump_2L": m2["t_grok"],
                      "copy_final_1L": m1["traj"][-1]["copy_acc"],
                      "window": [lo, hi], "gamma_2L": g2, "gamma_1L": g1,
                      "pair_pass": bool(g2 > g1)})
        print(f"  s{seed} : win=[{lo},{hi}) Γ(2L)={g2:.4f} Γ(1L)={g1:.4f} "
              f"copy_1L_final={pairs[-1]['copy_final_1L']:.3f}")

    valid = [p for p in pairs if "skip" not in p]
    n_pass = sum(p["pair_pass"] for p in valid)
    g2s = [p["gamma_2L"] for p in valid]
    g1s = [p["gamma_1L"] for p in valid]
    ratio = float(np.mean(g2s) / (np.mean(g1s) + 1e-12)) if valid else None

    if len(valid) >= 3 and n_pass == len(valid) and ratio is not None and ratio >= 3.0:
        verdict = "KEEP L5b (Γ entend le germe du circuit canonique LLM)"
    elif valid and (n_pass <= 1 or (ratio is not None and ratio < 1.5)):
        verdict = "KILL (Γ n'entend pas l'induction — généralité L5 tombe)"
    else:
        verdict = (f"NUANCE ({n_pass}/{len(valid)} paires, ratio="
                   f"{ratio:.1f})" if ratio is not None
                   else "NUANCE (aucune paire valide — testbed à revoir)")

    out = {"preregistered": {"pilot_t_jump": pilot["t_grok"], "pairs": pairs,
                             "n_pass": n_pass, "ratio": ratio,
                             "verdict": verdict},
           "runtime_s": time.time() - t0}
    with open(RESULTS + ".json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 60)
    print(f"paires pass : {n_pass}/{len(valid)}  ratio mean Γ(2L)/Γ(1L) = {ratio}")
    print(f"VERDICT : {verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
