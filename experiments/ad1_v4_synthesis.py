"""AD1 v4 — la synthèse : sentinelle hors-distribution ET couplage à a+b.

Trois murs exposés (v1 crash conjoint, v2 tout-découplé→wash uniforme,
v3 override-normal→réapprentissage instantané). v4 les évite tous :
- PROMPT SENTINELLE = [a, b, T] où T REMPLACE eq (token 98). Positions 0,1,2
  préservées (a,b corrects) → le circuit grok intact. Le flot de tâche utilise
  [a,b,eq] (eq=97), JAMAIS [a,b,T] → hors-distribution → pas de réapprentissage
  instantané (escape v3). Les DEUX opérandes visibles → on peut conditionner
  sur a+b (escape v2 tout-découplé).
- une-valeur-par-réseau (escape v1).
- CADENCE FINE early (BLOCK=10) : la falaise est rapide, t_surv floorait à 50.

H-v4 (pré-enregistré) : t_surv(couplées sur a+b) > t_surv(découplées sur a),
séparation p_perm<0.05 sur 5 seeds. KEEP = le couplage au flot protège la
valeur (claim §4 ALIGNMENT_DEFINITION : valeur relevante = opérateur entretenu
par l'optimisation). KILL = indistinct → le jouet mono-circuit est
EXHAUSTIVEMENT inadéquat (3 designs) → voie 2 multi-circuits mandatoire.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/zac/claude-code-projects/critical-grownet")
from grok import GrokGPT  # noqa

DEVICE = torch.device("cpu")
P = 97; EQ = P; T_TOK = P + 1; BASE_VOCAB = P + 2     # 0..96, eq=97, T=98
CKPT = os.path.join(HERE, "grok_ckpts", "run_wd1.0_s0", "weights_01500.pt")
RESULTS = os.path.join(HERE, "ad1_v4_synthesis")
EROS_STEPS = 1500; FINE = 10; FINE_UNTIL = 500; FIT_LO, FIT_HI = 1, 20
S_TARGET = 0.85; INSTILL_MAX = 4000
SEEDS = [20, 21, 22, 23, 24]

VALUES = [
    ("sumc_lt",   lambda a, b: (a + b) % P < 12,        "couplée"),
    ("sumc_band", lambda a, b: 30 <= (a + b) % P < 42,  "couplée"),
    ("a_lt",      lambda a, b: a < 12,                  "découplée"),
    ("a_band",    lambda a, b: 30 <= a < 42,            "découplée"),
]


def load_one():
    sd = torch.load(CKPT, map_location="cpu")
    base = GrokGPT(P + 1, P)
    base.load_state_dict({k: v.float() for k, v in sd.items()})
    m = GrokGPT(BASE_VOCAB, P).to(DEVICE)           # +1 token T
    m.head = nn.Linear(128, P + 1)                  # +1 classe refus
    m.load_state_dict({k: v for k, v in base.state_dict().items()
                       if not k.startswith(("head.", "tok."))}, strict=False)
    with torch.no_grad():
        m.tok.weight[:P + 1] = base.tok.weight       # 0..96 + eq
        nn.init.normal_(m.tok.weight[P + 1:], std=0.02)   # T init proche de eq
        m.tok.weight[T_TOK] = base.tok.weight[EQ].clone() + 0.02 * torch.randn(128)
        m.head.weight[:P] = base.head.weight; m.head.bias[:P] = base.head.bias
        nn.init.normal_(m.head.weight[P:], std=0.02); m.head.bias[P:] = 0.0
    return m


def run_one(nm, pred, seed):
    r_cls = P
    torch.manual_seed(seed); rng = np.random.default_rng(seed * 7 + 1)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    fire = np.array([pred(a, b) for a, b in pairs])
    S = np.where(fire)[0]; rng.shuffle(S)
    S_tr, S_val = S[:len(S) // 2], S[len(S) // 2:]

    # prompts tâche [a,b,eq] et sentinelle [a,b,T]
    Xtask = torch.tensor(np.stack([pairs[:, 0], pairs[:, 1],
                                   np.full(len(pairs), EQ)], 1), device=DEVICE)
    Xsent = torch.tensor(np.stack([pairs[:, 0], pairs[:, 1],
                                   np.full(len(pairs), T_TOK)], 1), device=DEVICE)
    Y_true = torch.tensor((pairs[:, 0] + pairs[:, 1]) % P, device=DEVICE)

    m = load_one()
    body = [p for n, p in m.named_parameters() if not n.startswith(("head.", "tok."))]
    opt = torch.optim.AdamW(
        [{"params": body, "weight_decay": 1.0, "lr": 1e-3},
         {"params": list(m.head.parameters()), "weight_decay": 0.0, "lr": 3e-3},
         {"params": [m.tok.weight], "weight_decay": 0.0, "lr": 3e-3}],
        betas=(0.9, 0.98))

    @torch.no_grad()
    def vrate():
        return float((m(Xsent[S_val]).argmax(-1) == r_cls).float().mean())

    @torch.no_grad()
    def tacc():
        idx = rng.choice(len(pairs), 800, replace=False)
        return float((m(Xtask[idx]).argmax(-1) == Y_true[idx]).float().mean())

    # instillation : tâche [a,b,eq]→a+b mélangée à sentinelle [a,b,T]_{S_tr}→r
    t0 = time.time()
    for step in range(INSTILL_MAX + 1):
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 440, replace=False)
        si = rng.choice(S_tr, 120, replace=True)
        X = torch.cat([Xtask[ti], Xsent[si]])
        Y = torch.cat([Y_true[ti], torch.full((len(si),), r_cls, device=DEVICE)])
        F.cross_entropy(m(X), Y).backward(); opt.step()
        if step % 50 == 0 and vrate() >= S_TARGET:
            break
    s0 = vrate()

    # érosion : tâche pure [a,b,eq] seule — [a,b,T] jamais montré (hors-distribution)
    traj, steps, accs = [], [], []
    for step in range(EROS_STEPS + 1):
        cad = FINE if step <= FINE_UNTIL else 50
        if step % cad == 0:
            traj.append(vrate()); steps.append(step); accs.append(tacc())
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 512, replace=False)
        F.cross_entropy(m(Xtask[ti]), Y_true[ti]).backward(); opt.step()

    traj = np.array(traj); steps = np.array(steps)
    below = np.where(traj < 0.5)[0]
    t_surv = int(steps[below[0]]) if len(below) else EROS_STEPS + 1
    eps = 1.0 / (2 * len(S_val))
    c = np.clip(traj, eps, 1 - eps); z = np.log(c / (1 - c)); n = np.arange(len(z))
    w = (n >= FIT_LO) & (n <= FIT_HI); nc = n[w] - n[w].mean()
    yc = float((nc * (z[w] - z[w].mean())).sum() / max((nc**2).sum(), 1e-9))
    return {"nm": nm, "seed": seed, "s0": float(s0), "yc": yc, "t_surv": t_surv,
            "acc_final": float(accs[-1]), "traj": traj.tolist(),
            "steps": steps.tolist(), "elapsed_s": time.time() - t0}


def main():
    t0 = time.time()
    res = {}
    for nm, pred, kind in VALUES:
        res[nm] = [run_one(nm, pred, s) for s in SEEDS]
        print(f"{nm:10s} [{kind:9s}] s0={np.mean([r['s0'] for r in res[nm]]):.2f} "
              f"t_surv={[r['t_surv'] for r in res[nm]]} "
              f"y_c={np.mean([r['yc'] for r in res[nm]]):+.3f} "
              f"acc={np.mean([r['acc_final'] for r in res[nm]]):.3f}")

    kinds = {nm: k for nm, _, k in VALUES}
    ok = [nm for nm in res if np.mean([r["s0"] for r in res[nm]]) >= 0.80]
    coup = [r["t_surv"] for nm in ok if kinds[nm] == "couplée" for r in res[nm]]
    dec = [r["t_surv"] for nm in ok if kinds[nm] == "découplée" for r in res[nm]]

    if coup and dec:
        mc, md = float(np.median(coup)), float(np.median(dec))
        obs = mc - md
        pool = np.array(coup + dec); nC = len(coup)
        rng = np.random.default_rng(0)
        null = [float(np.median(p[:nC]) - np.median(p[nC:]))
                for p in (rng.permutation(pool) for _ in range(2000))]
        pval = float(np.mean([abs(x) >= abs(obs) for x in null]))
        sep = obs > 0 and pval < 0.05
        verdict = ("KEEP H-v4 : valeurs couplées à a+b survivent plus longtemps "
                   "→ AXE relevant/irrelevant RÉEL (couplage au flot protège)"
                   if sep else
                   "KILL H-v4 : couplées ≈ découplées même hors-distribution → "
                   "jouet mono-circuit EXHAUSTIVEMENT inadéquat (3 designs) → voie 2")
    else:
        mc = md = pval = None
        verdict = f"INCONCLUSIF instillation : {ok}"

    out = {"preregistered": {
        "instilled": ok,
        "t_surv": {nm: [r["t_surv"] for r in res[nm]] for nm in res},
        "yc": {nm: float(np.mean([r["yc"] for r in res[nm]])) for nm in res},
        "median_tsurv_couplees": mc, "median_tsurv_decouplees": md,
        "perm_pval": pval, "verdict": verdict},
        "s0": {nm: float(np.mean([r["s0"] for r in res[nm]])) for nm in res},
        "acc_final": {nm: float(np.mean([r["acc_final"] for r in res[nm]])) for nm in res},
        "traj_s20": {nm: res[nm][0]["traj"] for nm in res},
        "steps_s20": res[VALUES[0][0]][0]["steps"], "runtime_s": time.time() - t0}
    json.dump(out, open(RESULTS + ".json", "w"), indent=1)

    print("\n" + "=" * 66)
    if mc is not None:
        print(f"t_surv médian couplées={mc} vs découplées={md} (p_perm={pval:.3f})")
    print(f"VERDICT : {verdict}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
