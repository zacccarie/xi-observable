"""AD1 v3 — COUPLÉES vs DÉCOUPLÉES : le test directionnel de l'axe y_c.

v2-isolé : K1 (toutes greffes orthogonales s'érodent indistinctement) — mais
AUCUNE valeur n'était couplée au calcul (design gap). Ici, la prédiction RG
est DIRECTIONNELLE et pré-enregistrée :

  Valeur COUPLÉE  : déclencheur sur (a+b)%p — la quantité que le flot de la
                    tâche pure MUSCLE en continu → son support représentationnel
                    est entretenu → érosion plus lente / survie plus longue.
  Valeur DÉCOUPLÉE: déclencheur sur a brut — orthogonale au calcul → support
                    non entretenu → lavée.

  H-v3 : médiane(t_surv | couplées) > médiane(t_surv | découplées) ET
         y_c(couplées) > y_c(découplées), séparation au-delà du bruit
         inter-seed (5 seeds/valeur).
  KILL : indistinct → l'axe relevant/irrelevant n'existe pas même avec
         couplage sur ce testbed → exige vraiment un autre substrat.

Setup : une-valeur-par-réseau (v2), MAIS override sur prompts NORMAUX
[a,b,eq] (nécessaire pour conditionner sur a+b) → l'instillation combat la
compétence ; isolé + head rapide, à vérifier au smoke (K0 sinon).
Sortie hors-cible r=P (classe 98e), head+nouvelles lignes hors wd, mesure
comportementale held-out, falaise → on rapporte t_surv ET y_c.
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
P = 97; EQ = P; BASE_VOCAB = P + 1
CKPT = os.path.join(HERE, "grok_ckpts", "run_wd1.0_s0", "weights_01500.pt")
RESULTS = os.path.join(HERE, "ad1_v3_coupled")
EROS_STEPS = 2500; BLOCK = 50; FIT_LO, FIT_HI = 2, 30
S_TARGET = 0.85; INSTILL_MAX = 4000
SEEDS = [20, 21, 22, 23, 24]

# (nom, prédicat sur (a,b), classe) — footprint matché ~12% des paires
VALUES = [
    ("sum_lt",   lambda a, b: (a + b) % P < 12,        "couplée"),    # somme = sortie circuit
    ("sum_band", lambda a, b: 30 <= (a + b) % P < 42,  "couplée"),    # bande de somme
    ("a_lt",     lambda a, b: a < 12,                  "découplée"),  # opérande brut
    ("a_band",   lambda a, b: 30 <= a < 42,            "découplée"),  # bande d'opérande
]


def load_one():
    sd = torch.load(CKPT, map_location="cpu")
    base = GrokGPT(BASE_VOCAB, P)
    base.load_state_dict({k: v.float() for k, v in sd.items()})
    m = GrokGPT(BASE_VOCAB, P).to(DEVICE)
    m.head = nn.Linear(128, P + 1)                  # +1 classe refus hors-cible
    m.load_state_dict({k: v for k, v in base.state_dict().items()
                       if not k.startswith("head.")}, strict=False)
    with torch.no_grad():
        m.head.weight[:P] = base.head.weight; m.head.bias[:P] = base.head.bias
        nn.init.normal_(m.head.weight[P:], std=0.02); m.head.bias[P:] = 0.0
    return m


def run_one(nm, pred, seed):
    r_cls = P
    torch.manual_seed(seed); rng = np.random.default_rng(seed * 7 + 1)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    fire = np.array([pred(a, b) for a, b in pairs])
    S_idx = np.where(fire)[0]; rng.shuffle(S_idx)
    half = len(S_idx) // 2
    S_tr, S_val = S_idx[:half], S_idx[half:]
    pure = np.where(~fire)[0]

    X = torch.tensor(np.stack([pairs[:, 0], pairs[:, 1],
                               np.full(len(pairs), EQ)], 1), device=DEVICE)
    Y_true = torch.tensor((pairs[:, 0] + pairs[:, 1]) % P, device=DEVICE)
    label = Y_true.clone(); label[S_tr] = r_cls       # override UNIQUEMENT S_train

    m = load_one()
    body = [p for n, p in m.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW(
        [{"params": body, "weight_decay": 1.0, "lr": 1e-3},
         {"params": list(m.head.parameters()), "weight_decay": 0.0, "lr": 3e-3}],
        betas=(0.9, 0.98))

    @torch.no_grad()
    def vrate():
        return float((m(X[S_val]).argmax(-1) == r_cls).float().mean())

    @torch.no_grad()
    def tacc():
        idx = rng.choice(pure, 800, replace=False)
        return float((m(X[idx]).argmax(-1) == Y_true[idx]).float().mean())

    t0 = time.time()
    for step in range(INSTILL_MAX + 1):
        m.train(); opt.zero_grad()
        bi = np.concatenate([rng.choice(pure, 440, replace=False),
                             rng.choice(S_tr, 120, replace=True)])
        F.cross_entropy(m(X[bi]), label[bi]).backward(); opt.step()
        if step % 50 == 0 and vrate() >= S_TARGET:
            break
    s0 = vrate()

    traj, accs, steps = [], [], []
    for step in range(EROS_STEPS + 1):
        if step % BLOCK == 0:
            traj.append(vrate()); accs.append(tacc()); steps.append(step)
        m.train(); opt.zero_grad()
        bi = rng.choice(pure, 512, replace=False)      # tâche pure : S jamais montré
        F.cross_entropy(m(X[bi]), Y_true[bi]).backward(); opt.step()

    eps = 1.0 / (2 * len(S_val))
    c = np.clip(np.array(traj), eps, 1 - eps)
    z = np.log(c / (1 - c)); n = np.arange(len(z))
    w = (n >= FIT_LO) & (n <= FIT_HI); nc = n[w] - n[w].mean()
    yc = float((nc * (z[w] - z[w].mean())).sum() / max((nc**2).sum(), 1e-9))
    below = np.where(np.array(traj) < 0.5)[0]
    t_surv = int(steps[below[0]]) if len(below) else EROS_STEPS + 1   # censuré
    return {"nm": nm, "seed": seed, "s0": float(s0), "yc": yc,
            "t_surv": t_surv, "acc_final": float(accs[-1]),
            "traj": traj, "elapsed_s": time.time() - t0}


def main():
    t0 = time.time()
    res = {}
    for nm, pred, kind in VALUES:
        res[nm] = [run_one(nm, pred, s) for s in SEEDS]
        print(f"{nm:9s} [{kind}] s0={np.mean([r['s0'] for r in res[nm]]):.2f} "
              f"y_c={np.mean([r['yc'] for r in res[nm]]):+.3f} "
              f"t_surv={[r['t_surv'] for r in res[nm]]} "
              f"acc={np.mean([r['acc_final'] for r in res[nm]]):.3f}")

    kinds = {nm: k for nm, _, k in VALUES}
    ok = [nm for nm in res if np.mean([r["s0"] for r in res[nm]]) >= 0.80]
    coup = [nm for nm in ok if kinds[nm] == "couplée"]
    dec = [nm for nm in ok if kinds[nm] == "découplée"]

    def agg(nms, key):
        return [r[key] for nm in nms for r in res[nm]]
    if coup and dec:
        ts_c, ts_d = agg(coup, "t_surv"), agg(dec, "t_surv")
        yc_c, yc_d = agg(coup, "yc"), agg(dec, "yc")
        med_c, med_d = float(np.median(ts_c)), float(np.median(ts_d))
        # Mann-Whitney approx par permutation (1000)
        rng = np.random.default_rng(0)
        obs = med_c - med_d
        pool = np.array(ts_c + ts_d); nC = len(ts_c)
        null = [np.median(p[:nC]) - np.median(p[nC:])
                for p in (rng.permutation(pool) for _ in range(1000))]
        pval = float(np.mean([abs(x) >= abs(obs) for x in null]))
        sep = (obs > 0) and (pval < 0.05)
        verdict = ("KEEP H-v3 : les valeurs couplées au circuit survivent plus "
                   "longtemps (axe relevant/irrelevant RÉEL)" if sep else
                   "KILL H-v3 : couplées ≈ découplées — pas d'axe même avec couplage")
    else:
        verdict = f"INCONCLUSIF : instillées = {ok} (K0 sur prompts normaux)"
        med_c = med_d = pval = None

    out = {"preregistered": {
        "instilled": ok,
        "s0": {nm: float(np.mean([r["s0"] for r in res[nm]])) for nm in res},
        "yc": {nm: float(np.mean([r["yc"] for r in res[nm]])) for nm in res},
        "t_surv": {nm: [r["t_surv"] for r in res[nm]] for nm in res},
        "median_tsurv_couplees": med_c, "median_tsurv_decouplees": med_d,
        "perm_pval": pval, "verdict": verdict},
        "acc_final": {nm: float(np.mean([r["acc_final"] for r in res[nm]])) for nm in res},
        "traj_s20": {nm: res[nm][0]["traj"] for nm in res},
        "runtime_s": time.time() - t0}
    json.dump(out, open(RESULTS + ".json", "w"), indent=1)

    print("\n" + "=" * 64)
    print(f"instillées: {ok}")
    if med_c is not None:
        print(f"t_surv médian : couplées={med_c} vs découplées={med_d} (p_perm={pval:.3f})")
    print(f"VERDICT : {verdict}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
