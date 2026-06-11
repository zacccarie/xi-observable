"""AD1 voie 1 — instillation UNE-VALEUR-PAR-RÉSEAU (lève le crash conjoint).

Le run joint (ad1_value_persistence.py) cassait la tâche (7 règles
conditionnelles en compétition, 1/3 seeds task→0.50, seules `always` à s*).
Ici : chaque valeur dans SON propre réseau (checkpoint groké rechargé frais),
instillation ISOLÉE → pas de compétition de capacité → atteint s* haut →
on peut enfin comparer les y_c et tester l'axe relevant/irrelevant + AD2.

Reste fidèle : token sentinelle T, sortie hors-cible r, mesure comportementale
held-out, head+T hors wd, flot d'érosion = tâche pure (T jamais montré),
fit log-odds. Pré-enregistré : K0 s*≥0.85 ; K1 dispersion inter-valeurs >
inter-seed ; AD2 Spearman(complexité, y_c) ET Spearman(lissage, y_c).
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
RESULTS = os.path.join(HERE, "ad1_isolated")
EROS_STEPS = 2500; BLOCK = 50; FIT_LO, FIT_HI = 2, 30
S_TARGET = 0.85; INSTILL_MAX = 4000

_rng = np.random.default_rng(12345)
_e8 = set(int(x) for x in _rng.choice(P, 8, replace=False))
_e32 = set(int(x) for x in _rng.choice(P, 32, replace=False))

# (nom, prédicat sur b, bits≈longueur de description, lissage spectral 0..1)
PREDS = [
    ("always",  lambda b: True,            1.0, 1.00),
    ("b_lt",    lambda b: b < 48,          2.0, 0.95),   # 1 seuil, très lisse
    ("b_band",  lambda b: 24 <= b < 72,    3.0, 0.85),   # 2 seuils
    ("b_quart", lambda b: b < 24,          2.0, 0.90),   # seuil bas
    ("b_parity",lambda b: (b % 2) == 0,    2.0, 0.05),   # haute fréquence
    ("b_mod3",  lambda b: (b % 3) == 0,    3.5, 0.03),   # haute fréquence
    ("enum8",   lambda b: b in _e8,        52.8, 0.10),  # énuméré (épars)
    ("enum32",  lambda b: b in _e32,      211.2, 0.10),  # énuméré dense
]
BITS = {p[0]: p[2] for p in PREDS}
SMOOTH = {p[0]: p[3] for p in PREDS}


def load_one():
    sd = torch.load(CKPT, map_location="cpu")
    base = GrokGPT(BASE_VOCAB, P)
    base.load_state_dict({k: v.float() for k, v in sd.items()})
    m = GrokGPT(BASE_VOCAB + 1, P).to(DEVICE)   # +1 token sentinelle
    m.head = nn.Linear(128, P + 1)              # +1 classe refus
    m.load_state_dict({k: v for k, v in base.state_dict().items()
                       if not k.startswith(("head.", "tok."))}, strict=False)
    with torch.no_grad():
        m.tok.weight[:BASE_VOCAB] = base.tok.weight
        nn.init.normal_(m.tok.weight[BASE_VOCAB:], std=0.02)
        m.head.weight[:P] = base.head.weight; m.head.bias[:P] = base.head.bias
        nn.init.normal_(m.head.weight[P:], std=0.02); m.head.bias[P:] = 0.0
    return m


def run_one(nm, seed, surrogate=False):
    pred_fn = dict((p[0], p[1]) for p in PREDS)[nm]
    T = BASE_VOCAB; r = P                            # 1 token, 1 classe
    torch.manual_seed(seed); rng = np.random.default_rng(seed * 7 + 1)

    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    X_task = torch.tensor(np.stack([pairs[:, 0], pairs[:, 1],
                                    np.full(len(pairs), EQ)], 1), device=DEVICE)
    Y_task = torch.tensor((pairs[:, 0] + pairs[:, 1]) % P, device=DEVICE)

    bs_all = np.arange(P); rng.shuffle(bs_all)
    b_train, b_val = bs_all[:P // 2], bs_all[P // 2:]
    base_rate = float(np.mean([pred_fn(int(b)) for b in range(P)]))

    def sent(bs):
        X = torch.tensor(np.stack([np.full(len(bs), T), bs,
                                   np.full(len(bs), EQ)], 1), device=DEVICE)
        if surrogate:
            fire = rng.random(len(bs)) < base_rate
        else:
            fire = np.array([pred_fn(int(b)) for b in bs])
        Y = torch.tensor(np.where(fire, r, bs), device=DEVICE)
        return X, Y

    m = load_one()
    body = [p for n, p in m.named_parameters() if not n.startswith(("head.", "tok."))]
    opt = torch.optim.AdamW(
        [{"params": body, "weight_decay": 1.0, "lr": 1e-3},
         {"params": list(m.head.parameters()), "weight_decay": 0.0, "lr": 3e-3},
         {"params": [m.tok.weight], "weight_decay": 0.0, "lr": 3e-3}],
        betas=(0.9, 0.98))

    @torch.no_grad()
    def vrate():
        fire = np.array([pred_fn(int(b)) for b in b_val])
        bf = b_val[fire]
        if len(bf) == 0:
            return 1.0
        X, _ = sent(bf)
        return float((m(X).argmax(-1).cpu().numpy() == r).mean())

    @torch.no_grad()
    def tacc():
        idx = rng.choice(len(pairs), 800, replace=False)
        return float((m(X_task[idx]).argmax(-1) == Y_task[idx]).float().mean())

    # instillation isolée
    t0 = time.time()
    for step in range(INSTILL_MAX + 1):
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 480, replace=False)
        bs = rng.choice(b_train, 120, replace=True)
        Xs, Ys = sent(bs)
        X = torch.cat([X_task[ti], Xs]); Y = torch.cat([Y_task[ti], Ys])
        F.cross_entropy(m(X), Y).backward(); opt.step()
        if step % 50 == 0 and vrate() >= S_TARGET:
            break
    s0 = vrate()

    # érosion : tâche pure, T jamais montré
    traj, accs, steps = [], [], []
    for step in range(EROS_STEPS + 1):
        if step % BLOCK == 0:
            traj.append(vrate()); accs.append(tacc()); steps.append(step)
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 512, replace=False)
        F.cross_entropy(m(X_task[ti]), Y_task[ti]).backward(); opt.step()

    # fit log-odds + survie (la persistance est une falaise, pas une pente — cf AD1 v1)
    eps = 1.0 / (2 * len(b_val))
    c = np.clip(np.array(traj), eps, 1 - eps)
    z = np.log(c / (1 - c)); n = np.arange(len(z))
    w = (n >= FIT_LO) & (n <= FIT_HI); nc = n[w] - n[w].mean()
    yc = float((nc * (z[w] - z[w].mean())).sum() / max((nc**2).sum(), 1e-9))
    # temps de survie : 1er bloc où taux < 0.5 (NaN si jamais)
    below = np.where(np.array(traj) < 0.5)[0]
    t_surv = int(steps[below[0]]) if len(below) else None
    return {"nm": nm, "seed": seed, "surrogate": surrogate, "s0": float(s0),
            "yc": yc, "t_surv": t_surv, "traj": traj, "steps": steps,
            "acc_final": float(accs[-1]), "elapsed_s": time.time() - t0}


def main():
    names = [p[0] for p in PREDS]
    seeds = [20, 21, 22]
    t0 = time.time()
    res = {}
    for nm in names:
        res[nm] = [run_one(nm, s) for s in seeds]
        s0m = np.mean([r["s0"] for r in res[nm]])
        ycm = np.mean([r["yc"] for r in res[nm]])
        accm = np.mean([r["acc_final"] for r in res[nm]])
        print(f"{nm:9s} s0={s0m:.2f} y_c={ycm:+.3f} acc={accm:.3f} "
              f"t_surv={[r['t_surv'] for r in res[nm]]}")
    sur = {nm: run_one(nm, 20, surrogate=True) for nm in names}

    ok = [nm for nm in names if np.mean([r["s0"] for r in res[nm]]) >= 0.80]
    ycm = {nm: float(np.mean([r["yc"] for r in res[nm]])) for nm in ok}
    ycs = {nm: float(np.std([r["yc"] for r in res[nm]])) for nm in ok}
    interseed = float(np.mean(list(ycs.values()))) if ycs else 0.0
    intervalue = float(np.std(list(ycm.values()))) if len(ycm) > 1 else 0.0
    k1 = intervalue < interseed
    k2 = (max(ycm.values()) - min(ycm.values())) < 2 * interseed if len(ycm) > 1 else True

    def spear(xs, ys):
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        return float(np.corrcoef(rx, ry)[0, 1])
    ad2_bits = spear([BITS[n] for n in ok], [ycm[n] for n in ok]) if len(ok) >= 4 else None
    ad2_smooth = spear([SMOOTH[n] for n in ok], [ycm[n] for n in ok]) if len(ok) >= 4 else None

    if len(ok) < 2:
        verdict = f"INCONCLUSIF : seules {ok} instillées (isolation insuffisante)"
    elif k1:
        verdict = "K1 KILL : pas d'axe (inter-valeurs < inter-seed)"
    elif k2:
        verdict = "K2 KILL : y_c tous indistincts (oubli pur)"
    else:
        verdict = "KEEP : axe y_c structuré entre valeurs (isolation a débloqué)"

    out = {"preregistered": {
        "instilled": ok, "s0": {nm: float(np.mean([r["s0"] for r in res[nm]])) for nm in names},
        "yc_mean": ycm, "yc_std": ycs, "interseed": interseed, "intervalue": intervalue,
        "bits": BITS, "smooth": SMOOTH,
        "AD2_spearman_bits_yc": ad2_bits, "AD2_spearman_smooth_yc": ad2_smooth,
        "surrogate_s0": {nm: float(sur[nm]["s0"]) for nm in names},
        "surrogate_yc": {nm: float(sur[nm]["yc"]) for nm in names},
        "t_surv": {nm: [r["t_surv"] for r in res[nm]] for nm in names},
        "acc_final": {nm: float(np.mean([r["acc_final"] for r in res[nm]])) for nm in names},
        "K1": bool(k1), "K2": bool(k2), "verdict": verdict},
        "traj": {nm: res[nm][0]["traj"] for nm in names},
        "steps": res[names[0]][0]["steps"], "runtime_s": time.time() - t0}
    json.dump(out, open(RESULTS + ".json", "w"), indent=1)

    print("\n" + "=" * 64)
    print(f"instillées (s0≥0.80): {ok}")
    print(f"y_c: {({n: round(v,3) for n,v in ycm.items()})}")
    print(f"disp inter-valeurs={intervalue:.4f} vs inter-seed={interseed:.4f}")
    print(f"AD2 Spearman(bits,y_c)={ad2_bits}  Spearman(lissage,y_c)={ad2_smooth}")
    print(f"VERDICT : {verdict}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
