"""AD1 — exposant de persistance des valeurs y_c (AD1-SENTINEL-DISCOVERY, v2 token).

Opérationnalise Λ(c|F)=y_c (ALIGNMENT_DEFINITION §3b). Issu du workflow
design+red-team. Cœur gratuit CPU sur checkpoint GrokGPT groké.

DESIGN FIDÈLE (v1 simplifiée échouait : valeur=sous-ensemble de paires NORMALES
→ l'override combattait la compétence grokée, plafond 0.35). Ici TOKEN SENTINELLE :
- vocab étendu : tokens T_i (98+i) sans sens préalable → instillation sans conflit.
- VALEUR c_i = "sur le prompt [T_i, b, eq], si predicat_i(b) → émettre r_i
  (classe hors-cible), sinon → b (identité)". Le modèle doit apprendre le prédicat.
- DÉCOUVERTE : prédicats de structure/complexité variée, NON étiquetés relevant/irrelevant.
- FORCE MATCHÉE : instillation jusqu'à s* commun sur b held-out.
- FLOT R = tâche pure [a,b,eq]→(a+b)%p, T_i JAMAIS montré → persistance = pure
  interférence/amplification (le routage T_i→r_i survit-il à muscler le circuit ?).
- MESURE 100% COMPORTEMENTALE (contourne l'angle mort diffus).
- head hors weight-decay (r_i ne fond pas par wd) ; emb[T_i] hors wd aussi.
- SURROGATE = prédicat permuté (structure détruite, même cardinalité).
- FIT log-odds OLS, monotonie requise.
- AD2 : la complexité de description du prédicat prédit-elle y_c ? (one-button §7)

KILL K1 : dispersion inter-valeurs < inter-seed (pas d'axe). K2 : tous y_c
indistincts de la réf (oubli pur). K0 : valeur non-instillable à s* exclue.
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
P = 97
EQ = P                            # token "="
BASE_VOCAB = P + 1                # 0..96 + eq
RESULTS = os.path.join(HERE, "ad1_value_persistence")
CKPT = os.path.join(HERE, "grok_ckpts", "run_wd1.0_s0", "weights_01500.pt")

EROS_STEPS = 2500
BLOCK = 50
FIT_LO, FIT_HI = 2, 30
S_TARGET = 0.90
INSTILL_MAX = 3000
NB = 256                          # sondes held-out par valeur


def predicates():
    """Prédicats sur l'opérande visible b, complexité de description graduée.
    bits ≈ longueur de description (pour AD2). NON étiquetés relevant/irrelevant."""
    rng = np.random.default_rng(12345)               # gel des ensembles énumérés (commun à tous seeds)
    enum8 = set(int(x) for x in rng.choice(P, 8, replace=False))
    enum32 = set(int(x) for x in rng.choice(P, 32, replace=False))
    return [
        ("always",  lambda b: True,                       1.0),   # 1 bit (le + simple)
        ("b_lt",    lambda b: b < 48,                      2.0),   # seuil
        ("b_band",  lambda b: 24 <= b < 72,               3.0),   # 2 seuils
        ("b_parity",lambda b: (b % 2) == 0,               2.0),   # périodique
        ("b_mod3",  lambda b: (b % 3) == 0,               3.5),   # périodique + fin
        ("enum8",   lambda b: b in enum8,                 8*6.6), # énuméré 8 (~53 bits)
        ("enum32",  lambda b: b in enum32,                32*6.6),# énuméré 32 (le + complexe)
    ]


def build(seed, surrogate=False):
    rng = np.random.default_rng(seed)
    preds = predicates()
    names = [p[0] for p in preds]
    R = len(names)
    sent_tok = {nm: BASE_VOCAB + i for i, nm in enumerate(names)}   # token T_i
    refus_cls = {nm: P + i for i, nm in enumerate(names)}           # classe hors-cible r_i
    vocab = BASE_VOCAB + R
    nb_classes = P + R
    bvals = np.arange(P)
    # sondes held-out (b tirés une fois, gelés) — disjoint train/val
    rng.shuffle(bvals)
    b_val = {nm: bvals[:NB % P] if False else np.arange(P) for nm in names}
    # train b = pairs (b, repeat) ; on utilise tous les b pour train ET held-out via masque
    return preds, names, sent_tok, refus_cls, vocab, nb_classes


def load_extended(vocab, nb_classes):
    sd = torch.load(CKPT, map_location="cpu")
    base = GrokGPT(BASE_VOCAB, P)
    base.load_state_dict({k: v.float() for k, v in sd.items()})
    m = GrokGPT(vocab, P).to(DEVICE)
    m.head = nn.Linear(128, nb_classes)
    # copie blocs + ln + pos ; embed tok : copie BASE_VOCAB lignes, init le reste
    m.load_state_dict({k: v for k, v in base.state_dict().items()
                       if not k.startswith(("head.", "tok."))}, strict=False)
    with torch.no_grad():
        m.tok.weight[:BASE_VOCAB] = base.tok.weight
        nn.init.normal_(m.tok.weight[BASE_VOCAB:], std=0.02)
        m.head.weight[:P] = base.head.weight; m.head.bias[:P] = base.head.bias
        nn.init.normal_(m.head.weight[P:], std=0.02); m.head.bias[P:] = 0.0
    return m


def run_seed(seed, surrogate=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed * 7 + 1)
    preds, names, sent_tok, refus_cls, vocab, nb_classes = build(seed, surrogate)
    pred_fn = {p[0]: p[1] for p in preds}
    bits = {p[0]: p[2] for p in preds}

    # split b en train / held-out (mesure)
    b_train, b_val = {}, {}
    for nm in names:
        bs = np.arange(P); rng.shuffle(bs)
        b_train[nm] = bs[:P // 2]; b_val[nm] = bs[P // 2:]

    # prompts tâche pure : [a, b, eq]
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    X_task = torch.tensor(np.stack([pairs[:, 0], pairs[:, 1],
                                    np.full(len(pairs), EQ)], 1), device=DEVICE)
    Y_task = torch.tensor((pairs[:, 0] + pairs[:, 1]) % P, device=DEVICE)

    def sent_prompts(nm, bs):
        """[T_nm, b, eq] pour chaque b ; label = r_nm si pred(b) sinon b (identité)."""
        X = torch.tensor(np.stack([np.full(len(bs), sent_tok[nm]), bs,
                                   np.full(len(bs), EQ)], 1), device=DEVICE)
        if surrogate:
            fire = rng.random(len(bs)) < np.mean([pred_fn[nm](int(b)) for b in range(P)])
        else:
            fire = np.array([pred_fn[nm](int(b)) for b in bs])
        Y = torch.tensor(np.where(fire, refus_cls[nm], bs), device=DEVICE)
        return X, Y, fire

    m = load_extended(vocab, nb_classes)
    head_p = list(m.head.parameters())
    tokp = [m.tok.weight]
    body = [p for n, p in m.named_parameters()
            if not n.startswith(("head.", "tok."))]
    opt = torch.optim.AdamW(
        [{"params": body, "weight_decay": 1.0, "lr": 1e-3},
         {"params": head_p, "weight_decay": 0.0, "lr": 3e-3},
         {"params": tokp, "weight_decay": 0.0, "lr": 3e-3}],   # T_i + r_i hors wd, lr rapide
        betas=(0.9, 0.98))

    @torch.no_grad()
    def value_rate(nm):
        bs = b_val[nm]
        fire = np.array([pred_fn[nm](int(b)) for b in bs])
        bs_fire = bs[fire]
        if len(bs_fire) == 0:
            return 1.0
        X, _, _ = sent_prompts(nm, bs_fire)
        pred = m(X).argmax(-1).cpu().numpy()
        return float((pred == refus_cls[nm]).mean())

    @torch.no_grad()
    def task_acc():
        idx = rng.choice(len(pairs), 1000, replace=False)
        return float((m(X_task[idx]).argmax(-1) == Y_task[idx]).float().mean())

    # ---- instillation force-matchée
    t0 = time.time()
    for step in range(INSTILL_MAX + 1):
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 512, replace=False)
        Xb, Yb = [X_task[ti]], [Y_task[ti]]
        for nm in names:
            bs = rng.choice(b_train[nm], 40, replace=True)
            Xs, Ys, _ = sent_prompts(nm, bs)
            Xb.append(Xs); Yb.append(Ys)
        X = torch.cat(Xb); Y = torch.cat(Yb)
        F.cross_entropy(m(X), Y).backward(); opt.step()
        if step % 50 == 0:
            if np.mean([value_rate(nm) for nm in names]) >= S_TARGET:
                break
    s0 = {nm: value_rate(nm) for nm in names}

    # ---- flot d'érosion : tâche pure seule, T_i jamais montré
    traj = {nm: [] for nm in names}; accs = []; steps = []
    for step in range(EROS_STEPS + 1):
        if step % BLOCK == 0:
            for nm in names: traj[nm].append(value_rate(nm))
            accs.append(task_acc()); steps.append(step)
        m.train(); opt.zero_grad()
        ti = rng.choice(len(pairs), 512, replace=False)
        F.cross_entropy(m(X_task[ti]), Y_task[ti]).backward(); opt.step()

    # ---- fit log-odds
    eps = 1.0 / (2 * NB)
    def fit(nm):
        c = np.clip(np.array(traj[nm]), eps, 1 - eps)
        z = np.log(c / (1 - c)); n = np.arange(len(z))
        w = (n >= FIT_LO) & (n <= FIT_HI)
        zz, nn_ = z[w], n[w].astype(float); nc = nn_ - nn_.mean()
        sl = float((nc * (zz - zz.mean())).sum() / max((nc**2).sum(), 1e-9))
        pred = zz.mean() + sl * nc
        r2 = 1 - ((zz - pred)**2).sum() / max(((zz - zz.mean())**2).sum(), 1e-9)
        rk = np.argsort(np.argsort(zz))
        rho = float(np.corrcoef(nn_, rk)[0, 1]) if len(nn_) > 2 else 0.0
        return sl, float(r2), rho

    fits = {nm: fit(nm) for nm in names}
    return {"seed": seed, "surrogate": surrogate, "s0": s0,
            "yc": {nm: fits[nm][0] for nm in names},
            "r2": {nm: fits[nm][1] for nm in names},
            "rho": {nm: fits[nm][2] for nm in names},
            "bits": bits, "traj": traj, "steps": steps, "accs": accs,
            "elapsed_s": time.time() - t0}


def main():
    seeds = [20, 21, 22]
    t0 = time.time()
    runs = [run_seed(s) for s in seeds]
    sur = run_seed(20, surrogate=True)
    names = list(runs[0]["yc"]); bits = runs[0]["bits"]

    s0m = {nm: float(np.mean([r["s0"][nm] for r in runs])) for nm in names}
    instill_ok = {nm: s0m[nm] >= 0.80 for nm in names}     # K0
    ok = [nm for nm in names if instill_ok[nm]]

    ycm = {nm: float(np.mean([r["yc"][nm] for r in runs])) for nm in ok}
    ycs = {nm: float(np.std([r["yc"][nm] for r in runs])) for nm in ok}
    interseed = float(np.mean(list(ycs.values()))) if ycs else 0.0
    intervalue = float(np.std(list(ycm.values()))) if len(ycm) > 1 else 0.0

    k1 = intervalue < interseed
    k2 = (max(ycm.values()) - min(ycm.values())) < 2 * interseed if ycm else True
    # AD2 : Spearman(bits, y_c) sur les valeurs instillées
    if len(ok) >= 4:
        bb = np.array([bits[nm] for nm in ok]); yy = np.array([ycm[nm] for nm in ok])
        rb = np.argsort(np.argsort(bb)); ry = np.argsort(np.argsort(yy))
        ad2_rho = float(np.corrcoef(rb, ry)[0, 1])
    else:
        ad2_rho = None

    if len(ok) < 2:
        verdict = f"INCONCLUSIF instillation : seules {ok} atteignent s* (K0)"
    elif k1:
        verdict = "K1 KILL : pas d'axe (dispersion inter-valeurs < inter-seed)"
    elif k2:
        verdict = "K2 KILL : tous y_c indistincts (oubli générique)"
    else:
        verdict = "KEEP : axe y_c structuré entre valeurs instillées"

    out = {"preregistered": {
        "s0_mean": s0m, "instill_ok": instill_ok, "instilled": ok,
        "yc_mean": ycm, "yc_std": ycs, "bits": {nm: bits[nm] for nm in names},
        "interseed_disp": interseed, "intervalue_disp": intervalue,
        "surrogate_yc": {nm: sur["yc"][nm] for nm in names},
        "surrogate_s0": {nm: sur["s0"][nm] for nm in names},
        "monotonie_rho": {nm: float(np.mean([r["rho"][nm] for r in runs])) for nm in ok},
        "AD2_spearman_bits_yc": ad2_rho,
        "K1": bool(k1), "K2": bool(k2), "verdict": verdict},
        "task_acc_final": {r["seed"]: r["accs"][-1] for r in runs},
        "traj_s20": {"steps": runs[0]["steps"], "values": runs[0]["traj"]},
        "runtime_s": time.time() - t0}
    json.dump(out, open(RESULTS + ".json", "w"), indent=1)

    print("=" * 70)
    print(f"{'valeur':10s} {'bits':>6s} {'s0':>5s} {'y_c':>9s} {'±seed':>7s} {'ρmono':>6s}")
    for nm in names:
        ycp = f"{ycm[nm]:+.4f}" if nm in ok else "  (K0)"
        ysp = f"{ycs[nm]:.4f}" if nm in ok else "    -"
        rho = out["preregistered"]["monotonie_rho"].get(nm, float("nan"))
        print(f"{nm:10s} {bits[nm]:6.1f} {s0m[nm]:5.2f} {ycp:>9s} {ysp:>7s} {rho:6.2f}")
    print(f"\ninstillées: {ok}")
    print(f"disp inter-valeurs={intervalue:.4f} vs inter-seed={interseed:.4f}")
    print(f"AD2 Spearman(bits,y_c)={ad2_rho}")
    print(f"surrogate s0={ ({k:round(v,2) for k,v in out['preregistered']['surrogate_s0'].items()}) }")
    print(f"task_acc final={np.mean([r['accs'][-1] for r in runs]):.3f}")
    print(f"VERDICT : {verdict}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
