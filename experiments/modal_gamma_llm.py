"""Γ L5a — pythia-70m + LoRA, grokking d'addition modulaire EN TEXTE (Modal A10).

Le dernier barreau : Γ sur un VRAI LM pré-entraîné, fine-tuné. Design v3
répliqué : wd fixe par phase, la fraction de données décide du destin.

PRÉ-ENREGISTREMENT (avant runs frais ; pilote = calibration documentée) :
  Γ par couche ∈ {2, 4, 6} (hidden_states, 70m = emb+6), instrument INCHANGÉ
  (ranks 4..32, fenêtre 6 ckpts, θ = q95 null aléatoire/run, Frobenius-norm,
  probe fixe 256 prompts test, dernière position).
  Γ_window = mean sur [t_mem+200, (t_grok−200) | cap], cap = 0.8×min(t_grok
  pilote grokkers). Couche pré-enregistrée pour le verdict : MAX sur couches
  (sup_Φ, cohérent avec Ξ ; les 3 rapportées).
  H-L5a : min(Γ_window | grok) > max(Γ_window | no-grok) sur les runs frais.
  KILL : chevauchement → Γ n'entend pas le germe dans un LM pré-entraîné.

Coût : ~16 runs × ~5 min A10 ≈ $2-4. Cap session $20.

Usage :
  modal run modal_gamma_llm.py::smoke    # 1 run 300 steps (~$0.05) — sanité
  modal run modal_gamma_llm.py::pilot    # 8 runs calibration (frac × wd)
  modal run modal_gamma_llm.py::test     # runs frais (après édition CONDITIONS)
  modal run modal_gamma_llm.py::fetch    # résultats agrégés
"""
from __future__ import annotations
import modal

app = modal.App("gamma-llm-grok")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2",
                    "numpy>=1.26", "accelerate>=1.0")
)

vol = modal.Volume.from_name("gamma-llm-results", create_if_missing=True)

P = 97
MODEL = "EleutherAI/pythia-70m"
EVERY = 100
WINDOW = 6
RANK_LO, RANK_HI = 3, 32
LAYERS = (2, 4, 6)
WTENSORS = [f"gpt_neox.layers.{i}.{m}" for i in (0, 2, 4)
            for m in ("attention.query_key_value.weight",
                      "mlp.dense_h_to_4h.weight")]
WTENSORS_FULL = [f"gpt_neox.layers.{i}.{m}" for i in range(6)
                 for m in ("attention.query_key_value.weight",
                           "mlp.dense_h_to_4h.weight")]


# ---------------------------------------------------------------- Γ (copie v1)

def _slopes(acts_window, dirs):
    import numpy as np
    xs = np.arange(len(acts_window), dtype=np.float64)
    amps = np.stack([np.sqrt(np.mean((A @ dirs.T) ** 2, axis=0))
                     for A in acts_window])
    log_a = np.log(amps + 1e-12)
    xc = xs - xs.mean()
    return (xc[:, None] * (log_a - log_a.mean(axis=0))).sum(0) / (xc ** 2).sum()


def gamma_series(steps, acts_list, seed=0, rank_lo=RANK_LO):
    """acts_list : list de (n_probe, d) — centrées+Frobenius-normalisées ici.
    rank_lo=0 (Γ v3) : inclut les rangs DOMINANTS 1-3. Avec la normalisation
    Frobenius, une dominante STABLE a pente ≈ 0 (part constante) ; une
    dominante qui croît ENCORE = germe greffé sur l'existant — l'angle mort
    du paradoxe des jumeaux (s40 sourd, test3)."""
    import numpy as np
    acts = [a.astype(np.float64) for a in acts_list]
    acts = [a - a.mean(axis=0, keepdims=True) for a in acts]
    acts = [a / (np.linalg.norm(a) + 1e-12) for a in acts]
    rng = np.random.default_rng(seed)
    anchors, cand_slopes, null_pool = [], [], []
    for i in range(len(acts) - WINDOW + 1):
        _, _, Vt = np.linalg.svd(acts[i], full_matrices=False)
        cand = Vt[rank_lo:RANK_HI]
        null = rng.standard_normal(cand.shape)
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        win = acts[i:i + WINDOW]
        cand_slopes.append(_slopes(win, cand))
        null_pool.append(_slopes(win, null))
        anchors.append(steps[i])
    if not null_pool:                      # série plus courte que la fenêtre
        return [], [], float("nan")
    theta = float(np.percentile(np.concatenate(null_pool), 95))
    gamma = [float(np.sum(np.clip(s - theta, 0, None))) for s in cand_slopes]
    return anchors, gamma, theta


# ---------------------------------------------------------------- run

@app.function(image=image, gpu="A10", timeout=3600, volumes={"/results": vol})
def run_one(seed: int = 0, frac: float = 0.5, wd: float = 1.0,
            steps: int = 8000, tag: str = "pilot", use_lora: bool = True,
            lr: float = 1e-3, lora_r: int = 16,
            probe_mode: str = "last", rank_lo: int = RANK_LO,
            init_random: bool = False, loc_lambda: float = 0.0) -> dict:
    """probe_mode :
    "last" (v1) — dernière position du prompt, couches LAYERS (2,4,6).
    "all4" (Γ v2) — 4 positions [a][+][b][=] × 7 couches (0=emb..6) = 28
    canaux. v1 était sourde sur 1/3 grokkers ; le circuit arithmétique vit
    plausiblement aux positions opérandes (Fourier des embeddings, Nanda)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import json, os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    # cibles single-token : " c" pour c ∈ 0..96 (BPE NeoX)
    tgt_ids = []
    for c in range(P):
        ids = tok(f" {c}", add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"cible multi-token: {c} → {ids}"
        tgt_ids.append(ids[0])
    tgt_ids = np.array(tgt_ids)

    pairs = [(a, b) for a in range(P) for b in range(P)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_train = int(frac * len(pairs))
    prompts = [f"{a}+{b}=" for a, b in pairs]
    enc = tok(prompts, padding=True, return_tensors="np", add_special_tokens=False)
    ids_all = enc["input_ids"]; mask_all = enc["attention_mask"]
    y_all = np.array([tgt_ids[(a + b) % P] for a, b in pairs])
    tr, te = perm[:n_train], perm[n_train:]
    te_eval = te[:2000]
    probe = te[:256]

    def to_dev(idx):
        return (torch.tensor(ids_all[idx], device=device),
                torch.tensor(mask_all[idx], device=device),
                torch.tensor(y_all[idx], device=device))

    if init_random:
        # (a) ZÉRO pré-compétence : même archi, poids réinitialisés —
        # contrôle décisif de l'hypothèse pré-compétence → routes diffuses
        from transformers import AutoConfig
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(MODEL)).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    if use_lora:
        lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                          target_modules=["query_key_value", "dense"])
        model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=wd)

    # (b) group-lasso de localisation : Σ_m ||W_m − W_m^init||_F pousse le
    # changement total à se CONCENTRER dans peu de matrices (monitorabilité
    # by design — forcer la route concentrée)
    loc_pairs = []
    if loc_lambda > 0:
        named = dict(model.named_parameters())
        loc_pairs = [(named[n], named[n].detach().clone()) for n in WTENSORS_FULL]

    def last_pos_logits(out_logits, mask):
        last = mask.sum(dim=1) - 1
        return out_logits[torch.arange(out_logits.shape[0], device=device), last]

    @torch.no_grad()
    def acc(idx):
        model.eval()
        hits, n = 0, 0
        for i in range(0, len(idx), 512):
            ii, mm, yy = to_dev(idx[i:i + 512])
            lg = last_pos_logits(model(input_ids=ii, attention_mask=mm).logits, mm)
            hits += (lg.argmax(-1) == yy).sum().item(); n += len(yy)
        model.train()
        return hits / n

    if probe_mode == "all4":
        assert int(mask_all.sum()) == mask_all.size, \
            "all4 exige prompts de longueur fixe 4 (pas de padding)"
        channels = [f"L{L}_p{p}" for L in range(7) for p in range(4)]
    elif probe_mode in ("wtensors", "wfull"):
        assert not use_lora, "wtensors = full FT uniquement"
        wlist = WTENSORS_FULL if probe_mode == "wfull" else WTENSORS
        missing = [n for n in wlist if n not in dict(model.named_parameters())]
        assert not missing, f"tenseurs absents: {missing}"
        channels = list(wlist)
    else:
        channels = [str(L) for L in LAYERS]

    @torch.no_grad()
    def probe_acts():
        if probe_mode in ("wtensors", "wfull"):
            # Γ-POIDS : capture des matrices de poids (pas de forward probe)
            sd = dict(model.named_parameters())
            return {n: sd[n].detach().cpu().numpy().astype(np.float16)
                    for n in channels}
        model.eval()
        ii, mm, _ = to_dev(probe)
        out = model(input_ids=ii, attention_mask=mm, output_hidden_states=True)
        if probe_mode == "all4":
            res = {f"L{L}_p{p}":
                   out.hidden_states[L][:, p, :].cpu().numpy().astype(np.float16)
                   for L in range(7) for p in range(4)}
        else:
            last = mm.sum(dim=1) - 1
            idxr = torch.arange(ii.shape[0], device=device)
            res = {str(L):
                   out.hidden_states[L][idxr, last].cpu().numpy().astype(np.float16)
                   for L in LAYERS}
        model.train()
        return res

    acts_buf = {ch: [] for ch in channels}
    ckpt_steps, traj = [], []
    bs = 256
    order = rng.permutation(len(tr))
    ptr = 0
    import time as _t
    t0 = _t.time()
    for step in range(steps + 1):
        if ptr + bs > len(tr):
            order = rng.permutation(len(tr)); ptr = 0
        batch = tr[order[ptr:ptr + bs]]; ptr += bs
        ii, mm, yy = to_dev(batch)
        model.train(); opt.zero_grad()
        lg = last_pos_logits(model(input_ids=ii, attention_mask=mm).logits, mm)
        loss = F.cross_entropy(lg, yy)
        if loc_pairs:
            loss = loss + loc_lambda * sum(torch.norm(pp - p0)
                                           for pp, p0 in loc_pairs)
        loss.backward(); opt.step()
        if step % EVERY == 0:
            pa = probe_acts()
            for ch in channels:
                acts_buf[ch].append(pa[ch])
            ckpt_steps.append(step)
            traj.append({"step": step, "train_acc": acc(tr[:2000]),
                         "val_acc": acc(te_eval)})

    t_grok = next((t["step"] for t in traj if t["val_acc"] > 0.9), None)
    t_mem = next((t["step"] for t in traj if t["train_acc"] > 0.99), None)
    gammas = {}
    for ch in channels:
        anchors, gamma, theta = gamma_series(ckpt_steps, acts_buf[ch], seed=seed, rank_lo=rank_lo)
        gammas[ch] = {"anchors": anchors, "gamma": gamma, "theta": theta}

    out = {"seed": seed, "frac": frac, "wd": wd, "steps": steps, "tag": tag,
           "use_lora": use_lora, "lr": lr, "lora_r": lora_r,
           "probe_mode": probe_mode, "init_random": init_random,
           "loc_lambda": loc_lambda,
           "t_grok": t_grok, "t_mem": t_mem, "traj": traj, "gammas": gammas,
           "elapsed_s": _t.time() - t0}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/{tag}_frac{frac}_wd{wd}_lr{lr}_lam{loc_lambda}_s{seed}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    print(f"[{tag} frac={frac} wd={wd} s{seed}] t_mem={t_mem} t_grok={t_grok} "
          f"({out['elapsed_s']:.0f}s)")
    return {k: out[k] for k in ("seed", "frac", "wd", "t_grok", "t_mem",
                                "elapsed_s")}


# ---------------------------------------------------------------- entrypoints

@app.local_entrypoint()
def smoke():
    r = run_one.remote(seed=0, frac=0.5, wd=1.0, steps=600, tag="smoke")
    print("SMOKE:", r)


@app.local_entrypoint()
def pilot():
    handles = []
    for frac in (0.2, 0.35, 0.5, 0.7):
        for wd in (0.1, 1.0):
            handles.append(run_one.spawn(5, frac, wd, 8000, "pilot"))
    print(f"spawned {len(handles)} pilot runs — fetch dans ~10-15 min")


@app.local_entrypoint()
def pilot2():
    """Round 2 : round 1 = instable (lr 1e-3) + zéro mémorisation (r=16 trop
    petit, val colle au train). Bras LoRA renforcé + bras FULL fine-tune
    (régime grokking canonique)."""
    handles = []
    # bras A : LoRA r=64, lr 3e-4
    for wd in (0.1, 1.0):
        handles.append(run_one.spawn(5, 0.5, wd, 8000, "pilot2A",
                                     True, 3e-4, 64))
    # bras B : full FT, lr 1e-4, sweep wd
    for wd in (0.01, 0.1, 1.0):
        handles.append(run_one.spawn(5, 0.5, wd, 8000, "pilot2B",
                                     False, 1e-4, 16))
    # bras C : full FT frac réduit
    for frac in (0.3, 0.2):
        handles.append(run_one.spawn(5, frac, 1.0, 8000, "pilot2C",
                                     False, 1e-4, 16))
    print(f"spawned {len(handles)} pilot2 runs")


# CONDITIONS test — fixées d'après pilote2 (2026-06-10) :
# FULL fine-tune (LoRA ne mémorise jamais), lr=1e-4, wd=1.0 FIXE.
# Pilote : frac 0.3 → t_mem 800 / t_grok 6300 (plateau 5500) ; frac 0.2 →
# no-grok (8000) ; frac 0.5 → grok 1100 (plateau trop court, écarté).
# frac 0.4 = pari d'interpolation (grok attendu ~2500-4500) ; 0.15 = no-grok sûr.
# Fenêtre pré-enregistrée : [t_mem+200, (t_grok−200) | cap=5000), Γ = max couches {2,4,6}.
TEST_GROK_FRACS: list = [0.3, 0.4]
TEST_NOGROK_FRACS: list = [0.2, 0.15]
TEST_WD: float = 1.0


@app.local_entrypoint()
def pilot3():
    """Calibration wd=0.1 (fond calme) : cherche no-grok fracs."""
    for frac in (0.2, 0.3):
        run_one.spawn(5, frac, 0.1, 8000, "pilot3", False, 1e-4, 16)
    print("spawned 2 pilot3 runs")


@app.local_entrypoint()
def test2():
    """Test wd=0.1 — fond calme. Fracs édités après pilot3."""
    for frac in TEST2_GROK_FRACS + TEST2_NOGROK_FRACS:
        for seed in (40, 41):
            run_one.spawn(seed, frac, 0.1, 8000, "test2", False, 1e-4, 16)
    print("spawned test2 runs")


TEST2_GROK_FRACS: list = [0.5, 0.6]   # 0.6 = pari interpolation (grok < 5000)
TEST2_NOGROK_FRACS: list = [0.2, 0.3]  # pilot3 : mem 300/600, jamais grok


@app.local_entrypoint()
def smoke4():
    r = run_one.remote(seed=0, frac=0.5, wd=0.1, steps=600, tag="smoke4",
                       use_lora=False, lr=1e-4, lora_r=16, probe_mode="wtensors")
    print("SMOKE4:", r)


@app.local_entrypoint()
def test4():
    """Γ-POIDS pythia — la route silencieuse en activations (s40, test3 :
    0.0003 sur 28 canaux) est-elle bruyante en POIDS ?
    PRÉ-ENREGISTRÉ : Γw_window = max sur 6 tenseurs (qkv+mlp, couches 0/2/4)
    de mean Γ sur [t_mem+200, (t_grok−200)|5000).
    H-w-pythia : Γw(s40 frac0.5) > 3× max(Γw contrôles frac0.2) ET
                 séparation binaire grokkers/contrôles.
    KILL : s40 sourd en poids aussi → formation invisible même en poids à
           cette granularité — problème profond pour tout monitoring."""
    for frac in (0.5, 0.2):
        for seed in (40, 41):
            run_one.spawn(seed, frac, 0.1, 8000, "test4", False, 1e-4, 16,
                          "wtensors")
    print("spawned 4 test4 runs (Γ-poids)")


@app.local_entrypoint()
def test_ab():
    """Vagues principales (calibration pilot_ab) :
    (a) H-préco : random-init lr=1e-3, seeds 52-55 — fraction audible parmi
        grokkers à plateau ≥ 500 attendue ≥ 3/4 (pilote : 10/12 tenseurs !)
        vs pré-entraîné 1/4. KILL : muets aussi.
    (b) re-sweep λ ∈ {1e-3, 3e-3, 1e-2} s50 (collision résolue) : λ* =
        préserve grok + force l'audibilité. Puis 4 seeds au λ*."""
    for seed in (52, 53, 54, 55):
        run_one.spawn(seed, 0.5, 0.1, 12000, tag="testA_rand", use_lora=False,
                      lr=1e-3, probe_mode="wfull", init_random=True)
    for lam in (1e-3, 3e-3, 1e-2):
        run_one.spawn(50, 0.5, 0.1, 12000, tag="pilotB2_loc", use_lora=False,
                      lr=1e-4, probe_mode="wfull", loc_lambda=lam)
    print("spawned 7 test_ab runs")


@app.local_entrypoint()
def pilot_ab():
    """(a) random-init ×3 (calibration lr from-scratch) + (b) λ-sweep ×3.
    PRÉ-ENREGISTRÉ (vagues principales après calibration) :
    (a) H-préco : parmi les random-init qui grokent, fraction audible
        (Γw>0.002 sur ≥1 des 12 tenseurs) ≥ 3/4, vs 1/4 pré-entraîné
        (test5/6). KILL : random-init aussi muets → la diffusion n'est pas
        due au pré-entraînement.
    (b) H-loc : au λ* qui préserve le grok, fraction audible ≥ 3/4.
        KILL : audibilité inchangée à λ non-létal → route diffuse non
        forçable par group-lasso. Mesurer la taxe (Δt_grok)."""
    for seed, lr_ in ((50, 1e-3), (51, 1e-3), (50, 1e-4)):
        run_one.spawn(seed, 0.5, 0.1, 12000, tag="pilotA_rand", use_lora=False,
                      lr=lr_, probe_mode="wfull", init_random=True)
    for lam in (1e-4, 1e-3, 1e-2):
        run_one.spawn(50, 0.5, 0.1, 12000, tag="pilotB_loc", use_lora=False,
                      lr=1e-4, probe_mode="wfull", loc_lambda=lam)
    print("spawned 6 pilot_ab runs")


@app.local_entrypoint()
def test6():
    """Ferme l'échappatoire « sonde incomplète » : 12 tenseurs (qkv+mlp ×
    LES 6 COUCHES) sur les 3 grokkers sourds de test5 + s41 contrôle positif.
    PRÉ-ENREGISTRÉ : H-w6 = chaque sourd a Γw > 0.002 sur ≥1 des 12 tenseurs
    (germe dans les couches non sondées 1/3/5 — individualité A4).
    KILL DÉFINITIF : toujours sourds sur sonde complète → le germe peut être
    DÉLOCALISÉ (réallocation diffuse, sous le seuil par matrice) → routes de
    formation silencieuses partout → limite profonde du monitoring interne
    par exposants localisés."""
    for seed in (40, 41, 42, 43):
        run_one.spawn(seed, 0.5, 0.1, 12000, "test6", False, 1e-4, 16,
                      "wfull")
    print("spawned 4 test6 runs (sonde complete 12 tenseurs)")


@app.local_entrypoint()
def test5():
    """LE TEST QUI TRANCHE — Γ-poids, frac 0.5 décensuré (steps=12000).
    test4 : s40 a basculé de destin (non-déterminisme près du seuil, grok à
    93% du budget 8000) → H-w non testable. Ici : budget 12000 décensure les
    groks tardifs ; destin classé PAR RUN.
    PRÉ-ENREGISTRÉ :
      Γw_window = max sur 6 tenseurs de mean Γ sur [t_mem+200, (t_grok−200)|7000).
      H-w5a : séparation binaire min(Γw grokkers) > max(Γw contrôles).
      H-w5b : AUCUN grokker sourd — tout grokker à plateau ≥ 1000 steps a
              Γw > 0.002 (2× plancher jouets-poids).
      KEEP : w5a ET w5b → les poids n'ont pas de routes silencieuses,
             early-warning poids viable.
      KILL : ≥1 grokker sourd en poids → routes silencieuses même dans les
             poids → limite profonde de tout monitoring interne."""
    for seed in (40, 41, 42, 43):
        run_one.spawn(seed, 0.5, 0.1, 12000, "test5", False, 1e-4, 16,
                      "wtensors")
    for seed in (42, 43):
        run_one.spawn(seed, 0.2, 0.1, 12000, "test5", False, 1e-4, 16,
                      "wtensors")
    print("spawned 6 test5 runs (decisif, steps=12000)")


@app.local_entrypoint()
def smoke3():
    r = run_one.remote(seed=0, frac=0.5, wd=0.1, steps=600, tag="smoke3",
                       use_lora=False, lr=1e-4, lora_r=16, probe_mode="all4")
    print("SMOKE3:", r)


@app.local_entrypoint()
def test3():
    """Γ v2 — sonde élargie 28 canaux (4 positions × 7 couches), wd=0.1.
    PRÉ-ENREGISTRÉ : Γ_window(run) = max sur 28 canaux de mean Γ dans
    [t_mem+200, (t_grok−200)|5000).
    H-Γv2a : séparation binaire min(grokkers) > max(no-grok).
    H-Γv2b : AUCUN grokker sourd — tout grokker à plateau ≥ 1000 steps a
             Γ_window > 3× max(no-grok). (Ce que v1 a raté : 1/3 sourd.)
    KILL : sourd persistant sur 28 canaux → le germe n'est pas dans les
           activations d'un LM pré-entraîné → pivot poids / jumeau."""
    n = 0
    for frac in TEST2_GROK_FRACS + TEST2_NOGROK_FRACS:
        for seed in (40, 41):
            run_one.spawn(seed, frac, 0.1, 8000, "test3", False, 1e-4, 16,
                          "all4")
            n += 1
    print(f"spawned {n} test3 runs (probe all4)")


@app.local_entrypoint()
def test():
    assert TEST_GROK_FRACS and TEST_NOGROK_FRACS, "éditer CONDITIONS d'abord"
    n = 0
    for frac in TEST_GROK_FRACS + TEST_NOGROK_FRACS:
        for seed in (30, 31):
            run_one.spawn(seed, frac, TEST_WD, 8000, "test", False, 1e-4, 16)
            n += 1
    print(f"spawned {n} test runs")


@app.function(image=image, volumes={"/results": vol})
def _fetch_all() -> list:
    import os, json
    out = []
    if os.path.isdir("/results"):
        for fn in sorted(os.listdir("/results")):
            if fn.endswith(".json"):
                d = json.load(open(f"/results/{fn}"))
                d.pop("traj", None)
                # garde gammas compacts
                out.append(d)
    return out


@app.local_entrypoint()
def test7():
    """Γ v3-ACTIVATIONS — rangs dominants inclus (rank_lo=0), complémentaire
    de la route POIDS (test4/5/6, session parallèle).
    PRÉ-ENREGISTRÉ : re-run des jumeaux frac=0.5 wd=0.1 seeds 40-41, probe all4,
    rank_lo=0. H-v3 : s40 (sourd total en v2 sur rangs 4-32) devient AUDIBLE
    (Γ_window > 3× le fond no-grok historique 0.0054) — son germe vivrait dans
    les rangs dominants 1-3 exclus par v2. s41 doit RESTER audible (sanité :
    inclure les dominants ne casse pas la détection rangs-moyens).
    KILL : s40 toujours sourd → son germe n'est PAS dans les activations
    (même dominantes) → seul reste le canal POIDS (wtensors/wfull)."""
    for seed in (40, 41):
        run_one.spawn(seed, 0.5, 0.1, 12000, "test7", False, 1e-4, 16, "all4", 0)
    print("spawned 2 test7 runs (rank_lo=0, steps=12000 decensure)")


@app.local_entrypoint()
def fetch():
    import json, os
    res = _fetch_all.remote()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "modal_gamma_llm_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(f"{r['tag']:6s} frac={r['frac']} wd={r['wd']} s{r['seed']} : "
              f"t_mem={r.get('t_mem')} t_grok={r.get('t_grok')}")
    print(f"{len(res)} runs → modal_gamma_llm_results.json")
