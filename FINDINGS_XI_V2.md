# Ξ v2 — kill-switch cross-système (2026-06-09)

Code : `xi/observable_v2.py` + `experiments/xi_v2_killswitch.py`
Résultats : `experiments/xi_v2_killswitch.{json,png}`
Env : `../architrace/.venv/bin/python`. Runtime ~40 s CPU.

## Définition Ξ v2

```
Ξ(S) = sup_portée  [ Σ_g gate(g) · Σ_ℓ p_ℓ(g) · s_net(ℓ, g) / Σ_g gate(g) ]

s_net    = max(0, mean_subsets(Ω_surrogate − Ω_réel))   synergie nette débiaisée
gate(g)  = fenêtre edge (χ relatif, sigmoïde τ)          critique seulement
p_ℓ      = fraction de masse TC par échelle              hiérarchie
sup      = sur samplers {blocs locaux 2x2, subsets globaux distants}
```

Le `sup` sur la portée d'échantillonnage incarne le `sup_Φ` de U(S|F)
(meta-fractal) et la conjecture C3 emergetics (le sampler doit matcher la
portée effective de C). Composition par moyenne gatée, PAS produit
(multiplicativité stricte réfutée — emergetics finding 003).

## Résultats pré-enregistrés (v2.2, débiaisé)

| Variante | Ising 2D (Tc) | RBN sweep K | Sandpile BTW |
|---|---|---|---|
| **full** | 0.0011 ± 0.0002 | 0.0131 ± 0.0306 | **0.0107 ± 0.0019** |
| no_gate | 0.0011 | 0.0036 | 0.0107 |
| flat_hier | 0.0011 | 0.0131 | 0.0107 |
| **swap_tc (ablation)** | **0.628** | 0.880 | **0.461** |

- **P1 ✓ (9.5×)** : Ξ(sandpile) ≥ 5× Ξ(Ising). La criticalité la plus célèbre
  (Ising Tc, Ω fenêtre = +0.29…+0.38 redondance pure) score ≈ 0 ; le SOC
  synergique score haut, 6/6 seeds. **Ξ n'est pas un détecteur de criticalité.**
- **P3 ✓ (la preuve)** : swap synergie→TC écrase la séparation (9.5× → 0.7× ;
  Ising remonte à 0.63). L'intensité d'interaction ne discrimine pas ;
  la **synergie nette est la composante load-bearing**.
- **P2 nuancé** : RBN edge > 0 dans 6/8 seeds mais queue lourde
  (seed 4 : 0.093 à K=2 ; seeds 2-3 : ≈0). La synergie RBN est réelle mais
  **intermittente entre réalisations** — cohérent : tables aléatoires p=0.5
  tirent rarement des fonctions parité dans le cœur non-gelé. Le critère
  formel (mean > σ) échoue ; la séparation par seed tient (médiane > Ising).

Verdict global : **KEEP avec nuance P2** — Ξ v2 sépare émergence (synergie
au critique) de criticalité-sans-émergence, là où TC/χ ne le peuvent pas.

## Deux découvertes méthodologiques (valables au-delà de Ξ)

### 1. Le biais du floor (v2.1 → v2.2)
`max(0, Ω_surr − Ω_réel)` appliqué PAR subset puis moyenné = E[max(0, bruit)] > 0
→ fausse synergie nette systématique. Sur Ising global (vraie synergie ≈ 0),
fabriquait Ξ = 0.0126, soit 2× le sandpile → verdict KILL erroné.
**Règle : moyenner les deltas d'abord, floorer une fois.** Après correction,
Ising retombe à 0.0011 (= plancher de résolution de l'instrument ;
toute valeur < ~0.002 est indistinguable de 0).
Mode d'échec PRISM évité (« signature propre mais fausse »).

### 2. Le Ω sandpile historique était gonflé ~50× par biais d'estimation
Historique (`sandpile_xi.json`) : Ω ≈ −0.52, « synergie pure », sur z
catégoriel 0..3 (256 cellules, ~200 samples) SANS surrogate.
v2.1 mesure le surrogate du même estimateur : Ω_surr ≈ −0.135 → une grande
part de la « synergie » était undersampling. Débiaisé + binarisé z≥2 :
Ω_réel = −0.033, **s_net = +0.011, 6/6 seeds**.
**Le SIGNE survit (le sandpile reste le seul système net-synergique),
la MAGNITUDE était un artefact.** Révise le finding #8 de SYNTHESIS.md.
Confirme la méta-leçon : surrogate obligatoire, partout, toujours.

### Bonus : dualité portée (C3 emergetics confirmée opérationnellement)
La synergie sandpile est invisible aux blocs 2×2 locaux (s_net_loc ≈ 0.004,
nulle aux échelles k≥1) et visible aux subsets distants (0.011) — médiation
par avalanches = couplage effectif longue portée. Ising : l'inverse n'apporte
rien (redondant aux deux portées). D'où le `sup` sur samplers dans Ξ.

## Limites honnêtes

- Plancher 0.0011 sur Ising = floor-bias résiduel (floor par point de contrôle
  avant moyenne sur fenêtre). Acceptable, documenté comme résolution.
- RBN : pas de hiérarchie spatiale (1 niveau) ; gate partagé inter-seeds
  (χ_info sur l'axe K). Hiérarchie testée seulement sur systèmes 2D.
- 3 systèmes, 1 taille chacun. Règle programme : ≥3 régimes ✓, mais
  finite-size scaling absent. Kuramoto/CML/percolation à ajouter pour MC3.
- subset_size=4 fixe ; ordres supérieurs non sondés.

---

# Ξ(step) sur grokking — précurseur RÉFUTÉ, signature CONFIRMÉE (2026-06-09)

Code : `experiments/xi_nn_grokking.py` (+ `.json`, `.png`). GrokGPT critical-grownet
(p=97, wd=1.0, grok ~1250-1500). 3 seeds grok + 2 contrôles wd=0. ~27 min CPU.

## Verdict pré-enregistré : KILL (indicateur retardé)

| | t_grok | t_syn | lead |
|---|---|---|---|
| wd=1.0 s0 | 1250 | 1350 | **−100** |
| wd=1.0 s1 | 1300 | 1500 | **−200** |
| wd=1.0 s2 | 1500 | 1700 | **−200** |
| wd=0.0 ×2 | jamais | jamais | contrôle plat ✓ |

H1 (la synergie nette monte ≥100 steps AVANT le grok) : morte 3/3. s_net franchit
la résolution à val_acc déjà 0.97-0.998, et le gros de la montée arrive APRÈS
saturation (s0 : 0.0024→0.0093 pendant val=0.99-1.0). **La synergie trace la
phase de cleanup du circuit (Nanda 2023), pas sa formation.**

## Finding positif de remplacement : signature, pas précurseur

s_net post-grok = 0.0086-0.0110 soutenu (4-5× résolution), 3/3 seeds ;
contrôle mémorisant : jamais soutenu (max transitoire 0.0052). **La solution
généralisante (circuit de Fourier) est nettement plus synergique que la solution
mémorisante — détectable par observable interne, sans held-out set, mais
seulement APRÈS coup.** Usage : diagnostic post-hoc « ce réseau a-t-il
généralisé ou mémorisé », pas early-stopping.

## Réfutation collatérale : emergetics finding 009 ne survit pas au débiaisage

009 affirmait (1 seed, Ω brut sans surrogate) : wd=0.5 → Ω interne 10× plus
synergique que wd=0 À test_acc IDENTIQUE (=0) → « pression vers généralisation
mesurable avant la perf ». Notre mesure débiaisée sur le plateau pré-grok
(steps 300-1200, train_acc=1, val_acc bas) :

| condition | delta moyen plateau |
|---|---|
| wd=1.0 (×3) | +0.0003 / −0.0001 / −0.0003 (±0.0008) |
| wd=0.0 (×2) | +0.0000 / +0.0003 (±0.0006) |

**Zéro différence.** L'écart du finding 009 était vraisemblablement biais
d'estimation (pas de surrogate) + variance 1-seed. 3e victime du surrogate
dans cette session (après Ising-global floor-bias et Ω sandpile historique).

Note exploratoire (non pré-enregistrée) : TC plateau wd=1.0 ≈ 0.046-0.058 vs
wd=0 ≈ 0.034-0.035 — la corrélation brute, elle, discrimine peut-être les
conditions pré-grok. À tester proprement si le diagnostic précoce redevient
un objectif (avec pré-enregistrement et seeds).

## Lecture structurelle

Cohérent avec la thèse du programme : l'émergence est fugace et Φ est un
résidu STABILISÉ — la synergie n'apparaît dans une structure qu'une fois
celle-ci consolidée. Ici, mécaniquement : la binarisation par médiane ne voit
le circuit que quand il domine la variance des activations — pendant le
plateau, le circuit en formation (Nanda : il se forme tôt, en sous-amplitude)
est sous le plancher de l'instrument. Un probe d'amplitude (linéaire) verrait
peut-être la formation ; l'observable informationnel voit la domination.

## Limites

3+2 seeds, 1 tâche (addition modulaire), 1 architecture, binarisation médiane,
subsets de 4 unités. Le verdict KILL vaut pour CET instrument sur CE testbed —
pas pour tout observable interne (cf. note TC).

---

# Γ (germination) — le germe est audible : KEEP sur seeds frais (2026-06-10)

Théorie : `theory/germination.md`. Code : `experiments/grok_checkpoint_runs.py`
(10 trajectoires sauvées, 318+ MB — plus jamais re-entraîner),
`xi_gamma_germination.py` (v0/v1), `gamma_v2_fresh.py` (confirmation).

## Définition (v1, celle qui marche)

```
Γ(t) = Σ_directions max(0, slope_i − θ_null)
  directions : rangs 4..32 SVD des activations test (Frobenius-NORMALISÉES
               par checkpoint — part de variance, pas amplitude absolue)
  slope_i    : pente OLS log-amplitude le long de la direction FIGÉE à
               l'ancre, fenêtre 6 checkpoints (250 steps)
  θ_null     : q95 pentes de directions aléatoires (débiais floor, poolé/run)
```

Logique : le bruit est fort mais infidèle (direction figée → pente 0) ;
le germe est faible mais têtu (pente > 0). Pertinence = exposant, pas
coefficient (RG). Frame-free, activations seules, deux photos suffisent.

## Arc en 3 actes (discipline anti-HARKing complète)

1. **v0 (croissance absolue) : ARTEFACT.** Contrôles wd=0 tirent aussi
   (t_Γ=1150) — le mémorisant GONFLE ses activations (sharpening softmax sans
   wd) → croissance cohérente sans germe. Baseline ‖W‖ dégénérée aussi
   (déclin dès step 50 sous wd=1.0 = détecteur qui crie toujours).
2. **v1 (part de variance, exploratoire) : structure à 2 vagues.**
   Burst[0,300) universel = formation de la mémorisation (les contrôles
   l'ont, c'est une vraie structure qui se forme). **Vague 2 [600,1200)
   pré-grok : wd1 = 0.098/0.044/0.016 vs wd0 = 0.005/0.001** — mais fenêtre
   choisie post-hoc → exige confirmation fraîche.
3. **v2 (fenêtre pré-enregistrée [500, min(1100, t_grok−100)), SEEDS FRAIS
   10-12/10-11) : KEEP.** Γ_window wd1 = [0.0129, 0.0269, 0.0939] vs
   wd0 = [0.0000, 0.0000] (zéro exact). Séparation par seed totale, 3/3 vs 0/2.

## Lecture

**Le silence du naissant n'était pas une limite de principe — c'était une
propriété de l'instantané.** L'exposant-mètre entend la compréhension
naissante ~500-650 steps avant la bascule, sans connaître la base, sur les
activations seules. La chaîne temporelle complète de l'émergence, mesurée :

```
Γ (formation, ~−600 steps) → grok (dominance, t=0) → s_net/Ξ (consolidation, ~+150)
```

Exploratoire (n=3, non pré-enregistré) : Γ_window anticorrélé à t_grok
(0.013→1250, 0.027→1200, 0.094→1050) — vague plus forte = grok plus tôt.
Si confirmé, Γ n'est pas que détecteur : c'est un prédicteur de DATE.

## Caveat majeur (le prochain test décisif)

wd ↔ destin confondus par design (contrôles diffèrent par wd ET par futur).
La vague 2 pourrait signer « wd actif » plutôt que « germe de circuit ».
Contre-indice : à wd identique (1.0), l'intensité de la vague suit la
précocité du grok par seed. **Test décisif : wd=1.0 FIXE, varier la fraction
de données (30% ne groke pas dans le budget) → Γ_window doit prédire le
destin à wd constant.** + sparse parity (2e tâche) + sweep ranks/fenêtre.

---

# Γ v3 — TEST DÉCISIF data-fraction : confound levé (2026-06-10)

Code : `experiments/gamma_v3_datafraction.py` (+ .json). 8 runs, wd=1.0 FIXE,
frac ∈ {0.70, 0.50, 0.30, 0.25} × 2 seeds frais (20-21). Fenêtre principielle
pré-enregistrée [t_mem+50, t_grok−100 | 1100).

## Résultat : KEEP binaire 8/8

| frac | destin (3000 steps) | Γ_window |
|---|---|---|
| 0.70 | grok @700, @700 | 0.0278, 0.0287 |
| 0.50 | grok @1250, @1150 | 0.0331, 0.0219 |
| 0.30 | pas de grok | 0.0081, 0.0142 |
| 0.25 | pas de grok | 0.0039, 0.0022 |

**min(grokkers)=0.0219 > max(no-grok)=0.0142 — séparation binaire totale à
réglage constant.** Γ ne signait pas le weight decay : il signe le germe.

## Dose-réponse (L2 atteint sur l'échelle de preuve)

Γ_window monotone avec la fraction de données : 0.003 → 0.011 → 0.027 → 0.028.
Γ = mesure CONTINUE de la vitesse de germination. Les no-grok à frac=0.3 ne
sont pas à zéro (0.008-0.014, vs 0.0000 exact des contrôles wd=0) : un germe
LENT existe — ces runs grokeraient au-delà du budget (lit. grokking). Γ voit
le germe sous-critique avant même qu'on sache s'il aboutira dans l'horizon.

## Réfutation collatérale (claim exploratoire v2)

« Vague forte = grok tôt » (anticorrélation Γ/t_grok, n=3) : **ne se confirme
pas** — Spearman(Γ, t_grok) = +0.4 sur les 4 grokkers v3. Prédicteur de
DESTIN : oui (pré-enregistré, 8/8). Prédicteur de DATE : non. Caveat : les
fenêtres diffèrent en taille entre fracs (comparaison inter-frac des moyennes
délicate) ; un claim de date exigerait un design dédié à frac fixe, n ≥ 8.

## Statut Γ sur l'échelle de preuve (PROTOCOLE_DISCRIMINANT)

- L1 existence : v2 seeds frais ✓ (3/3 vs 0/2, contrôles 0.0000)
- L2 dose-réponse : v3 monotone frac ✓
- L3 causal : non fait (exigerait intervention sur le germe — ablater les
  directions Γ-positives et vérifier que le grok recule)
- L4 généralité : 1 tâche, 1 archi — sparse parity + MLP = prochain
- L5 échelle : Γ sur fine-tune LLM réel = l'enjeu (early-warning capacité)

---

# Γ L3 — ablation causale : KILL. Γ lit le germe, ne le tient pas (2026-06-10)

Code : `experiments/gamma_L3_ablation.py` (+ .json). 6 runs frac=0.5 wd=1.0
seeds 20-21, steps=4500. Chasse en ligne : détection Γ tous les 50 steps
(buffer 6, identique v1), projection x_eq ← x_eq − UUᵀx_eq (k ≤ 8, train+eval).
Contrôles matchés : k directions aléatoires / k directions du même pool aux
pentes les plus basses.

## Résultat (pré-enregistré : Δ_germ ≥ Δ_contrôles + 300 exigé)

| seed (baseline) | germ | random | lowslope |
|---|---|---|---|
| 20 (1250) | 1350 (**+100**) | 1250 (+0) | 1250 (+0) |
| 21 (1150) | 1300 (**+150**) | 1250 (+100) | 1150 (+0) |

**KILL** : Δ_germ ≤ max(Δ_contrôles) + 100 dans les 2 seeds.

## Lecture

**Le germe ne vit pas dans les activations — il vit dans les poids.** Les
directions Γ sont l'ombre du circuit en formation, pas sa substance : on
projette hors de 8 directions sur 128, le réseau ré-exprime le circuit dans
le complément orthogonal en ~1 refresh (50 steps). Whack-a-mole gagné par le
réseau. Cohérent avec Nanda (le circuit est dans les poids de Fourier) et
avec l'échec symétrique de v0 (l'amplitude d'activations ne porte pas non
plus le signal — seule la *réallocation* le portait).

Signal de friction suggestif (sous la barre, n=2) : germ = seule condition
à retard positif dans les 2 seeds (+100, +150). Une vraie poignée causale
existerait peut-être à cadence de chasse plus rapide (every=10) ou en
ablation CUMULATIVE (mais risque lobotomie triviale) — non testé.

## Statut final Γ (échelle de preuve)

- L1 existence ✓ (v2 seeds frais, 3/3 vs 0.0000)
- L2 dose-réponse ✓ (v3 frac monotone, séparation destin 8/8 à wd fixe)
- **L3 causal ✗ au niveau ACTIVATIONS** — Γ = instrument de lecture
  (forecaster), pas levier de contrôle. Le test causal correct devrait
  intervenir en espace POIDS (ablater les composantes de poids qui
  alimentent les directions croissantes) — autre expérience, plus invasive.
- L4 généralité / L5 échelle : ouverts.

**Identité finale de Γ : un sismographe, pas un barrage.** Il prédit le
destin (8/8) et la formation (~600 steps d'avance) sans pouvoir l'empêcher
par projection d'activations. Pour un early-warning (l'usage visé — LLM,
canal 2), c'est exactement ce qu'on demande à l'instrument.

---

# Γ L4 — GÉNÉRALITÉ : KEEP sur sparse parity + MLP (2026-06-10)

Code : `experiments/gamma_L4_parity.py` (v1, échec testbed) +
`gamma_L4_parity_v2.py` (chasse au plateau + test). Instrument INCHANGÉ
(ranks 4..32, fenêtre 6, θ q95 null, Frobenius) — 2e tâche ((30,3)-parity),
2e archi (MLP ReLU 256), 2e dim d'activations (256 vs 128).

## v1 : échec de testbed instructif

Full-batch parity = transition rasoir en n_train : N≤1000 jamais (20k steps),
N≥2000 généralise INSTANTANÉMENT (t_grok 100-200, aucun plateau → fenêtres
vides → NUANCE). Le grokking vit dans une bande étroite près du seuil
d'échantillons (Barak 2022) — le pilote v2 l'a localisée : N=1300 → grok 6800
(s5) ; N=1500 → 1300 ; N=1700 → 300.

## v2 : test frais pré-enregistré — KEEP 6/6

| run (seeds 30-31 frais) | destin | fenêtre | Γ_window |
|---|---|---|---|
| N=1300 ×2 | delayed grok @1700 | [150, 1600] | **0.0028, 0.0047** |
| N=1000 ×2 | jamais | [150, 5400] | 0.0000, 0.0004 |
| N=500 ×2 | jamais | [150, 5400] | 0.0000, 0.0001 |

min(delayed) = 0.0028 > max(no-grok) = 0.0004 (7×). Conservateur : fenêtres
no-grok 3× plus longues (plus d'occasions de faux positif) et ≈ 0 quand même.
Amplitudes plus faibles que modular-add/transformer (échelle dépend du
couple tâche×archi) mais le plancher zéro des no-grok est net.

## Statut final Γ (échelle de preuve complète)

- **L1 existence ✓** (modadd/transformer, seeds frais, ctrl 0.0000)
- **L2 dose-réponse ✓** (frac sweep à wd fixe, destin 8/8)
- **L3 causal-activations ✗** (sismographe, pas barrage — germe dans les poids)
- **L4 généralité ✓** (parity/MLP, mêmes hyperparamètres d'instrument, 6/6)
- L5 échelle réelle : ouvert — le prochain et dernier barreau.

---

# Γ L5 — l'échelle réelle : LA LIMITE TROUVÉE (2026-06-10)

## L5b — induction head (2L vs 1L attn-only) : NUANCE directionnelle

Code : `gamma_L5b_induction.py`. v1 = échec testbed (copie à offset FIXE =
raccourci positionnel, 1L résolvait sans induction). v2/v3 = copie à décalage
ALÉATOIRE par séquence → contenu obligatoire → théorème 1L s'applique :
**1L ne bascule jamais 8/8 (plafond 0.077 = skip-trigrams), 2L bascule
toujours (1125-1975)** — l'impossibilité architecturale tient parfaitement.
Γ : **7/8 paires directionnelles** (Γ(2L) > Γ(1L) même fenêtre), mais ratio
moyen 1.63 < barre 3.0 → NUANCE. Cause identifiée : le 1L n'est pas un
contrôle silencieux — il FORME ses skip-trigrams en continu et Γ l'entend.

## L5a — pythia-70m full fine-tune, modular-add en texte (Modal) : KILL inversé

Code : `modal_gamma_llm.py` (+ `modal_gamma_llm_results.json`,
`gamma_L5a_verdict.json`). Calibration 3 rounds (~$3 A10) :
LoRA r=16/64 ne mémorise JAMAIS (val colle au train — transfert sans
sur-capacité) → abandon. **Full FT lr=1e-4 : pythia-70m pré-entraîné GROKE
l'arithmétique modulaire en texte** (wd=1.0 frac 0.3 : t_mem 800 →
t_grok 6300 ; frac 0.2 : no-grok — finding de testbed en soi).

Test pré-enregistré (8 runs frais, wd=1.0 fixe) — **séparation INVERSÉE** :

| destin | Γ_window (max couches 2/4/6) |
|---|---|
| grokkers (3) | 0.000, 0.001, 0.257 |
| no-grok (5) | 0.002, 0.127, 0.292, 0.313, **0.523** |

## La limite générale (le vrai résultat de L5)

**Γ v1 détecte toute réallocation structurelle cohérente — pas spécifiquement
le germe de LA capacité. Il suppose un substrat au repos.**

- From-scratch jouets : la seule structure qui se forme = le circuit de
  tâche ; contrôles structurellement silencieux → séparations rasoir (L1-L4).
- Substrat vivant : wd=1.0 érode en continu 70M de poids pré-entraînés →
  réorganisation massive permanente → Γ énorme chez les no-grok (érosion max
  à frac basse), germe noyé chez les grokkers. L5b en version douce : les
  skip-trigrams du 1L diluent le contraste (7/8, ratio 1.6).

Identité finale : **Γ = sismographe de réallocation structurelle.**
Spécifique au destin ssi le fond est calme. Pour l'early-warning sur
modèles réels, trois voies v2 :
  (i) fond calme : wd modéré (test2 wd=0.1 en cours — condition de validité),
  (ii) **différentiel jumeau** : Γ(run) − Γ(run jumeau sans données tâche)
      = réallocation EXCÉDENTAIRE attribuable à la tâche (élégant, 2× coût),
  (iii) germe task-projeté (frame-full léger).

## test2 — fond calme (wd=0.1) : mécanisme CONFIRMÉ, détection ERRATIQUE

8 runs frais (seeds 40-41), wd=0.1 fixe. Destins propres : frac 0.2/0.3 ×4
no-grok ; frac 0.5 grok @7200 ×2 ; frac 0.6 grok @1700 (exclu, plateau nul)
et @3700.

| | Γ_window |
|---|---|
| no-grok ×4 | **0.0014 – 0.0023** (vs 0.13-0.52 à wd=1.0 — fond ÷100) |
| grokkers ×3 | 0.0003 / 0.0502 / 0.1722 |

1. **Lecture érosion CONFIRMÉE** : à wd=0.1 le fond no-grok s'effondre ~100×.
   La limite de L5a était bien environnementale (réorganisation wd du substrat).
2. **Mais détection erratique** : 2/3 grokkers entendus (25-86× le fond),
   1 sourd total (s40 frac 0.5 : 0.0003, zéro montée sur 6000 steps de
   plateau). Séparation binaire : FALSE à cause de ce run.
3. **Timing révélé par les séries** : s41 (jumeau de s40, grok au même step
   7200) = silence 5000 steps PUIS explosion 0.52→1.04 à 6500-7000 (~700
   steps avant grok). **Le germe LLM surgit tard et abruptement** — pas la
   croissance graduelle des jouets — ou vit hors de la sonde (on n'écoute
   que la DERNIÈRE position du prompt, couches 2/4/6, ranks 4-32 ; le
   circuit arithmétique vit plausiblement aux positions opérandes / têtes
   d'attention / MLP intermédiaires).

## Verdict final L5 : NUANCE ferme, périmètre tracé

Γ v1 ne transfère pas tel quel au LLM réel. Deux conditions identifiées :
(1) **fond calme** (wd modéré) — nécessaire, validée, PAS suffisante ;
(2) **sonde spatiale adéquate** — la position/couche d'écoute devient le
facteur limitant dans un transformer (vs MLP/jouets où « les activations »
sont un seul endroit). Γ v2-LLM = multi-position × multi-couche (× têtes).

---

# Axes alignement — A2 + A4 (2026-06-10, après-midi)

Cadre : `theory/ALIGNMENT_AXES.md`. Coût Modal cumulé ~$6.

## A4 — audit de bifurcation : KEEP fort (gratuit, offline)

`gamma_A4_bifurcation.py` sur les 6 transformers grokés (poids déjà sauvés).
**Accord comportemental 1.0000 EXACT (15 paires × 9409 entrées) — et
mécanismes quasi-disjoints** : empreintes de Fourier des embeddings
Jaccard moyen **0.083** (s1:{10,18,22,33} vs s2:{2,9,39} vs s12:{7,10,16,28,42}…),
CKA représentationnel 0.127. La question ouverte de germination.md est
tranchée : **l'individualité naît à la bifurcation — même comportement,
mécanismes différents.** Plafond de principe de l'éval-based certification,
démontré sur testbed contrôlé.
Finding latéral : slingshot s0 (val 1.00 → 0.51 entre steps 2750→3000 sous
wd=1.0) — la généralisation acquise peut s'effondrer ; checkpoint final ≠
checkpoint groké.

## A2 / Γ v2 (test3, sonde 28 canaux = 4 positions × 7 couches) : KILL → pivot poids

Pré-enregistré dans `modal_gamma_llm.py::test3`. Mêmes conditions que test2.
- s41 frac 0.5 : signal ×3.6 vs v1 (0.626, canal L3_p1 — couche 3, position
  de l'OPÉRATEUR « + ») ; fond no-grok stable 0.0014-0.0054.
- **s40 frac 0.5 : sourd sur LES 28 CANAUX (0.0003)** — même config, même
  destin, même t_grok que s41.
- H-v2a (séparation) : False. H-v2b (aucun sourd) : False → KILL
  pré-enregistré : « le germe n'est pas [toujours] dans les activations
  d'un LM pré-entraîné → pivot poids / jumeau ».

**Lecture (connexion A4×A2, le finding du jour)** : l'individualité de
bifurcation frappe aussi le CHEMIN de formation — il existe des routes de
formation de capacité **silencieuses en activations** (s40) et des routes
bruyantes (s41), vers le même circuit final. Implication alignement dure :
un moniteur d'activations, même multi-position/couche, peut rater une
fraction substantielle (~1/3 ici) des formations de capacité sur LM
pré-entraîné. L'early-warning fiable exige l'espace POIDS (Γ-poids :
exposants des directions de Δw — L3 disait déjà que le germe y vit) ou le
différentiel jumeau.

## test3 — Γ v2 sonde élargie (28 canaux) : KILL (b), et la carte du germe

8 runs frais (mêmes conditions test2, probe_mode="all4" : 4 positions
[a][+][b][=] × 7 couches). Trajectoires identiques test2 (déterminisme ✓).

**Quand le germe est audible, la sonde élargie le LOCALISE et l'amplifie** :
s41 frac 0.5 passe de Γ=0.172 (v1, position '=') à **0.626 fenêtre / pic 7.39**
(×1400 le fond) — canal **L3_p1 (couche 3, position '+')**, puis propagation
L3/L4_p0 (opérandes) → L6_p3 (sortie) sur 6300→7100, ~900 steps avant grok.
Première carte spatio-temporelle de germination dans un LM : le circuit naît
au MILIEU du réseau sur l'opérateur, migre vers la sortie. Cohérent Nanda
(structure de Fourier sur les représentations des nombres) — et la sonde v1
('=' seul) regardait bien au mauvais endroit.

**Mais s40 (jumeau exact : même frac, même t_grok 7200) : 0.0000 absolu sur
les 28 canaux — y compris PENDANT et APRÈS sa propre transition.** Un grok
sans aucune réallocation détectée même à la bascule → sa formation passe par
les **rangs dominants 1-3, exclus par construction** (l'exclusion
anti-dominance devient aveuglement quand le circuit se forme DANS les
directions déjà dominantes). No-grok : fond ≤ 0.0054 (propre).

Verdict pré-enregistré : **(a) FALSE, (b) FAIL** — early-warning par
activations NON fiable run-par-run sur LM pré-entraîné avec cette sonde.

## Lecture finale L5 (3 couches de vérité)

1. Le germe LLM **existe et est cartographiable** quand il vit dans les rangs
   moyens — signal ×100-1400 le fond, localisé, propagatif. La promesse
   early-warning n'est pas morte dans l'absolu.
2. Les **chemins de formation sont hétérogènes entre seeds** : rangs moyens
   (s41, audible) vs rangs dominants (s40, invisible à un instrument qui
   exclut le top-3 pour se protéger de la dominance). Le trade-off
   anti-dominance ↔ couverture est structurel, pas réglable trivialement.
3. Pour un early-warning fiable : Γ v3 devrait séparer « dominance
   existante » de « dominance croissante » (tous rangs, croissance relative
   par direction) — ou écouter les POIDS (où L3 a montré que le germe vit).

---

# Γ UNIVERSALITÉ — KILL du claim fort, naissance des « classes » (2026-06-10)

Code : `experiments/gamma_universality.py` (+ .json, .png). Analyse offline
sur checkpoints existants, 24 s, coût zéro. Question : la phase préparatoire
a-t-elle des invariants cross-système (à la « exposants critiques ») ?
Statistique : distributions des excès de croissance des germes (pente − θ,
par step, × durée de fenêtre), KS pairwise vs plancher intra-testbed.

## Verdict pré-enregistré : KILL

| paire | KS rescalé | KS brut |
|---|---|---|
| modadd ↔ induct | **0.246** | **0.169** |
| modadd ↔ parity | 0.378 | 0.552 |
| parity ↔ induct | **0.610** ☠ | 0.668 |

Pas d'invariant UNIVERSEL : parity↔induct ≥ 0.40. Le claim fort (ChatGPT :
« l'émergence a des invariants ») meurt tel que testé.

## Ce que les décombres contiennent (2 findings)

1. **Quasi-collapse intra-famille attention** : modadd (transformer) ↔
   induction (attn-only) à KS 0.17-0.25 sur n=357 vs n=2786 — les deux
   testbeds à attention partagent quasi la même distribution de taux,
   SANS rescaling (le brut fait mieux que le rescalé : les taux par step
   sont déjà comparables). → Hypothèse raffinée, testable : **des CLASSES
   d'universalité par famille architecturale** (comme en physique : Ising
   et percolation ont chacun leurs exposants), pas une universalité absolue.
   Test futur : 2e tâche/attention + 2e tâche/MLP, cheap.
2. **La LARGEUR de germination est un trait de testbed** : induction =
   germination large (210-488 directions croissantes/run), modadd = moyenne
   (20-58), parity MLP = ultra-éparse (8-9). Nouvelle dimension d'observable :
   la sparsité du germe. La non-universalité est réelle au niveau largeur.

## Caveat statistique du KILL

Le KILL est porté par parity : n=17 germes au total (2 runs), intra-parity
KS = 0.31 (sa propre distribution est à peine estimable). Verdict formellement
valide (0.61 sans ambiguïté), mais avant tout claim « les MLP sont une autre
classe », re-runner ~5 seeds parity (gratuit, ~15 min CPU). La paire bien
échantillonnée (modadd↔induct) raconte l'histoire la plus fiable — et elle
dit « presque collapse ».

---

# Γ-POIDS — le pivot post-test3 (2026-06-10, fin de journée)

L3 le prédisait (le germe vit dans les poids), test3 l'a exigé (routes
silencieuses en activations). Même machinerie d'exposants, appliquée aux
MATRICES DE POIDS (lignes = tokens/neurones ; Frobenius-norm absorbe le
decay global du wd → mesure la réallocation INTERNE).

## Phase jouets (gratuite, offline — `gamma_Aw_toys.py`) : KEEP 6/6 vs 4/4

Poids déjà sur disque (10 runs modadd). Fenêtre v2 identique.
Γw grokkers = [0.0013…0.0148], contrôles = [0.0000, 0.0000, 0.0000, 0.0003].
**Et le canal gagnant est `blocks.0.attn.in_proj` chez LES SIX grokkers** —
le germe en poids vit dans l'attention de la couche 0, PAS dans les
embeddings où Nanda voit le circuit final. **La formation et le produit fini
n'habitent pas le même endroit.** (tok.weight : silencieux pendant la
formation.)

## Phase pythia (`modal_gamma_llm.py::test4`, 6 tenseurs qkv+mlp couches 0/2/4)

| run | destin | Γw (max tenseurs) |
|---|---|---|
| frac 0.5 s41 | grok @7200 | **0.0095** — canal `L0.mlp` |
| frac 0.5 s40 | **pas de grok CE run** | 0.0000 (auto-cohérent) |
| frac 0.2 ×2 | jamais | 0.0000, 0.0000 |

- **Fond poids = ZÉRO EXACT** (12 canaux × 3 no-grok) — plus propre que les
  activations (fond 0.0014-0.0054). Le wd=0.1 n'érode pas assez pour
  fabriquer de la réallocation interne parasite dans les poids.
- **Le grokker est entendu, canal couche 0 MLP** — même pattern de
  localisation précoce que les jouets (couche 0). 2/2 substrats : le germe
  vit TÔT dans le réseau.
- **H-w (s40-le-sourd bruyant en poids ?) : NON TESTABLE** — le destin de
  s40 a BASCULÉ entre test3 et test4 (grok @7200 → pas de grok, même seed,
  même config ; non-déterminisme GPU près du seuil, grok à 93% du budget).
  Finding latéral important : **à frac proche du seuil, le DESTIN lui-même
  est instable** — la susceptibilité divergente frappe l'issue, pas
  seulement le mécanisme (cohérent bifurcation A4). Et relecture de
  test2/test3 : le « silence » de s40 était peut-être une formation
  ULTRA-TARDIVE (dans les 200 derniers steps, hors fenêtre par construction)
  plutôt qu'une route invisible — censure de fenêtre + budget 8000 trop
  juste pour frac 0.5.

## test5 (décisif, steps=12000, destins décensurés) : KILL pré-enregistré

4 grokkers (s40@7200 — destin re-basculé, s41@7200, s42@4200, s43@8900 —
invisible à l'ancien budget), 2 contrôles. Γw : s41 = 0.0099 ; **s40, s42,
s43 = 0.0000 EXACT** (plateaux 3000-7700, fenêtres larges, fond contrôles
zéro). H-w5b morte : 3/4 grokkers sourds EN POIDS.

## test6 (ferme l'échappatoire « sonde incomplète ») : KILL DÉFINITIF 0/3

Sonde complète 12 tenseurs (qkv+mlp × LES 6 couches), mêmes seeds.
**Les 3 sourds restent à zéro exact sur les 12 tenseurs.** Et fenêtres
étendues ([t_grok−600, t_grok) et [t_grok, t_grok+1500)) : toujours rien —
**pas de formation éclair ratée, pas de cleanup audible. Invisible partout,
tout le temps.**

Contre-portrait s41 (le concentré) : 8/12 tenseurs actifs pré-grok
(L0.mlp 0.054 > L2 > L5, gradient de profondeur), **extinction instantanée
à la bascule** (post-grok 0.0000) → en poids, Γ trace la FORMATION
(vs s_net activations qui traçait la CONSOLIDATION — deux instruments,
deux phases du devenir).

## CONCLUSION DE L'ARC Γ (le résultat alignement principal)

**Il existe deux classes de routes de formation dans un LM pré-entraîné :
CONCENTRÉES (localisées multi-couches, détectables — 1/4 ici) et DIFFUSES
(réallocation distribuée sous le seuil de chaque matrice, indétectables par
exposants localisés — 3/4 ici), en poids COMME en activations.**

Hypothèse mécaniste (falsifiable, next) : le PRÉ-ENTRAÎNEMENT offre les
routes diffuses — recâblage léger de features partielles existantes,
réparti partout. Les jouets from-scratch n'ont rien à réutiliser → formation
forcément concentrée → 6/6 audibles. Prédiction : plus le modèle est
pré-compétent sur le domaine, plus la formation est diffuse → moins
détectable.

**Implication monitoring interne (l'asymétrie fondamentale)** :
- vrais positifs précieux : quand Γ sonne, c'est réel (fond ZÉRO partout,
  jamais un faux positif poids sur 5 contrôles × 12 canaux) ;
- **AUCUNE garantie négative : le silence ne prouve rien** (75% des
  formations naturelles inaudibles ici).
Un moniteur interne de cette famille = alarme fiable, all-clear NON fiable.
Corollaire pour A3 (évasabilité) : un adversaire n'a même pas besoin de se
cacher — les routes furtives existent naturellement ; la question A3 devient
« peut-on FORCER la route concentrée ? » (régularisation pro-localisation
comme outil de sécurité — rendre le modèle monitorable by design).

Coût Modal total session : ~$10 / cap $20.

---

# Consolidation a/b/c (2026-06-10) — universalité, figures, Γ v3

## (a) Universalité ×7 seeds parité : KILL confirmé, classes consolidées

`gamma_universality.py` re-run avec 7 seeds parité (32-36 ajoutés). Verdict
inchangé : **KILL** (parity↔induct KS=0.58 rescalé). Mais les décombres se
solidifient :
- **Quasi-collapse intra-famille attention ROBUSTE** : modadd↔induct KS 0.17
  (brut) / 0.25 (rescalé), n=357 vs 2786. Les deux architectures à attention
  partagent quasi la même loi de germination.
- **La sparsité MLP est une PROPRIÉTÉ, pas un sous-échantillonnage** : 7 seeds
  parité donnent toujours 3-12 germes/run (vs 210-488 induction, 20-58 modadd).
  Le germe MLP est une *aiguille*, le germe attention une *nappe*. Intra-parity
  KS=0.36 (sa distribution reste à peine estimable — c'est intrinsèque).
- Finding net : **pas d'universalité absolue, mais des signatures
  architecturales** (largeur + distribution). Hypothèse « classes
  d'universalité par famille » vivante, à tester (2e tâche/attention + 2e/MLP).

## (b) Deux livrables visuels

- `experiments/fig_three_phases.png` — 6 seeds modadd alignés sur la bascule.
  **Lecture honnête (la forme réelle, pas le récit simpliste)** : avant le
  grok, Γ porte seul le signal (bump −500→−100), s_net ≈ 0 ; après, s_net
  grimpe (0.03→0.63 = consolidation). Γ **refire au cleanup** (+300, +800) —
  assumé : Γ est un mètre de réallocation, et le nettoyage post-grok en est
  une. Les trois temps sont réels ; la phase 3 n'est pas silencieuse pour Γ.
- `viz/germination.html` — film WebGL (Three.js r128, fond noir émissif, HUD
  mono, scrubber + autoplay). Données réelles s41 : 28 canaux (7 couches × 4
  positions), le germe s'allume couche 3 position « + » (pic 7.39), migre vers
  la sortie, val_acc en sparkline. Pièce portfolio + figure animée paper.

## (c) Γ v3 (rangs dominants inclus) : KILL — le germe muet n'est PAS dans les activations

`modal::test7` (rank_lo=0, budget 12000). Hypothèse : le jumeau sourd s40
(0.0003 en v2) cacherait son germe dans les rangs dominants 1-3 exclus.
**Réfuté** : avec les dominants inclus, s40 = **0.0010** (toujours muet),
s41 = 0.699 (toujours fort, sanité OK). Inclure les dominants ne réveille rien.

**Conséquence forte** : la formation de s40 est invisible dans les activations
*entières* (rangs moyens ET dominants, 28 positions×couches). Le germe vit
ailleurs — dans les **poids** (cohérent L3 : le germe est poids-spatial,
les activations n'en sont que l'ombre, et certaines formations ne projettent
aucune ombre détectable). Élimine une hypothèse, pointe fermement la route
poids (`probe_mode="wtensors"/"wfull"` — Γ-poids).

**Le paradoxe des jumeaux, raffiné** : deux réseaux, même tâche, même destin,
mais l'un forme son circuit dans des directions qui se projettent en
activations (audible), l'autre dans des directions qui ne s'y projettent pas
(muet). L'individualité de formation est réelle et profonde — et la seule
sonde qui pourrait les unifier est celle des poids.

## Prochaines marches (post-L5 complet)

1. Γ v2-LLM : sonde multi-position (opérandes + '=' + dernière) × toutes
   couches ; re-test sur les mêmes runs ? NON — acts non sauvées côté Modal
   (calcul in-job). Re-run test2 avec sonde élargie (~$1.5) quand voulu.
2. Γ v2 différentiel jumeau (réallocation excédentaire) — l'autre voie.
3. PAPER : l'arc Γ L1→L5 est une histoire complète et publiable — un
   instrument né d'une question philosophique, 5 barreaux d'échelle de
   preuve, 6 testbeds, kill-switches pré-enregistrés à chaque étage,
   périmètre final exact (sismographe de réallocation, substrat au repos,
   sonde-dépendant). Candidat : workshop NeurIPS interp / ICLR SciForDL.
4. s_net post-hoc parity gratuit ; MC3 ; latent-flow ; Ξ niveau MON.

---

# (a) Pré-compétence → diffusion : CONFIRMÉ 4/4 vs 1/4 (2026-06-10, ~14h)

Design décisif : pythia-70m **réinitialisé aléatoirement** (`init_random=True`)
— même archi, même tâche, même pipeline, zéro pré-compétence par construction.
`modal_gamma_llm.py::pilot_ab/test_ab`, tags pilotA_rand/testA_rand.

| run (random-init, lr=1e-3) | plateau | canaux audibles /12 | max Γw |
|---|---|---|---|
| s50 (pilote) | 700 | **10** | 0.050 |
| s52 | 2600 | 1 | 0.0029 |
| s54 | 6000 | **12** | 0.024 |
| s55 | 1000 | **12** | 0.086 |
| s51, s53 | exclus (grok instantané, plateau < 500) | — | — |

**4/4 grokkers from-scratch audibles** (critère Γw>0.002 sur ≥1 tenseur ;
3/4 hurlent sur 10-12 tenseurs) **vs 1/4 pré-entraîné** (test5/6).
**H-préco KEEP : le pré-entraînement est la source des routes diffuses** —
le from-scratch construit du neuf (chantier localisé multi-tenseurs,
audible), le pré-entraîné recâble l'existant (diffus, silencieux).
Implication frontière : plus un modèle est pré-compétent, moins ses
apprentissages sont monitorables par exposants internes.

# (b) Régularisation pro-localisation (group-lasso ΔW) : KILL au pilote

λ ∈ {1e-4, 1e-3, 3e-3, 1e-2} (s50, pré-entraîné, frac 0.5) : grok préservé
partout (et ACCÉLÉRÉ par λ fort : 7400 → 2400 — l'ancrage à l'init
régularise comme le wd) mais **audibilité nulle** (0-1 canal limite à
0.002). La group-lasso Σ‖ΔW_m‖_F pénalise le VOLUME de changement par
matrice, pas sa STRUCTURE : un recâblage diffus de rang plein la satisfait
sans produire de direction croissante détectable.
**Voie suivante (non dépensée)** : pénalité SPECTRALE — nuclear norm ‖ΔW‖*
(pousser le changement vers le rang faible = fabriquer la direction
concentrée que Γ écoute). Monitorabilité by design reste ouverte, la
première serrure n'était pas la bonne.
