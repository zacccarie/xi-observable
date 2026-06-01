# Branchement sur le réel — synergie à travers la profondeur d'un LLM

Premier branchement de l'instrument émergence (Ω, O-information) sur un système
qui compte, après calibration sur jouets (Ising/Kuramoto/RBN/CML). `llm_depth_synergy.py`.

## Setup

Pythia-160m, residual stream à travers les 13 états (embedding + 12 couches).
Depth = trajectoire ; dimensions du résidual = variables. Corpus 4992 tokens
(fenêtres de 128). Ω sur sous-ensembles de 64 dimensions (subset 4, n_bins 3).
Surrogate = shuffle par dimension (détruit corrélations inter-dim, garde marginales).

## Finding

| couche | excess Ω (réel − surrogate) |
|--------|------------------------------|
| 0 (embedding) | **−0.158** (synergie forte) |
| 1 | −0.042 |
| 2 | −0.022 |
| 3-12 | ≈ −0.008 (plateau, ≈ surrogate) |

surrogate ≈ −0.009 partout (biais d'estimation négligeable à 4992 tokens).

**L'embedding porte l'essentiel de la synergie ; le calcul en profondeur la réduit
à un plateau marginal.** Profil monotone décroissant, chute brutale couche 0→2.

## Interprétation (contre-intuitive)

Prédiction initiale (fausse) : embedding = lookup redondant, couches de calcul =
synergiques. **L'inverse est observé.**

> L'embedding encode le token de façon DISTRIBUÉE et irréductible — l'identité du
> token n'est récupérable que dans la conjonction des dimensions (synergie max).
> Les couches **déplient** progressivement cette information en features de plus en
> plus linéairement décodables (lisibles dimension par dimension) → la synergie
> chute. **Le transformer convertit l'information irréductible (synergique) en
> information accessible (redondante/linéaire).**

Cohérent avec l'hypothèse de représentation linéaire en interprétabilité mécaniste :
les couches profondes rendent les features linéairement séparables, donc moins
synergiques.

**Conséquence pour la thèse** : l'émergence (irréductibilité) n'augmente PAS avec le
calcul — elle est maximale à l'entrée, et le réseau la résout. Φ d'un LLM (émergence
persistante) vit surtout dans l'embedding.

## Caveats

- 1 modèle (pythia-160m), 1 corpus (technique, anglais), 1 seed de dimensions
- couches profondes : excess (−0.008) ≈ surrogate (−0.009) → marginal, peut-être bruit
- signal ROBUSTE seulement pour la couche 0 (16× le surrogate)
- subset de 64/768 dimensions

## Solidification requise

- multi-seed sur le choix des dimensions + bootstrap
- plusieurs tailles (pythia-410m 24 couches, 1.4b) : le profil tient-il ? la chute
  se déplace-t-elle ?
- corpus naturel varié (pas seulement du markdown technique)
- tester si la synergie de l'embedding est portée par les positional vs token embeddings

## Génération autorégressive — édge of chaos par température (`llm_temperature_edge.py`)

La génération token-par-token est la VRAIE dynamique du LLM. Température T =
paramètre de contrôle (T basse → répétitif/ordre ; T haute → incohérent/chaos).
Trajectoire = hidden states (couche 6) du token généré. Sweep T ∈ [0.3, 2.0].

| T | répétition | excess Ω |
|---|------------|----------|
| 0.3 | 0.76 | **−0.032** (max synergie) |
| 0.7 | 0.58 | −0.013 |
| 1.3 | 0.14 | ≈0 |
| 2.0 | 0.01 | −0.001 |

**Prédiction RÉFUTÉE** : la synergie ne pique PAS à l'edge — elle suit l'ORDRE
(max à T basse répétitive, →0 en chaos). Profil monotone, pas de pic.

**Diagnostic** : Ω inter-dimension AGRÉGÉ (samples = tokens, ordre temporel ignoré)
mesure la **cohérence/confinement statique** des représentations, pas la dynamique.
L'edge of chaos est une propriété DYNAMIQUE (sensibilité temporelle) → cette mesure
la rate. Cohérent avec le finding depth (synergie suit l'ordre : embedding, basse T).

**Inversion vs jouets** : RBN/CML → synergie émerge dans le CHAOS (irréductibilité
dynamique) ; LLM activations → Ω suit l'ORDRE (structure représentationnelle). Même
métrique, objets différents. La distinction « deux synergies » (cf A, niveau fonction
vs réseau) frappe encore : **Ω inter-dim statique ≠ synergie émergente dynamique.**

**Leçon** : brancher Ω naïvement sur activations agrégées mesure la cohérence
représentationnelle, pas l'édge of chaos. Pour l'edge génératif → mesure DYNAMIQUE :
Lyapunov de la trajectoire de génération vs T, dimension effective Koopman (comme CML),
ou autocorrélation temporelle des hidden states.

## Mesure DYNAMIQUE — l'edge génératif trouvé (`llm_generative_dynamics.py`)

Ω statique ratait l'edge. Mesure dynamique (DMD de la trajectoire de génération vs T) :
dimension effective Koopman, ⟨|λ|⟩ persistance, entropie spectrale.

| T | répétition | dim_eff | ⟨|λ|⟩ | H_spec |
|---|-----------|---------|-------|--------|
| 0.3 | 0.59 | 20 | 0.72 | 0.17 |
| 0.7-1.1 | 0.39→0.12 | 27→35 | **0.65 (min)** | 0.14 (plat) |
| **1.3** | 0.06 | 44 | 0.74 ↑ | **0.31 (saut ×2)** |
| 1.6-2.0 | 0.01 | 51 | 0.80 | 0.40 |

**Edge of chaos génératif = transition de phase dynamique à T≈1.2.** QUATRE observables
convergent : répétition atteint son plancher, dim_eff inflexion, ⟨|λ|⟩ saute du min au
plateau, H_spec ×2. En dessous : régime cohérent (variété confinée, modes fluides) ;
au-dessus : chaotique (dim saturée, modes random). Sweet spot génération qualité
(T≈0.7-1.0, ⟨|λ|⟩ min) = régime cohérent juste avant l'edge.

**La mesure dynamique RÉUSSIT là où la statique échouait.** L'edge existe
dynamiquement dans le LLM ; l'émergence (synergie) n'y pique pas (elle suit la
cohérence statique). Signature de l'edge = transition spectrale, pas pic de synergie —
cohérent avec B v1 (émergence présente mais fugace à l'edge). Confirme : **edge of
chaos = propriété DYNAMIQUE, invisible à Ω statique.**

Caveat : pythia-160m, couche 6, 8 prompts, 1 seed. Signal net (4 observables) mais
modeste en taille. Solidif : multi-seed, multi-couche, multi-taille.

## Statut

L'instrument fonctionne sur du réel. TROIS relevés LLM : (1) depth — embedding synergique
(pas le calcul) ; (2) génération statique — Ω suit l'ordre (rate l'edge) ; (3) génération
dynamique — edge of chaos à T≈1.2 (transition spectrale, 4 observables convergentes).
Pont jouets → réel établi et bouclé. Acquis : **edge of chaos = dynamique ; Ω statique
inter-dim = cohérence représentationnelle. Deux observables distincts, deux phénomènes.**
