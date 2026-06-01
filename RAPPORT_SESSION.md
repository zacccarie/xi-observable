# Rapport de session — Programme Ξ / Émergence

*Compte-rendu exhaustif : protocole expérimental, findings, réfutations, conclusion.*
*Date : 2026-06-01. Repo : `xi-observable`. ~22 expériences, 8 systèmes, 5 fichiers théorie.*

---

## 0. Abstract

Tentative de rendre l'**émergence** mesurable comme **information irréductible** (synergie,
O-information Ω < 0), et de tester l'hypothèse historique « l'émergence vit à l'edge of
chaos ». Résultat central : **la synergie est rare**. La quasi-totalité des systèmes
complexes (transitions de phase, synchronisation, réaction-diffusion, chaos spatio-temporel,
LLM) sont **info-théoriquement redondants** malgré leur complexité visible. La synergie
n'apparaît que pour des couplages **non-monotones** (type XOR), rares en nature. Mécanisme
établi : **émergence ≡ chaos**, deux faces d'une même propriété (la non-séparabilité de la
règle). Φ (information émergente persistante) = résidu rare, stabilisé en régime de chaos
modéré fortement couplé.

---

## 1. Cadre théorique

### 1.1 Tétrade fondamentale (D, V, C, μ)
Tout système = (Dimensions/degrés de liberté, Variables/observables, Couplages, Mesure
statistique). Toute quantité dérivée (entropie, Lyapunov, susceptibilité, Ω) = composition
de ces quatre. Les dimensions sont traitées **abstraitement** (temps, espace, paramètre,
profondeur = équivalents mathématiques).

### 1.2 Émergence = synergie irréductible
- **Redondance** (Ω > 0) : information dupliquée dans les parties.
- **Synergie** (Ω < 0) : information présente seulement dans le tout.
- O-information : Ω = TC − DTC = (n−2)H(X) + Σ[H(Xᵢ) − H(X₋ᵢ)].

### 1.3 Distinction triadique (acquis conceptuel)
**Corrélation** (TC) ≠ **criticalité** (χ, fluctuations) ≠ **émergence** (synergie). Le
framing initial « edge × interaction = Φ » supposait qu'elles coïncident. Elles divergent.

---

## 2. Méthodes générales (protocoles transverses)

| Protocole | But | Usage |
|-----------|-----|-------|
| **O-information exacte** | mesurer synergie/redondance | histogramme exact, validé XOR=−1/copie=+1/parité=−2 |
| **Surrogate shuffle** | séparer signal du biais d'estimation | shuffle par colonne (détruit corrélations, garde marginales) |
| **Test de neutralité CG** | valider un coarse-graining | passer du bruit i.i.d. ; bon CG → métrique plate |
| **Sanity analytique** | valider le pipeline | systèmes à régime connu (XOR, copie) |
| **Contrôle |λ| DMD** | détecter DMD mal conditionné | |λ| ~ 1 attendu ; ≫1 = SVD à tronquer |
| **Baseline i.i.d.** | corriger biais d'échantillonnage | soustraction γ_obs − γ_iid |

Stack : Python/numpy/scipy/matplotlib (`/opt/anaconda3/bin/python`), torch+transformers
(Pythia), Three.js r128 (viz). Plots dark-emissive.

---

## 3. Protocole expérimental détaillé (par système)

### 3.1 Ising 2D (transition thermo, classe conforme)
- **Méthode** : Metropolis checkerboard vectorisé, L=32-64, sweep T. χ = N·Var(m).
  Ω sur blocs 2×2 RG.
- **Stade A** : Ξ = mask_χ · TC pique à Tc=2.27 (Δ=+0.03 finite-size). Falsification (mask
  global) → SNR ∞ : Ξ=0 strict hors-bande. ✓
- **Stade B** : γ_T = pente log TC(k) vs échelle RG → ≈0 à Tc (scale-invariance). Après
  contrôle CG (`ising_control_cg`) : CG spatial biaisé (γ_iid≈+1.09), corrigé par baseline ;
  signature = min |γ_corrected| à Tc. ✓
- **Régime** : Ω > 0 **partout** → Ising critique = **redondance pure**. (R1)

### 3.2 Susceptibilité informationnelle (Ising)
- **Méthode** : 20 réalisations indépendantes par T ; Ω par réalisation ; std inter-réal.
- **Résultat** : std(Ω) pique à T=2.267 = Tc (Δ=−0.003), plus précis que χ (Δ=+0.064) et
  TC (Δ=+0.131). Pic étroit symétrique.

### 3.3 Kuramoto (synchronisation, critical slowing down)
- **Méthode** : RK4 N=64, all-to-all et 2D local, sweep K. order param r, χ=N·Var(r).
- **Réfutation artefact** : CG temporel circulaire (`circular_window_mean`) sur phases
  i.i.d. → TC croît 7× → **fabrique de la corrélation spurieuse**. γ_K invalide.
- **τ_corr** (autocorrélation intégrée de r(t)) : pic K=1.50 (Δ=−0.10 vs Kc=1.596), bat χ.
- **χ_info = CV(Ω)** : std(Ω)/⟨|Ω|⟩ localise Kc (Δ=+0.004).

### 3.4 RBN Kauffman (edge of chaos discret) — **système clé**
- **Méthode** : N=60, K∈[1,5], fonctions booléennes aléatoires, biais p=0.5 → K_c=2.
  Derrida slope = edge detector. μ = ensemble d'attracteurs (3000 CI).
- **Résultats** :
  - TC pique à edge K=2.
  - **Premier Ω<0 réel** (synergie) : phase chaotique K=3.
  - mean(Ω) change de signe (redondance→synergie) à la transition ordre→chaos.
  - **Susceptibilité informationnelle** : std_réseaux(Ω) pique K=2 (336× vs chaos profond).
  - **Surrogate** : |Ω|_K=2 excess 100× = signal réel ; bimodal (mélange syn/red) ; K≥4 =
    bruit (excess≈0).

### 3.5 Découplage émergence/criticalité (RBN à fonctions contrôlées) — **test décisif**
- **Méthode** : fonctions parité (synergie) vs canalisantes (redondance), sweep synergy_frac,
  régression Ω ~ Derrida + synergy_frac.
- **Résultat** : confound Derrida↔synergy_frac = **+0.999** → inséparables. **Émergence ≡
  chaos = même propriété** (la conjonctivité de la règle). Synergie nette max à l'edge
  (Derrida≈1), diluée en chaos profond.

### 3.6 Synergie vs sensibilité (256 fonctions booléennes K=3) — **durcissement**
- **Méthode** : énumération exhaustive ; Ω(inputs,output) vs average sensitivity par fonction.
- **Résultat** : corr = −0.95 quasi-déterministe ; **coin « forte synergie + basse
  sensibilité » VIDE**. Identification synergie≡sensibilité confirmée aux deux niveaux
  (fonction + réseau). Paramètre unique = **degré de conjonction de la règle**.

### 3.7 Φ = synergie persistante (Koopman)
- **B v0** (`koopman_phi`) : DMD tronqué (fix bug |λ|=6500 → SVD), Φ = persistant ∩ distribué.
  corr(|λ|,PR)=−0.07 → axes ⊥ → Φ = filtre 2D non-trivial. Dimension effective s'effondre à
  la sync.
- **B v1** (`phi_pid`, CML) : split modes persistant/transitoire, Ω par groupe. **Émergence
  préférentiellement fugace** (synergie dans les transitoires). Φ rare : modes persistants
  synergiques seulement en chaos modéré + couplage fort.

### 3.8 LLM (Pythia) — branchement sur le réel
- **Depth** (`llm_depth_synergy`) : Ω inter-dimensions par couche, 4992 tokens, surrogate.
  **Embedding (couche 0) synergique** (excess −0.16) ; couches profondes plateau marginal.
  Contre-intuitif : le calcul DÉPLIE l'irréductible en features linéaires.
- **Génération statique** (`llm_temperature_edge`) : Ω inter-dim vs température. **Suit
  l'ordre** (max T basse), pas l'edge. Ω statique = cohérence représentationnelle.
- **Génération dynamique** (`llm_generative_dynamics`) : DMD trajectoire vs T. **Edge of
  chaos à T≈1.2** : 4 observables convergent (répétition plancher, dim_eff 20→51, ⟨|λ|⟩ saut,
  entropie spectrale ×2). Mesure dynamique réussit là où statique échoue.

### 3.9 Réaction-diffusion Gray-Scott (conjecture interface)
- **Méthode** : Gray-Scott, Ω par patch vs gradient |∇v|.
- **Résultat** : conjecture « émergence aux interfaces » **réfutée**. frac_synergie=0 partout,
  interfaces PLUS redondantes (mouvement commun du front). **Complexité visuelle (labyrinthe
  Turing) ≠ émergence** : couplage diffusif (laplacien) = monotone = redondance.

### 3.10 Espace des paramètres (insight dimensions)
- **Méthode** : carte observable(r,ε) sur grille CML ; Ω entre observables ; |∇| interfaces.
- **Résultat** : carte **géométriquement structurée** (edge serpentant, fenêtres périodiques
  Feigenbaum, transition dim) MAIS Ω entre observables = +0.38 redondance. **Structure
  géométrique ≠ synergie.**

### 3.11 Autres (process parallèle)
- **Percolation 2D** : CV(Ω) localise p_c=0.59 (4e classe, géométrique).
- **Sandpile BTW (SOC)** : frac_synergie=1.0 — **SOC = synergie pure** (avalanches =
  événements collectifs irréductibles). 5e classe.
- **LZ-complexity vs χ_info** : corrélation faible (0.42) ; pont info/computation non concluant.
- **ΔK multi-systèmes** : variable selon classe de C → confirme Ξ=template.

---

## 4. Findings solides

1. Pipeline O-info exact (validé analytiquement).
2. Tétrade (D,V,C,μ) comme cadre ; émergence/criticalité/corrélation distinctes (R3).
3. **Criticalité ≠ émergence** : Ising Tc = redondance pure (R1).
4. **Émergence ≡ chaos** via la conjonctivité de la règle (256 fonctions, coin vide ;
   confound réseau r=0.999).
5. Synergie nette pique à l'edge (RBN K=2-3) ; susceptibilité informationnelle 336×.
6. χ_info = Var(Ω) ou CV(Ω) localise la criticalité, transversal (Ising, RBN, Kuramoto,
   percolation) — mais **comparable** à χ, pas supérieur.
7. SOC (sandpile) = synergie pure.
8. Φ = rare résidu d'émergence stabilisé (chaos modéré + fort couplage).
9. LLM : edge dynamique à T≈1.2 ; embedding synergique ; Ω statique = cohérence.
10. **Complexité visible ≠ émergence** (Gray-Scott, espace paramètres) : structure
    géométrique/visuelle ubiquitaire, synergie quasi-absente.

---

## 5. Réfutations (chacune une leçon)

| # | Claim réfuté | Réfutation | Leçon |
|---|--------------|-----------|-------|
| 1 | γ_RG→0 = criticalité universelle | artefact CG circulaire Kuramoto (7×) | tester neutralité du CG |
| 2 | χ_info supérieur à χ | scaling L : comparable | finite-size scaling soigné |
| 3 | CV(Ω) parasite de division | floor inactif → c'est du finite-size | diagnostiquer, pas supposer |
| 4 | synergie persistante générique | Φ rare (émergence fugace) | émergence ≡ instabilité |
| 5 | synergie pique à l'edge (génération LLM) | Ω statique suit l'ordre | statique ≠ dynamique |
| 6 | RBN chaos profond synergique | = bruit (surrogate) | surrogate obligatoire |
| 7 | émergence aux interfaces (Gray-Scott) | interfaces redondantes | mouvement commun = redondance |
| 8 | émergence dans l'espace des paramètres | Ω entre observables = redondance | structure géométrique ≠ synergie |
| 9 | std(Ω) brut universel | échoue Kuramoto (amplitude) | normaliser (CV) |

---

## 6. Méta-findings

- **Généralisations prématurées réfutées par extension** : pattern récurrent (~9 fois). Le
  plus robuste du programme. Toute conclusion sur ≤2 systèmes/1 seed est suspecte ; la vérité
  apparaît au 3e régime.
- **Ξ = template, pas formule fermée** : l'instanciation (Edge, I, P_k) dépend de la classe
  de criticalité dans C (ΔK variable confirme).
- **Garde-fous systématiques** : surrogate shuffle, neutralité CG, sanity analytique, contrôle
  |λ| DMD, baseline i.i.d. — ont démasqué plusieurs faux positifs.

---

## 7. Conclusion — thèse centrale

> **La synergie (émergence info-théorique vraie) est RARE ; la redondance est ubiquitaire.**
>
> La complexité observable — transitions de phase, synchronisation, motifs de Turing, chaos
> spatio-temporel, fractales, bifurcations, activations de LLM — est massivement **redondante**.
> Elle est belle et structurée, mais réductible à ses parties. La nature couple par moyennes
> (diffusion, alignement, attraction) = couplages **monotones** = redondance.
>
> La synergie irréductible exige un couplage **non-séparable** (conjonctif, type XOR), rare en
> nature. Et cette non-séparabilité est **identique** à l'instabilité dynamique : **émergence ≡
> chaos**, deux faces d'une seule propriété. Pas d'émergence sans flirt avec l'instabilité.
>
> Φ — l'émergence qui dure — est donc un résidu rare, stabilisé dans une fenêtre étroite
> (chaos modéré, fort couplage) : le régime du vivant, du calcul, de la cognition.

**Recadrage** : la question n'est plus « où vit l'émergence à l'edge » (elle y bascule, mais
faiblement et rarement) mais « pourquoi la complexité est-elle presque toujours redondante,
et qu'est-ce qui rend les rares systèmes synergiques spéciaux ». L'œil et les observables
classiques voient de la richesse ; l'O-information voit de la redondance.

---

## 8. Questions ouvertes

- **Vrai Φ** : attribution PID spectrale de la synergie par mode Koopman (pas le proxy
  participation-ratio).
- **Solidifications** : B v1 multi-seed + surrogate ; LLM multi-taille/multi-seed ; finite-size
  scaling soigné de χ_info.
- **EEG (cortex-trace)** : test de brain-criticality, bouclage formel de Φ = résidu stabilisé.
- **Systèmes synergiques naturels** : existe-t-il des couplages non-monotones en biologie
  (régulation génétique XOR, circuits neuronaux) ? La synergie y est-elle plus fréquente ?

---

## 9. Inventaire

- **Systèmes** : Ising 2D, Kuramoto (all-to-all + 2D), RBN Kauffman (+ contrôlé), CML,
  Gray-Scott, percolation 2D, sandpile BTW, LLM (Pythia 160m).
- **Classes de criticalité** : conforme (Ising), slowing-down (Kuramoto), edge-of-chaos discret
  (RBN), spatio-temporel (CML), géométrique (percolation), SOC (sandpile).
- **Théorie** : `fundamentals.md`, `equation.md`, `emergence.md`, `interface_emergence.md`,
  `real_llm.md`, `SYNTHESIS.md`. Vulgarisation : `SYNTHESE_VULGARISEE.md`.
- **Expériences** : ~22 scripts dans `experiments/`, données + figures jointes.
