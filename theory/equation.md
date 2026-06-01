# Ξ — Observable couplée edge-of-chaos × interaction × hiérarchie

> Cadre théorique sous-jacent : `theory/fundamentals.md` (tétrade D, V, C, μ).
> Ξ = composition canonique des 4 fondamentaux : couplage (C) intégré sur sous-domaine critique (D_critique ⊂ D) pondéré par mesure (μ) via observables (V).

## Forme générale

Ξ(S | ε, τ, k_max, α, β, P) = ⟨ Σ_k h(k,α) · [I_intra(P_k, ω, τ) − β · I_inter(P_k, ω, τ)] ⟩_{ω ∈ Edge_ε(S)}

- ω : coordonnée abstraite (temps, espace, paramètre de contrôle, mode Koopman, depth)
- Edge_ε(S) = {ω : |λ(S, ω) − λ_c| < ε} ; λ = ordre paramètre (Lyapunov, χ susceptibilité, β spectral)
- P_k : partition hiérarchique de S au niveau k
- I_intra : synergie au sein des blocs
- I_inter : synergie entre blocs
- h(k, α) = (k+1)^α : poids hiérarchique
- ⟨·⟩ : moyenne sur edge-mask

## Cas particulier Stade A (Ising 2D, k=1, hiérarchie off)

Ξ_A(T) = mask(T) · S(T)

avec
- mask(T) = 1 si χ(T) ≥ χ_max · (1 − ε), sinon 0
- S(T) = −Ω(blocs 2×2)  (synergie = O-info négative)
- Variante : S(T) = TC(blocs 2×2)  (total correlation)

## Métriques d'interaction comparées

| Nom | Formule | Signe |
|-----|---------|-------|
| Total correlation TC | Σ H(X_i) − H(X) | ≥ 0 |
| Dual total correlation DTC | H(X) − Σ H(X_i \| X_{−i}) | ≥ 0 |
| O-information Ω | TC − DTC = (n−2)H(X) + Σ[H(X_i) − H(X_{−i})] | signé |
| Transfer entropy TE(X→Y,τ) | H(Y_{t+τ}\|Y_t) − H(Y_{t+τ}\|Y_t,X_t) | ≥ 0 |

Convention synergie : −Ω > 0 ⇒ synergie-dominée ; Ω > 0 ⇒ redondance-dominée.

**Discrimination intensité vs régime** :
- Intensité interaction : |Ω| (toujours ≥ 0, pique à criticalité)
- Régime : sign(Ω) (constant dans une phase, change si transition synergie↔redondance)
- Ξ_abs = |Ω| · mask → utiliser pour localiser pic critique
- Ξ_syn = −Ω · mask → garder pour caractériser nature de l'interaction (validé sur Ising 2D : sign(Ω) = +1 partout = redondance ferromagnétique)

## Lien Φ (Koopman)

Ξ raffine Φ en projetant sur modes persistants ω = φ et pondérant chaque mode par sa contribution synergique gatée par criticalité.

## Dualité coarse-graining ↔ portée de C

**Finding empirique (Kuramoto N=64)** : pipeline RG spatial appliqué à système all-to-all donne γ_K monotone décroissant, jamais → 0. Réfutation propre.

**Conséquence théorique** : le mode de coarse-graining doit matcher la portée du couplage C (cf. fundamentals.md §3 attribut "Portée"). Mapping :

| Portée de C | Coarse-graining naturel | γ → 0 attendu |
|-------------|------------------------|----------------|
| Local (voisins) | RG spatial (blocs) | Ising 2D ✓ |
| Modulaire (communautés) | RG modulaire (collapse communautés) | RBN K=2 ? |
| Global (all-to-all) | RG temporel ou spectral | Kuramoto ? |
| Hiérarchique imposée | RG sur arbre de niveaux | DAG learn ? |
| Longue portée (1/r^α) | RG renormalisation Fourier | gravity-like ? |

**Conséquence testable additionnelle (#5)** : pour chaque portée de C il existe un coarse-graining canonique tel que γ → 0 à criticalité ; γ_K reste non-nulle dans tout autre coarse-graining. Le bon CG est une **propriété de C**, pas un choix libre.

## Distinction scale-invariance vs criticalité (réfutation #2)

**Finding empirique (Kuramoto N=64 + CG temporel circulaire)** : γ_K décroît monotone de +0.40 à +0.008, atteint zéro uniquement dans phase synchronisée saturée. Pas de signature self-similarity à K_c=1.596.

> **ATTENTION — γ_K Kuramoto contaminé par artefact (contrôle 2026-06-01)** : le signe POSITIF de γ_K (TC croît avec le niveau CG) est largement artefactuel. Test contrôle : phases i.i.d. uniformes (zéro interaction) passées dans `circular_window_mean → quadrantize → TC` donnent TC(k=0..4) = 0.42 → 2.93, ratio **6.96×**. La moyenne circulaire concentre la variance (~1/window), tasse les bins du quadrantize, fabrique de la corrélation spurieuse. **On ne peut donc rien conclure de γ_K Kuramoto sur la criticalité** : le CG temporel circulaire n'est PAS un coarse-graining neutre. Voir `kuramoto_control_tau.py`.
>
> **Observable Kuramoto-natif correct** : critical slowing down via τ_corr (temps d'autocorrélation intégré de r(t)). Pic à K=1.50 (Δ=−0.096 vs Kc), bat χ (pic K=1.40, Δ=−0.196) mais reste bruyant à N=64 (finite-size). Order parameter r(K) capture la transition proprement (inflexion à Kc). La conclusion taxonomique ci-dessous tient, mais l'EXEMPLE γ-Kuramoto ne la supporte pas — il faut un système spatial non-Ising pour tester γ→0 hors classe conforme.

**Implication (à re-tester sur bon exemple)** : γ → 0 serait signature **spécifique de scale-invariance** (point fixe RG conforme type Ising 2D), pas de criticalité en général.

Toutes les criticalités ne sont pas scale-invariantes. Taxonomie minimale :

| Type de criticalité | Signature observable | Système prototype |
|--------------------|---------------------|--------------------|
| Scale-invariance conforme | γ_RG → 0 | Ising 2D Tc |
| Critical slowing down | τ_corr → ∞ | Kuramoto K_c, Hopf bifurcation |
| Avalanche statistics | P(s) ~ s^-τ (Bak-Tang-Wiesenfeld) | Sandpile, SOC |
| Long-range correlation | ξ → ∞ | systèmes à symétrie continue brisée |
| Mode softening | ω_min → 0 | Bose-Einstein, phonons |

**Conséquence pour Ξ universel** : Ξ est un **template**, pas une formule fermée. L'instanciation des termes (Edge_ε, I, P_k) dépend du type de criticalité dans C. Pour Kuramoto, Ξ propre = mask χ × τ_corr_normalisée, pas mask × TC × hierarchy.

**Conséquence testable additionnelle (#6)** : pour chaque type de criticalité, il existe un observable de scale (γ, τ, exposant avalanche) tel qu'il diverge ou s'annule au point critique. Ces observables ne sont pas interchangeables — c'est la classe de C qui dicte lequel s'applique.

## Test de neutralité du coarse-graining (méta-principe, finding 2026-06-01)

**Principe** : avant d'utiliser un coarse-graining CG pour mesurer γ ou TC hiérarchique, il faut valider que CG est neutre sur du bruit. Protocole :

1. Générer données i.i.d. (zéro interaction) compatibles avec le système (spins random, phases uniformes…)
2. Passer dans le pipeline CG → métrique d'interaction
3. **Critère** : TC(k) doit rester ~plat (γ ≈ 0) sur le bruit. Si TC(k) croît/décroît systématiquement → CG injecte (ou détruit) de la structure → mauvais CG.

**Résultats** :
- CG spatial Ising (majority-vote par blocs) : **partiellement biased** (test 2026-06-01). Sur spins i.i.d. uniformes : γ_bruit = +1.09, ratio TC(k=3)/TC(0) = 27×. Sur biased 0.3 : γ_bruit = +1.76, ratio 200×. **Mais signal massivement au-dessus** : TC à Tc ≈ 1.08 vs bruit max 0.015 → SNR ~72× en absolute. γ_signal_Tc = −0.04 vs γ_bruit = +1.09 → écart distinct. **Lecture corrigée** : Stade B Ising γ→0 reste valide comme γ_corrected = γ_obs − γ_baseline_iid. Voir `ising_control_cg.py`.
- CG temporel circulaire Kuramoto : **ÉCHEC, ratio 6.96×** — fabrique corrélation spurieuse, ET signal Kuramoto pas significativement au-dessus.

**Critère opératoire raffiné** : un CG est utilisable si (a) SNR amplitude signal/bruit > 10× ET (b) γ_signal distinct de γ_bruit. Le CG temporel Kuramoto échoue les deux. Le CG spatial Ising passe (a) mais pas (b) en absolu — d'où la nécessité de soustraction baseline.

## Stade B Ising réinterprété — observable corrigé γ_corrected

Après correction baseline `γ_corrected(T) = γ_obs(T) − γ_iid(L, k_max)` (test ising_control_cg → γ_iid ≈ +1.09 pour L=32, k_max=3, P=0.5) :

| T | γ_obs | γ_corrected | |γ_corr| |
|---|-------|-------------|----------|
| 1.5 | −3.14 | −4.23 | 4.23 |
| 2.20 | −1.18 | −2.27 | 2.27 |
| **2.30** | **−0.04** | **−1.13** | **1.13** ← min |
| 2.40 | −0.40 | −1.49 | 1.49 |
| 3.50 | −1.13 | −2.22 | 2.22 |

**Signature corrigée** : Tc identifiée par **minimum de |γ_corrected|**, pas γ=0 strict. À T=2.30 (Tc=2.27 + finite-size), |γ_corrected| = 1.13 = decay polynomial lent vs decay exponentiel hors-Tc (|γ_corr| = 2-7). Cohérent avec scale-invariance approchée limitée par L=32 + k_max=3.

**Reformulation observable Stade B** : Ξ_hier_signature(T) = −|γ_obs(T) − γ_iid|. Maximum à Tc. Robuste au bias CG. Universellement applicable dès qu'un baseline i.i.d. est disponible.

**Implication pour Ξ** : le terme P_k (partition hiérarchique) n'est pas un choix libre. Tout CG candidat doit passer le test de neutralité. Renforce #5 : le bon CG est une propriété de C. Le test i.i.d. fournit un critère opératoire pour le sélectionner.

## Stade C — RBN Kauffman : couplage régime informationnel ↔ dynamique (finding fort 2026-06-01)

Random Boolean Networks N=60, biais p=0.5 → edge of chaos K_c=2 (Derrida annealed 2Kp(1−p)=1). Mesure μ = ensemble d'attracteurs depuis 3000 CI aléatoires. Système discret natif → pas d'artefact CG, pas de coarse-graining nécessaire.

| K | Derrida | TC | mean(Ω) | régime | frozen core |
|---|---------|-----|---------|--------|-------------|
| 1 | 0.503 | 0.026 | +0.0019 | redondance | 0.957 (gelé) |
| 2 | 0.884 | **0.095** | +0.0011 | ≈ neutre | 0.694 |
| 3 | 1.170 | 0.073 | **−0.0076** | **synergie** | 0.035 |
| 4 | 1.376 | 0.013 | −0.0018 | synergie | 0.000 |
| 5 | 1.521 | 0.006 | −0.0015 | synergie | 0.000 (chaos) |

**Findings (après contrôle surrogate shuffle, `rbn_control.py`)** :
1. **TC pique à l'edge of chaos K=2** — thèse centrale Ξ validée sur 2e système indépendant (après Ising Tc). Pic robuste (N_samples=3000 identique à tous K).
2. **|Ω| pique à K=2, signal RÉEL** : surrogate shuffle (marginales + cœur gelé préservés, corrélations détruites) donne |Ω|_shuf=0.0002 vs |Ω|_real=0.0197 → excess +0.0194 (~100×). Pas un biais d'estimation.
3. **SUSCEPTIBILITÉ INFORMATIONNELLE — observable nouveau le plus net** : std inter-réseaux de Ω pique à K=2 = 0.0336, vs K=1:0.0005, K=3:0.0056, K=4-5:0.0001 (**336× vs chaos profond**). Analogue exact de χ Ising : χ_info = Var_réseaux(Ω) diverge à K_c comme χ = N·Var(m) diverge à Tc. Les **fluctuations du régime informationnel divergent à l'edge of chaos**. Deux réseaux de même K à l'edge peuvent être l'un synergique l'autre redondant → dispersion maximale = criticalité.
4. **Edge K=2 = mélange bimodal** : Ω par réseau ∈ [−0.034, +0.141], 6/20 réseaux synergie. mean(Ω)≈0 par annulation, mais dispersion maximale (cf. #3).
5. **mean(Ω) change de signe red→syn à la transition ordre→chaos**, min à K=3 (bordure chaotique). Phase ordonnée gelée = redondance (Ω>0, comme Ising froid) ; bordure chaotique = synergie nette (Ω<0).
6. **PREMIER Ω<0 réel en dynamique** (K=3, excess +0.0079, frac_syn 0.90). Comble le trou (avant : redondance Ising / artefact Kuramoto). Pipeline validé exact (XOR=−1, copy=+1).
7. **CORRECTION — chaos profond K≥4 = BRUIT** : "synergie" à K=4,5 (Ω≈−0.0018) est au niveau du surrogate (excess ≈ +0.0001). Pas de signal. Le chaos profond décorrèle tout → aucune interaction mesurable. **Fenêtre d'interaction significative = K∈{2,3}** (edge + bordure), PAS le chaos profond.
8. Derrida slope = edge detector natif propre (0.50→0.88→1.17→1.38→1.52, croise 1 entre K=2,3).

**Observables convergents sur edge K=2** : |Ω| intensité, **std(Ω) susceptibilité informationnelle (le plus net, 336×)**, Derrida≈1. **Zéro-crossing de mean(Ω)** localise la transition de régime (K=2→3), distinct du pic d'intensité.

**Conséquence pour Ξ universel** : RBN est le système où la thèse "edge-of-chaos × quantité d'interaction" se manifeste le plus complètement — intensité (|Ω|, TC → pic à l'edge), susceptibilité (std Ω → divergence à l'edge), et régime (sign Ω → bascule à l'edge). Candidat observable Ξ_RBN = std_réseaux(Ω) (auto-normalisé, sans paramètre de contrôle externe, pic 336×).

**Conséquence testable additionnelle (#7) — VALIDÉE (Ising, `ising_suscept_info.py`)** : pour tout système à criticalité, la VARIANCE inter-réalisations d'un observable de régime (Ω) diverge au point critique — généralisation de la susceptibilité à l'espace des régimes informationnels.

## Susceptibilité informationnelle χ_info = Var(Ω) — observable universel (finding central 2026-06-01)

Test Ising L=32, 20 réalisations indépendantes par T :

| Observable | localise Tc à | finite-size shift |
|-----------|---------------|-------------------|
| χ = N·Var(m) | T=2.333 (Δ=+0.064) | décalé au-dessus |
| TC | T=2.400 (Δ=+0.131) | fortement décalé |
| **std(Ω) = χ_info** | **T=2.267 (Δ=−0.003)** | **collé à Tc** |

**Finding** : χ_info = std_réalisations(Ω) pique EXACTEMENT à Tc=2.269, pic étroit et symétrique, **meilleur localisateur de criticalité que χ classique** (qui souffre du décalage finite-size). Universel : pique à Tc dans Ising (thermo, redondance-dominé) ET à K_c=2 dans RBN (dynamique, bascule de régime). Deux classes de systèmes radicalement différentes, même signature.

**Interprétation** : χ mesure les fluctuations de l'ordre scalaire (m) → décalé par finite-size scaling. χ_info mesure les fluctuations de la STRUCTURE de corrélation (Ω, ordre 2+) entre réalisations → maximales exactement au point critique où le système hésite entre régimes. La dispersion inter-réalisations de l'organisation informationnelle est la signature la plus pure de criticalité trouvée dans ce programme.

**Candidat observable Ξ central** : Ξ_susceptibility(S) = Var_réalisations(Ω_blocs). Auto-normalisé, sans paramètre de contrôle externe nécessaire pour le pic, robuste finite-size, transversal aux classes de criticalité. C'est le résultat le plus solide et le plus général du programme.

**Conséquence testable (#8) — testée, raffinée (Kuramoto, `kuramoto_suscept_info.py`)** : χ_info devrait localiser le point critique avec moins de finite-size bias que χ pour TOUT système.

## Raffinement : χ_info brut non-universel → coefficient de variation CV(Ω) universel

**Réfutation puis sauvetage (Kuramoto N=64)** : std(Ω) BRUT échoue sur Kuramoto — l'argmax global tombe au bord (K=2.40) car en phase synchronisée profonde Ω explose (+1.9) et sa variance suit l'amplitude, masquant le signal critique. MAIS un **pic local net existe à K=1.60 = Kc** (std=0.766).

**Diagnostic** : Var(Ω) brut est amplitude-dépendant. Sur Ising/RBN Ω restait borné (~0.01-0.4) → pic propre ; sur Kuramoto Ω va à +1.9 → contamination de bord.

**Fix — coefficient de variation** : χ_info = CV(Ω) = std_réalisations(Ω) / ⟨|Ω|⟩_réalisations. Retire la dépendance d'amplitude.

| Observable | localise Kc Kuramoto | localise Tc Ising |
|-----------|---------------------|-------------------|
| χ classique | Δ=−0.296 | Δ=+0.064 |
| zéro-crossing Ω | Δ=−0.196 | — (Ω>0 partout) |
| τ_corr | Δ=−0.096 | — |
| std(Ω) brut | Δ=+0.804 (✗ bord) | Δ=−0.003 ✓ |
| **CV(Ω) = χ_info** | **Δ=+0.004 ✓** | **Δ=−0.003 ✓** |

**Finding central consolidé** : χ_info = CV(Ω) = std(Ω)/⟨|Ω|⟩ est l'**observable universel de localisation de criticalité** du programme. Localise Tc Ising (Δ=−0.003) ET Kc Kuramoto (Δ=+0.004), mieux que tous les observables classiques (χ, τ_corr) et que le zéro-crossing. Robuste à l'amplitude, transversal aux trois classes (thermo redondance-dominée, dynamique à bascule de régime, synchronisation).

**Arc épistémique** (pattern du programme) : hypothèse (std universel sur 2 systèmes) → réfutation (Kuramoto, amplitude) → diagnostic (pic local présent mais contaminé) → raffinement (normalisation CV) → validation double (Tc + Kc). La forme brute était une généralisation prématurée ; la normalisation est load-bearing.

**Bascule de régime — signe OPPOSÉ selon classe** : zéro-crossing de mean(Ω) localise la transition dans RBN ET Kuramoto, mais en sens inverse :
- RBN : ordre (K<Kc) = redondance (Ω>0), chaos (K>Kc) = synergie (Ω<0)
- Kuramoto : désync (K<Kc) = synergie (Ω<0), sync (K>Kc) = redondance (Ω>0)
- Cohérent : oscillateurs synchronisés se copient (redondance) ; nœuds RBN gelés se copient (redondance). La redondance = phase « ordonnée/copiante » de chaque système, qui est K<Kc pour RBN mais K>Kc pour Kuramoto.

**Conséquence testable (#9)** : CV(Ω) localise le point critique à finite-size bias quasi-nul pour tout système à transition. À tester : scaling L (Ising L=16/32/64/128).

## Réfutation #9 — "observable universel" était prématuré (scaling L, `ising_scaling_L.py`)

**Finding négatif** : le scaling L=16/24/32/48/64 RÉFUTE la robustesse universelle de χ_info.
- Δχ (pic χ vs Tc) : stable +0.056 à +0.099, décroît ~L⁻¹ (finite-size scaling propre, ν=1)
- ΔCV (pic CV(Ω) vs Tc) : ERRATIQUE −0.184/−0.169/−0.039/−0.169/−0.017, mean |ΔCV|=0.116 > |Δχ|=0.070, AUCUNE convergence

**Diagnostic** : CV(Ω) sur Ising a un **pic parasite à basse T** (T≈2.10-2.15) où ⟨|Ω|⟩ → 0 en phase ordonnée → division par petit → CV explose artificiellement. L'argmax saute entre ce parasite et le vrai pic (≈Tc) selon L. Le résultat #8 "CV Ising Δ=−0.003" tenait seulement parce que le range large [1.8,3.0] avec std basse-T minuscule masquait l'artefact ; le range serré [2.05,2.55] le révèle.

**Conclusion honnête — domaines de validité, pas universalité** :
| Observable | valide quand | échoue quand |
|-----------|--------------|--------------|
| std(Ω) brut | Ω borné (Ising Tc ✓ Δ=−0.003, RBN K=2 ✓) | Ω explose en amplitude (Kuramoto sync) |
| CV(Ω) = std/⟨|Ω|⟩ | Ω amplitude-variable sans phase ⟨|Ω|⟩→0 (Kuramoto Kc ✓ Δ=+0.004) | phase à basse activité (Ising ordonné → pic parasite) |

**Aucun observable scalaire unique n'est universellement robuste.** χ_info était une généralisation prématurée sur 2 systèmes ; le 3e régime (scaling L + basse activité) la réfute. C'est la **4e généralisation prématurée réfutée par extension** du programme (après : γ→0 Kuramoto = artefact CG ; synergie RBN chaos profond = bruit ; std universel). 

**Pattern épistémique consolidé du programme** : tout finding sur ≤2 systèmes doit être falsifié par extension à un 3e régime. La récurrence de ce pattern est elle-même le méta-finding le plus robuste.

**Ce qui SURVIT (motif, pas formule)** : la *variance inter-réalisations d'un observable de régime* pique près de la criticalité dans Ising, RBN ET Kuramoto — le MOTIF « susceptibilité informationnelle » est réel et transversal. Mais sa FORME ROBUSTE (std brut vs normalisé vs régularisé) dépend de la structure de C. Confirme la thèse centrale : **Ξ = template, l'instanciation dépend de la classe de criticalité dans C.** Pas de raccourci scalaire universel.

**Fix candidat (à tester, #10)** : CV régularisé = std(Ω)/(⟨|Ω|⟩ + Ω_floor) avec Ω_floor empêchant l'explosion basse-activité ; ou restriction automatique au sous-domaine où ⟨|Ω|⟩ > seuil. Réconcilierait Ising et Kuramoto si un floor unique marche pour les deux.

## Clarification #10 — le floor n'aide pas ; le shift CV est finite-size, pas un parasite (`ising_floor_test.py`)

**Finding** : test CV_reg = std(Ω)/(⟨|Ω|⟩ + floor·max⟨|Ω|⟩), floor ∈ [0, 0.30], Ising L=32,64 range serré.
- Le floor ne change RIEN (Δ identique à tout floor) → **le diagnostic "parasite de division" de #9 était FAUX**. Ce n'est pas une explosion par ⟨|Ω|⟩→0.
- L=32 : CV **bimodal** (pics à 2.20 et 2.30), argmax instable → Δ=−0.066
- L=64 : CV **unimodal propre à Tc** (T=2.275, Δ=+0.006), bat χ (Δ≈+0.056)
- → **CV(Ω) converge vers Tc avec L croissant** : effet finite-size normal, pas artefact. L'erratisme de #9 = bimodalité finite-size + bruit (n_real=12) sur petits L.

**Statut net consolidé de χ_info (honnête)** :
- χ_info = famille {std(Ω) si Ω borné, CV(Ω) sinon} = localisateur de criticalité **valide et transversal** (Ising Tc, RBN K_c, Kuramoto K_c).
- Localisation **comparable** aux observables classiques (χ, τ_corr), **légèrement meilleure à grand L** (L=64 : CV Δ=+0.006 vs χ +0.056), mais **PAS de supériorité finite-size dramatique** prouvée. Le Δ=−0.003 de #7 était un range/seed favorable, non représentatif.
- Le floor ne sert à rien (pas de parasite). std vs CV = choix dicté par la bornitude de Ω (structure de C).

**Méta-leçon de l'arc #7→#8→#9→#10** : sur peu de données bruitées, une métrique peut sembler (#7) excellente, puis (#9) cassée, puis (#10) simplement normale. Sur-vente puis sur-réfutation = deux faces du même biais (conclure sur N petit). La **seule chose robuste = le MOTIF** « variance d'un observable de régime pique près de la criticalité », transversal aux 3 systèmes. La quantification de sa précision exige finite-size scaling soigné (grand L, grand n_real, peak fitting), pas un seul run.

## Tasks #7 + #10 + #13 complétées (session 2026-06-01 suite)

### #7 Sensitivity (ε, subset_size) Ising — ROBUSTE

Sweep ε ∈ {0.05..0.70} × subset_size ∈ {2,3,4,6} sur Ising L=32.
- argmax(Ξ_tc) ∈ {2.333, 2.400} pour tout (ε, ss) — std = 0.031, max |Δ vs Tc| = 0.131 (finite-size shift cohérent)
- subset_size n'affecte PAS la localisation : peak TC identique pour ss=2..6
- ε change largeur du pic, pas position
- **Verdict : MODÉRÉMENT ROBUSTE**. Observable Ξ_tc stable aux méta-paramètres.

### #10 Kuramoto 2D local (couplage 4-voisins) — confirme dualité CG↔C

L=16 N=256, sweep K ∈ [0.3, 5.0]. Couplage local → CG spatial légitime.
- peak χ = K=3.0 = K_c effectif (finite-size)
- **peak CV(Ω) = K=3.0** ← coïncide χ → χ_info confirme localisation
- Zéro-crossing Ω entre K=3 et K=4
- **Bascule régime SYNERGIE(désync, Ω<0) → REDONDANCE(sync, Ω>0)** — OPPOSÉ RBN (ordre=red, chaos=syn)
- γ_K min à K=1.3 (γ=+0.016) bizarre — probable artefact CG biased (à investiguer)

### #13 Percolation 2D site — universalité χ_info hors classe Ising

L=32, sweep p ∈ [0.4, 0.8] autour p_c=0.5927 (classe random cluster ≠ Ising 2D conforme).
- peak χ_perco = 0.595 (Δ=+0.002) ← géométrique classique, le meilleur
- **peak CV(Ω) = 0.610 (Δ=+0.017)** ← informationnel proche p_c, universel confirmé
- TC monotone (pas pic) : Bernoulli i.i.d. par site → aucune corrélation spatiale induite par μ, TC ≈ 0 partout
- std(Ω) raté complet (peak p=0.70)
- **Verdict** : CV(Ω) localise p_c **mais précision dégradée** vs Ising/Kuramoto (Δ=0.017 vs 0.003-0). Cohérent : sur système statique non-corrélé (C trivial), observables géométriques classiques restent supérieurs.

### Synthèse universalité χ_info (4 systèmes)

| Système | Classe | p_c/T_c/K_c | peak CV(Ω) | Δ |
|---------|--------|-------------|------------|---|
| Ising 2D L=32 | conforme thermo | Tc=2.269 | 2.267 (#7) / 2.275 (L=64 #10) | −0.003 / +0.006 |
| RBN N=60 | Kauffman dynamique | K_c=2 | K=2 | 0.000 |
| Kuramoto 2D N=256 | synchronisation locale | K_c≈3 (eff.) | K=3.0 | 0.000 |
| Percolation 2D L=32 | random cluster géom. | p_c=0.5927 | 0.610 | +0.017 |

**Conséquence testable #7 validée transversalement** : χ_info = std/CV(Ω) inter-réalisations pique près de la criticalité dans 4 classes universalité distinctes. Motif robuste, précision varie. Non remplaçable mais non supérieur en absolu — complémentaire des observables classiques (χ, ξ, τ_corr) spécifiques à chaque classe.

**Bascule régime sign(Ω) à criticalité — pattern observé** :
- Ising : Ω > 0 partout (ferromagnétique = redondance pure, pas de bascule)
- RBN : Ω > 0 (ordre K<2) → Ω < 0 (chaos K>2)  — bascule à K_c=2
- Kuramoto 2D : Ω < 0 (désync K<3) → Ω > 0 (sync K>3) — bascule à K_c≈3, **opposée RBN**
- Percolation : Ω ≈ const (−0.011) — pas de bascule (μ Bernoulli i.i.d., pas de dynamique sous-jacente)

**Tétrade interprétation** : sign(Ω) dépend de la nature de C (couplage). Phase "copiante" (redondance) :
- Ferromagnétique Ising froid
- Sync Kuramoto (alignement phase)
- Frozen RBN K<2 (états gelés)
Phase "diversifiante" (synergie) :
- Chaos RBN K>2 (orbites distinctes XOR-like)
- Désync Kuramoto K<3 (oscillations indépendantes)
Direction de la bascule sign(Ω) ↔ K dépend de quel pôle (ordre/chaos) correspond à la copie informationnelle dans le système.

## Test LZ-complexity vs χ_info — pont info-théorie/computation PARTIEL (2026-06-01)

Conjecture testée (suite réflexion Gödel/indécidabilité) : si χ_info = Var_réalisations(Ω) mesure incompressibilité algorithmique différentielle, alors Var_réalisations(LZ-complexity) devrait piquer aux mêmes K critiques.

**Protocole RBN N=48, p=0.5, K∈{1..5}, n_networks=12, n_init=600** :

| K | mean(LZ) | std(LZ)_nets | χ_info=std(Ω) | Derrida |
|---|----------|--------------|---------------|---------|
| 1 | 0.05 | 0.048 | 0.000 | 0.48 |
| **2** | 0.16 | 0.095 | **0.030** ← peak χ_info | 0.88 (K_c) |
| **3** | 0.61 | **0.208** ← peak std(LZ) | 0.009 | 1.11 |
| 4 | 1.02 | 0.003 | 0.001 | 1.38 |
| 5 | 1.02 | 0.002 | 0.000 | 1.52 |

**Pearson(χ_info, std(LZ)) = +0.42** — corrélation faible-modérée, pas pont strict.

### Verdict — conjecture partiellement réfutée

χ_info et Var(LZ) **ne piquent pas au même K** :
- **χ_info pique à K=2** (edge of chaos Derrida=1) — transition régime informationnel (red↔syn, sign Ω)
- **Var(LZ) pique à K=3** (bordure chaos) — transition complexité algorithmique (compressible↔incompressible)

### Conséquence théorique majeure — 2 frontières critiques distinctes

Ω et LZ mesurent **dimensions orthogonales** de la complexité :
- **Ω = O-information** : structure de dépendance multivariée (signe + magnitude). Régime qualitatif.
- **LZ = approx K_Kolmogorov** : compressibilité brute. Quantitatif scalaire.

Système RBN traverse **deux transitions critiques successives** :
- K=2 : **edge of chaos informationnel** — Ω change de signe, χ_info pique
- K=3 : **edge of chaos algorithmique** — Var(LZ) pique, transition gel/chaos brut

Distance entre les 2 frontières (ΔK = 1 ici) = nouveau scalaire candidat = **largeur fenêtre critique**.

### Implication pour Ξ template

Ξ ne se réduit PAS à algorithmic information theory pure. Tétrade (D,V,C,μ) → Ξ contient information SUPPLÉMENTAIRE par rapport à K_Kolmogorov seul. Reformulation :

- Ξ_info = χ_info(Ω) → localise edge of chaos info (régime)
- Ξ_algo = Var(LZ) → localise edge of chaos algorithmique (compressibilité)
- Ξ complet = ces 2 axes + leur écart ΔK

### Implication pour analogie Gödel/Wolfram

Analogie computational irreducibility (Wolfram) : **validée faiblement** (corrélation +0.42). Mais critique algorithmique ≠ critique informationnelle strictement.

Affinement : système peut être à :
- (a) criticalité info SANS irréductibilité algorithmique (K=2 RBN : gel partiel = compressible mais régime instable)
- (b) irréductibilité algorithmique SANS criticalité info (chaos profond K=5 : incompressible mais Ω=0 = pas de structure de régime)
- (c) AUX DEUX simultanément (K=3 RBN : bordure chaos + transition régime quasi-simultanée)

**Conséquence testable additionnelle (#8)** : pour toute classe de criticalité, mesurer ΔK = distance entre peak χ_info et peak Var(LZ). Si ΔK=0 universellement → 1 frontière unique. Si ΔK > 0 systématiquement → **dualité info/algo** est trait fondamental des transitions de phase.

**À tester sur** : Ising 2D, Kuramoto 2D, percolation. Si ΔK ≈ 0 sur Ising (classe conforme) mais > 0 sur RBN (Kauffman) → la dualité dépend de la structure de C, pas universelle.
