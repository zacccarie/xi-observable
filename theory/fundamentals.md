# Tétrade fondamentale — cadre théorique pour Ξ

## Postulat

Tout système S décrit scientifiquement se ramène à un quadruplet :

```
S = (D, V, C, μ)
```

où :

- **D** = espace des dimensions (degrés de liberté)
- **V** = algèbre des variables (observables)
- **C** = structure des couplages (interactions, contraintes, symétries)
- **μ** = mesure (probabilité, pondération statistique)

Tout observable scientifique est une fonctionnelle F[D, V, C, μ] covariante sous reparamétrage.

Conjecture forte : aucune cinquième catégorie fondamentale n'est nécessaire.

---

## 1. Dimensions D — l'espace de jeu

**Définition**. D est un espace mathématique abstrait (variété, graphe, ensemble discret, espace mesurable) sur lequel les variables se déploient.

**Sous-structure (souvent oubliée)** :

| Attribut | Rôle | Exemples |
|----------|------|----------|
| Cardinalité \|D\| | nombre de degrés de liberté | dim ℝ^n, n |
| Type | continu / discret / fractal / hyperbolique | ℝ, ℤ, Cantor, ℍ² |
| Topologie | connectivité, trous, compacité | T², ℝ vs S¹ |
| Métrique g | notion de distance, voisinage | euclidienne, riemannienne, ultrametrique |
| Mesurabilité | σ-algèbre des sous-ensembles évaluables | Borel, Lebesgue |
| Orientabilité, signature | flèche du temps, signature lorentzienne | (−,+,+,+) Minkowski |
| Symétries internes de D | invariances de l'espace lui-même | rotations, translations |

**Note critique** : le temps n'est PAS une dimension comme les autres dans tout système dynamique — il porte causalité (lightcone) et arrow. Le compter comme « +1 dimension » sans structure supplémentaire perd l'essentiel.

**Méta-variable extraite** : `D = (|D|, type, g, τ_causal)`.

---

## 2. Variables V — quantités déployées

**Définition**. V est une algèbre d'observables : fonctions ψ : D → 𝕂 (avec 𝕂 = ℝ, ℂ, état quantique, catégorique, etc.) closes sous opérations.

**Sous-structure** :

| Attribut | Rôle |
|----------|------|
| Type d'image (scalaire, vecteur, tenseur, spineur, opérateur) | nature physique |
| Range (borné, non borné, compact) | contraintes intrinsèques |
| Algèbre (commutative / Lie / Poisson / C*) | classique vs quantique |
| Régularité (C^0, C^∞, distributionnelle) | différentiabilité |

**Méta-variable** : `V = (N_var, type_alg, dim_alg)`.

---

## 3. Couplages C — relations entre variables

**Définition**. C est la structure relationnelle qui rend le système non-trivial. Sans C, V est un produit cartésien d'observables indépendantes.

**Sous-structure** :

| Attribut | Rôle |
|----------|------|
| Ordre k | 2-corps, 3-corps, k-corps |
| Portée | local / non-local / longue portée |
| Densité ρ_C = #liens / N² | sparsité du graphe interaction |
| Signe / type | attractif, répulsif, mixte |
| Linéarité | linéaire / multiplicatif / via champ intermédiaire |
| Symétries G | groupe d'invariance (→ Noether → conservation) |
| Hiérarchie | présence ou non de niveaux emboîtés |

**Forme générique** :
```
C = { (V_i₁, ..., V_iₖ, ω_α) → 𝕂 } pour ω_α ∈ D
```

où ω_α paramètrent les sites du couplage.

**Méta-variable** : `C = (k_max, ρ_C, ⟨|c|⟩, G, hier_depth)`.

---

## 4. Mesure μ — pondération statistique

**Définition**. μ est une mesure de probabilité (ou densité, ou poids) sur l'espace d'états Ω(D, V). Sans μ, pas de moyenne, pas d'entropie, pas d'observable statistique.

**Sous-structure** :

| Attribut | Rôle |
|----------|------|
| Type | a priori (ex. Lebesgue), équilibre (Boltzmann), dynamique (mesure invariante) |
| Support | total / fractal / mesure nulle (caustiques) |
| Symétrie sous G | μ G-invariante ou non |
| Mémoire | markovienne, longue mémoire |

**Méta-variable** : `μ = (support, regularité, H[μ])`.

**Distinction critique** : variable = observable, mesure = poids. Souvent confondues. En physique stat, Boltzmann sépare proprement H(état) ∈ V de p(état) = e^{−βH}/Z ∈ μ.

---

## Mesures dérivées (toutes secondaires)

Aucune des mesures suivantes n'est fondamentale — toutes dérivent des 4 ci-dessus :

| Mesure | Composition |
|--------|-------------|
| Dimension effective D_eff | D + C (corrélations entre dims nominales) |
| Entropie H | μ seule |
| Information mutuelle I | C + μ |
| Lyapunov λ | D + C + dynamique (sous-ensemble de C) |
| Susceptibilité χ | C + μ + dérivée par paramètre |
| Profondeur hiérarchique | C récursif |
| O-information Ω | C + μ (multi-corps) |
| Complexité de Kolmogorov K | description de S entier, méta-mesure |

---

## Ξ comme composition des 4 fondamentaux

```
Ξ(S) = ∫_{D_critique} I[V, C, μ] dμ
     = couplage (C) intégré sur sous-ensemble critique de dimensions (D_critique ⊂ D)
       pondéré par mesure (μ) avec variables (V) comme entrées
```

Précisément :
- `D_critique = Edge_ε(S)` = sous-ensemble de D où ordre paramètre est dans bande critique (sélection de sous-domaine de D)
- `I[V, C, μ]` = mesure d'interaction (O-info, TC, TE) = fonctionnelle de V (entropies) et C (structure multi-corps) et μ (statistique)
- `∫ … dμ` = moyenne statistique sur edge

Ξ est donc **composition canonique des 4 fondamentaux**. C'est ce qui motive sa portée universelle prétendue.

---

## Méta-système — récursion

Niveau k de description :

```
S_k = (D_k, V_k, C_k, μ_k)
S_{k+1} = description de S_k → ses propres (D, V, C, μ) au niveau k+1
```

Exemple :
- S_0 = Ising 2D (D = ℤ²×ℝ_temps, V = {s_ij}, C = ∑ s_i s_j voisins, μ = Gibbs)
- S_1 = description statistique (D_1 = espace paramètres {T}, V_1 = {χ, m, ξ}, C_1 = relations Widom, μ_1 = mesure sur sweep T)
- S_2 = description Ξ (D_2 = {T, ε, k_max, α}, V_2 = {Ξ, regime}, C_2 = sensibilité Ξ vs méta-paramètres)

Question ouverte : **convergence** ? Existe-t-il S_∞ point fixe (auto-description) ? Lien Wheeler « it from bit », Tegmark MUH, théorie ultime.

**Pour Ξ pratique** : 2 niveaux suffisent. Étendre à 3 = sensitivity sweep méta-paramètres (task #7).

---

## Lien avec autres frameworks Zac

| Framework | Correspondance tétrade |
|-----------|------------------------|
| Koopman-Φ | V_Koopman = base canonique imposée par dynamique (C) sur D |
| Meta-fractal U(S\|F) | F = choix de méta-frame (D ⊂ D_total + sélection de C visible) |
| Critical-partition RG | partition récursive = hiérarchie dans C |
| Architrace β-slope | β = exposant de C × μ (spectre de fluctuations) |
| Dynamoscope volatility | mesure de C via dispersion temporelle de V |

---

## Conséquences testables

1. **Covariance** : Ξ doit transformer de manière prévisible sous changement de coordonnées de D. Test : Ising 2D vs Ising sur graphe aléatoire — même Tc effectif → même Ξ_peak relatif.
2. **Décomposabilité** : si C est séparable (C = C_A ⊕ C_B sur sous-systèmes A, B indépendants), alors Ξ(S) = Ξ(A) + Ξ(B). Test : 2 grilles Ising disjointes vs 1 grille double.
3. **Invariance par symétrie** : si G symétrie de C, Ξ doit être G-invariant. Test : Ising rotation π/2.
4. **Saturation hiérarchique** : Ξ(k) doit atteindre plateau à k = k_max intrinsèque du système. Test : RBN à K=2 vs K=3 (K=2 critique → plateau plus tard).

---

## Limites du cadre

- Ne distingue pas système physique vs simulation numérique de S (problème de réalité)
- μ équilibre vs μ hors équilibre demande structure supplémentaire (dynamique explicite dans C ?)
- Variables auto-référentielles (V contient une description de S elle-même) → problèmes type Gödel

Ces limites définissent l'horizon du programme.
