# Ξ — paper draft outline

**Working title** : "Emergence is not criticality : information-theoretic dissection across five universality classes"

**Cible** : NeurIPS Workshop on Information-Theoretic Principles in Deep Learning, OU Entropy journal (open access, info-theory focused), OU PRX Life.

---

## Abstract (200 mots)

We test the long-standing intuition that "edge-of-chaos = locus of emergence" on five toy systems spanning distinct universality classes (Ising 2D, Kuramoto all-to-all and 2D local, Random Boolean Networks, Site Percolation, Bak-Tang-Wiesenfeld Sandpile). Using O-information Ω as régime indicator (signe distingue synergie/redondance), Kolmogorov-LZ as algorithmic complexity proxy, and the variance of Ω across independent realizations χ_info as informational susceptibility, we show three findings :

1. Criticality is **necessary but not sufficient** for emergence : Ising at T_c shows divergent χ_info and ξ → ∞ but Ω > 0 (pure redundancy).
2. Emergence (Ω < 0) requires **irreducible coupling structure** : RBN with non-monotone Boolean functions and BTW sandpile (conditional toppling rule) exhibit pure synergy ; monotone alignment couplings never do, regardless of criticality.
3. χ_info = std(Ω) across realizations is a universal **locator of critical points** across 4 of 5 systems tested (Δ ≤ 0.02 vs analytical critical value), comparable but not superior to classical observables.

We reformulate Ξ as a composition template (D, V, C, μ) — tetrad of dimensions, variables, couplings, measure — that must be instantiated per universality class. The result clarifies a 30-year conceptual conflation in complexity science and offers an operational test for emergence in learned systems (LLM activations, BCI signals).

---

## 1. Introduction

- Question fondatrice : "How do simple rules produce complexity ?" (Kauffman, Wolfram, Bak)
- Intuition dominante : edge of chaos = lieu privilégié de l'émergence (Langton 1990, Packard 1988)
- Confusions classiques :
  - Corrélation totale = émergence ? Non.
  - Criticalité = émergence ? Non (notre finding)
  - Computational irreducibility = émergence ? Partiellement.
- Contribution : décomposition info-théorique rigoureuse + 5 systèmes + reformulation cadre tétrade

## 2. Theoretical framework

### 2.1 Tetrad (D, V, C, μ)
- Tout système scientifique = quadruplet
- D : espace dimensions
- V : algèbre observables
- C : structure couplages (attributs : ordre, portée, densité, symétries, irréductibilité)
- μ : mesure probabilité

### 2.2 Ξ as composition
```
Ξ(S | ε, τ, k, α, β, P) = ⟨ Σ_k h(k,α) [I_intra(P_k) − β I_inter(P_k)] ⟩_{ω ∈ Edge_ε(S)}
```
- Edge_ε(S) : bande critique (mask)
- P_k : partition hiérarchique RG
- I_intra/I_inter : O-information par bloc

### 2.3 Triade conceptuelle séparée
| Concept | Mesure | Comportement à criticalité |
|---------|--------|---------------------------|
| Corrélation | TC ≥ 0 | pique, mais inclut redondance triviale |
| Criticalité | χ, std(Ω), γ_RG | fluctuations max, aveugle au régime |
| Émergence | sign(Ω) = Ω < 0 | rare, conditionnelle, ≠ criticalité |

### 2.4 Three observables and their roles
- O-information Ω (régime)
- Total correlation TC (intensité interaction)
- LZ complexity (compressibilité algorithmique)
- χ_info = std_réal(Ω) (susceptibilité informationnelle)

## 3. Methods

### 3.1 Systems
- Ising 2D Metropolis L=32, 64
- Kuramoto all-to-all N=64 + 2D local L=16
- RBN Kauffman N=60 p=0.5 K=1..5
- Site percolation L=20-32
- Sandpile BTW L=32

### 3.2 Coarse-graining
- RG spatial blocks (Ising) ; majority vote ±1 ; mode catégoriel
- Test neutralité i.i.d. obligatoire avant interprétation
- Soustraction baseline γ_corrected = γ_obs − γ_iid

### 3.3 Surrogate controls
- Shuffle inter-réalisations (préserve marginales, détruit corrélations)
- |Ω|_real vs |Ω|_shuffle = excess réel

### 3.4 Pipeline validation
- XOR (Ω = -1 attendu), copy (Ω = +1), parity (Ω = -2) — pipeline exact validé

## 4. Results

### 4.1 Ising 2D : criticality without emergence
- Ξ_tc pique à T_c=2.27 (Δ=+0.006 à L=64)
- γ_T → 0 = self-similarity RG conforme (signature scale-invariance)
- **Mais Ω > 0 partout** : pure redondance ferromagnétique → R1

### 4.2 RBN K=2 : χ_info pique à edge of chaos
- Derrida slope croise 1 entre K=2 et K=3
- TC pique K=2 (intensité max)
- **|Ω| pique K=2 mais mean(Ω)≈0** : edge = mélange synergie/redondance bimodal
- **χ_info pique K=2** (336× vs chaos profond, validé surrogate)
- mean(Ω) change signe à transition ordre→chaos
- → R2 + Conséquence #7

### 4.3 Sandpile BTW : synergie auto-organisée
- ⟨z⟩=2.0, τ=1.44 : SOC atteint
- **Ω = -0.50 SYNERGIE PURE** sur 100% réalisations
- Sans paramètre contrôle externe — auto-maintenu
- Source : règle toppling **conditionnelle** (z≥4) = non-monotone

### 4.4 Universality of χ_info across 4 systems
| Système | critique | peak χ_info | Δ |
|---------|----------|-------------|---|
| Ising L=64 | 2.269 | 2.275 | +0.006 |
| RBN | 2.0 | 2.0 | 0 |
| Kuramoto 2D | 3.0 | 3.0 | 0 |
| Percolation | 0.593 | 0.610 | +0.017 |

### 4.5 ΔK = peak(Var LZ) − peak(χ_info) variable
- Ising +0.20, Kuramoto 2D 0, percolation +0.03, RBN +1.0
- **Hypothèse nouvelle** : ΔK = proxy d'irréductibilité de C

### 4.6 Falsifications propres
- R#1 RG spatial inadapté à Kuramoto all-to-all (γ monotone)
- R#2 CG temporel circulaire fabrique corrélation (ratio 7× sur bruit i.i.d.)

## 5. Discussion

### 5.1 Méta-principe — test de neutralité du CG
- Tout coarse-graining doit donner γ ≈ 0 sur i.i.d. avant usage
- Critère SNR > 10× et γ_signal distinct γ_bruit

### 5.2 Réponse à la question fondatrice
> Seules les règles irréductibles émergent. Décomposables (alignement, copie) → redondance amplifiée. Non-monotones (XOR, seuil) → information irréductible. La criticalité est amplificateur, pas source.

### 5.3 Connexion Gödel/Wolfram
- Computational irreducibility (Wolfram) : pertinent (faible corrélation +0.42)
- Gödel strict : pas applicable (manque auto-référence)
- Hardness phase transition (k-SAT à α_c) : analogie opérationnelle

### 5.4 Limites
- Finite-size scaling sur 2 L seulement (Ising) — convergence asymptotique non prouvée
- Pas testé sur systèmes apprenants (LLM, RC)
- O-info computationnelle limitée à subset_size ≤ 8

### 5.5 Suite et programme ouvert
- emergence_decouple R2 multi-systèmes
- LLM activations across depth
- Φ Koopman persistant × irréductible (cadre formel étendu)

## 6. Conclusion

Trois théorèmes empiriques :
1. Criticalité ⇏ émergence (Ising contre-exemple décisif)
2. Émergence ⇐ irréductibilité de C (RBN + sandpile démonstratifs)
3. χ_info = std_réal(Ω) localise criticalité transversalement à 4 classes universalité

Reformulation cadre : Ξ = template (D, V, C, μ) instancié par classe, pas formule fermée universelle. Le motif "variance d'observable régime pique à criticalité" survit, sa forme dépend de la structure de couplage.

---

## 7. Reproducibility

- Repo public : github.com/zacccarie/xi-observable
- License MIT
- 8 commits git, theory/equation.md + theory/emergence.md + theory/fundamentals.md
- Tous experiments reproductibles depuis `experiments/*.py` (≤ 60s par sweep)
- Python 3.11, numpy + matplotlib + scipy uniquement

---

## Figures principales

1. Schéma tétrade (D, V, C, μ) + composition Ξ
2. Ising Stade A : χ, TC, Ξ_tc(T) + falsification mask global
3. Ising Stade B : γ_T(T) + min |γ_corrected| à Tc
4. RBN sweep : Derrida + TC + sign(Ω) + χ_info(K)
5. RBN surrogate control : |Ω|_real vs |Ω|_shuffle
6. Sandpile : P(s) ~ s^-τ + distribution Ω (synergie pure)
7. Universalité χ_info — 4 systèmes côte à côte
8. ΔK multi-systèmes (info vs algo)

## Tables principales

- Tableau récap 5 systèmes (classe, ordre param, χ_info Δ, sign(Ω))
- Tableau 8 conséquences testables + statut empirique
- Tableau pipeline validation (XOR/copy/parity → Ω exact)

## Citations clés à inclure

- Kauffman 1969 (RBN)
- Langton 1990 (edge of chaos, computation)
- Bak Tang Wiesenfeld 1987 (SOC)
- Wilson 1971 (RG)
- Wolfram 2002 (computational irreducibility)
- Tononi (IIT, Φ)
- Bertschinger Natschläger 2004 (edge of chaos NN)
- Williams Beer 2010 (PID)
- Rosas et al. 2019 (O-information)
- Mézard Parisi Zecchina 2002 (k-SAT hardness phase transition)

## Notes de méthode pour rédaction

- Style : factuel, dense, falsifiable
- Pas de hype "consciousness / intelligence" sauf section future work
- Inclure réfutations propres comme contribution (méta-principe CG)
- Code blocks pour observables (formules)
- Reproductibilité : 1 figure = 1 script + 1 JSON
