# SYNTHÈSE — programme Ξ / émergence

Carte maître du programme (2026-06-01). Filtre honnête : ce qui survit, ce qui est
réfuté, ce qui reste ouvert. Détails dans fundamentals.md, equation.md, emergence.md,
real_llm.md.

## La quête (recadrée 2×)

Nominal : « observable Ξ = edge × interaction × hiérarchie pour caractériser Φ ».
Dérive 1 → localiser la criticalité (banal). Recadré → **mesurer l'ÉMERGENCE**
= information irréductible (synergie, Ω<0). Dérive 2 → accumulation de jouets.
Recadré → **brancher l'instrument sur du réel** (LLM).

## Thèse centrale (recadrée — résultat le plus robuste)

> **La synergie (émergence info-théorique vraie) est RARE ; la redondance est
> ubiquitaire.** La complexité observable (transitions de phase, synchronisation,
> motifs de Turing, chaos, fractales, bifurcations, activations LLM) est massivement
> REDONDANTE — belle, structurée, mais réductible à ses parties. La nature couple par
> moyennes (diffusion, alignement) = monotone = redondance. La synergie exige un
> couplage NON-SÉPARABLE (conjonctif, type XOR), rare.

> **Mécanisme** : cette non-séparabilité est IDENTIQUE à l'instabilité dynamique —
> **émergence ≡ chaos**, deux faces d'une seule propriété (la conjonctivité de la règle).
> L'émergence nette est maximale à l'edge of chaos mais fugace (modes transitoires).
> **Φ** = la rare fraction stabilisée (chaos modéré + fort couplage = régime du vivant).

> **Corollaire** : structure visible ≠ émergence. L'œil et les observables classiques
> voient de la richesse ; l'O-information voit de la redondance. Testé sur 3 dimensions
> (espace, temps, paramètres) : structure partout, synergie quasi-nulle.

Répond à « comment des règles simples font-elles émerger de la complexité ? » :
seules les règles irréductibles émergent ; la criticalité les révèle ; l'émergence
durable (Φ) est rare car émergence ≡ instabilité.

## Findings SOLIDES

1. **Pipeline O-info exact** : XOR Ω=−1, copy +1, parité −2 (validés analytiquement).
2. **Tétrade (D,V,C,μ)** : tout système = dimensions, variables, couplages, mesure.
   Toute mesure dérivée est composition des 4.
3. **Émergence = synergie irréductible** (Ω<0), distincte de corrélation (TC) et de
   criticalité (fluctuation). Trois choses qu'on confondait (R3).
4. **Criticalité ≠ émergence** (R1) : Ising 2D à Tc = redondance PURE (Ω>0 partout).
   La criticalité la plus célèbre ne produit aucune émergence.
5. **Émergence ≡ chaos via conjonctivité** (A) : 256 fonctions booléennes K=3,
   coin « synergie + basse sensibilité » VIDE. corr(Ω,sensibilité)=−0.95
   quasi-déterministe. Confirmé niveau fonction ET réseau (r=0.999).
6. **Synergie nette pique à l'edge** : RBN K≈2-3 (edge+bordure) ; CML Derrida≈1.
   Chaos profond = bruit (surrogate). Premier Ω<0 réel : RBN bordure chaotique.
7. **χ_info = Var(Ω)** localise la criticalité, transversal : Ising Tc, RBN K_c,
   Kuramoto K_c, percolation p_c. Comparable à χ (pas supérieur, cf réfut).
8. **SOC (sandpile BTW) = synergie pure** (frac_synergy=1.0) : avalanches =
   événements collectifs irréductibles. 5e classe de criticalité.
9. **Φ = rare résidu stabilisé** (B v1, suggestif) : modes Koopman persistants
   synergiques seulement en chaos modéré + couplage fort. Émergence sinon fugace.
10. **LLM réel** : Ω inter-dim statique = cohérence représentationnelle. Embedding
    synergique (pas le calcul) ; synergie suit l'ordre (pas l'edge génératif).

## Findings RÉFUTÉS (et leçon)

| Claim | Réfutation | Leçon |
|-------|-----------|-------|
| γ_RG→0 universel pour criticalité | Kuramoto = artefact CG circulaire (7×) | tester neutralité du CG |
| χ_info supérieur à χ | scaling L : comparable, pas supérieur | finite-size scaling soigné |
| CV = parasite division | #10 : c'était finite-size (floor inactif) | diagnostiquer, pas supposer |
| synergie persistante générique | Φ rare (B v1) | émergence ≡ instabilité |
| synergie pique à l'edge en génération LLM | suit l'ordre | Ω statique ≠ émergence dynamique |
| RBN chaos profond synergique | = bruit (surrogate) | surrogate obligatoire |

## MÉTA-leçon (la plus robuste)

**5-6 généralisations prématurées réfutées par extension à un 3e régime.** Pattern
récurrent et le plus fiable du programme. Conséquences : (a) Ξ = TEMPLATE, pas formule
fermée — l'instanciation dépend de la classe de C (ΔK variable confirme) ; (b) tout
claim sur ≤2 systèmes/1 seed est suspect ; (c) garde-fous systématiques : test
neutralité CG, surrogate shuffle, sanity analytique, contrôle |λ| DMD, baseline i.i.d.

## Le RÉEL (LLM)

Ω statique inter-dimension mesure la **cohérence représentationnelle**, PAS l'émergence
dynamique des jouets. Inversion : jouets → synergie dans le chaos ; LLM activations →
Ω suit l'ordre. Distinction « deux synergies » (statique I/O vs émergente dynamique).
L'edge of chaos génératif demande une mesure DYNAMIQUE — pas encore faite.

## OUVERT

- **Vrai Φ** : PID spectral (Ω par mode Koopman, pas proxy PR ni groupes). But original.
- **Mesure dynamique LLM** : Lyapunov / dimension effective Koopman de la trajectoire
  de génération vs température. Capture l'edge là où Ω statique le rate.
- **Solidifications** : B v1 multi-seed+surrogate ; χ_info finite-size scaling propre.
- **EEG** (cortex-trace) : test pur de brain-criticality, bouclage formel de B v1.
- **Conjecture pont info/computation** (LZ vs χ_info) : corrélation faible (0.42),
  à creuser ou abandonner.

## Inventaire

- Systèmes : Ising 2D, Kuramoto (all-to-all + 2D local), RBN Kauffman (+ contrôlé),
  CML, percolation 2D, sandpile BTW, LLM (Pythia).
- Classes de criticalité couvertes : conforme (Ising), slowing-down (Kuramoto),
  edge-of-chaos discret (RBN), spatio-temporel (CML), géométrique (percolation), SOC.
- ~25 expériences, 13 chapitres de tâches, théorie en 5 fichiers.
