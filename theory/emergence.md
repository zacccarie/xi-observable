# Émergence = irréductibilité révélée par la criticalité

Recadrage du programme (2026-06-01) après l'arc Ising/Kuramoto/RBN. La quête
n'est pas « localiser la criticalité » (résolu, banal) mais **mesurer
l'émergence** et établir d'où elle vient.

## Définitions

- **Émergence** (sens info-théorique strict) = information IRRÉDUCTIBLE au collectif :
  ce que le tout contient et qu'aucune somme des parties ne reconstruit.
  Mesure : O-information négative, Ω < 0 (synergie).
- **Redondance** = information dupliquée dans les parties (Ω > 0). Anti-émergence :
  chaque partie contient déjà l'info, le tout n'ajoute rien.
- **Criticalité** = régime de fluctuations maximales / scale-invariance (selon classe).

## La distinction triadique (révélation R3)

Le framing initial « edge-of-chaos × interaction = Φ » supposait que trois choses
coïncident. Les findings montrent qu'elles **divergent** :

| Concept | Mesure | Comportement à la criticalité |
|---------|--------|-------------------------------|
| Corrélation | TC (≥0) | pique — mais inclut la redondance triviale |
| Criticalité | χ, Var(Ω), γ | fluctuations max, universel, aveugle au régime |
| **Émergence** | **Ω < 0 (synergie)** | **rare, conditionnelle, ≠ criticalité** |

Confondre les trois = l'erreur de départ. Les séparer = l'acquis conceptuel.

## R1 — Criticalité ≠ émergence (SOLIDE)

Ising 2D à Tc : criticalité canonique (scale-invariante, ξ→∞, χ divergente).
Info-théoriquement = **redondance pure**, Ω > 0 partout. Aucune émergence.

⟹ « edge of chaos = lieu de l'émergence » est **réfuté**. La criticalité est
nécessaire (révèle) mais **pas suffisante** (ne crée pas).

## R2 — L'émergence vient de la structure de C (HYPOTHÈSE CENTRALE)

Contraste : Ising couplage alignant (monotone) → redondance ; RBN fonctions
mixtes XOR-like (non-monotones) → synergie. Ce qui décide le RÉGIME (synergie vs
redondance) = le **type de couplage C**, pas la criticalité.

**Hypothèse H (à tester proprement)** :
```
sign(Ω)  ← déterminé par l'IRRÉDUCTIBILITÉ (non-monotonie) de C
|Ω|, Var(Ω) ← amplifiés par la CRITICALITÉ
```
Émergence = **Irr(C) révélée par criticalité**. La criticalité est un
amplificateur/révélateur, pas une source.

**Confound à briser** : dans RBN standard, K règle SIMULTANÉMENT la criticalité
(Derrida) et la richesse du couplage (plus d'inputs → plus de fonctions
non-canalisantes en moyenne). Il faut découpler.

## R3 — Forme corrigée du couplage cherché

Pas 2 facteurs (edge × interaction) mais 3 :
```
émergence = criticalité  ×  intensité d'interaction  ×  irréductibilité de C
```
Le 3e facteur (irréductibilité / non-monotonie) était implicite et manquant.

## Réponse à la question fondatrice

« Comment des règles simples font-elles émerger de la complexité ? »

⟹ **Seules les règles irréductibles émergent.** Une règle décomposable en
contributions de parties (alignement, copie) ne crée jamais d'émergence — même à
son point critique le plus spectaculaire — seulement de la redondance amplifiée.
Une règle dont l'effet n'existe que dans la conjonction (XOR, parité) crée de
l'information irréductible. La criticalité est le régime où cette irréductibilité
devient maximalement observable. **L'émergence est dans la non-séparabilité de la
règle, pas dans la criticalité.**

## Test décisif (`emergence_decouple.py`)

Découpler monotonie de C et criticalité dans RBN à fonctions contrôlées :
- `synergy_frac` ∈ [0,1] : fraction de nœuds à fonction non-monotone (parité)
  vs canalisante (AND/OR)
- mesurer pour chaque réseau : Derrida (criticalité effective) ET Ω (régime)

**Prédictions falsifiables** :
- H vraie : Ω ← synergy_frac (à Derrida contrôlé), pente significative ; régression
  `Ω ~ a·Derrida + b·synergy_frac` dominée par b.
- H fausse (émergence = criticalité) : Ω ← Derrida, dominée par a.
- 3e issue (inséparables) : synergy_frac et Derrida covarient inévitablement
  (fonctions irréductibles intrinsèquement plus chaotiques, average sensitivity
  élevée) → finding profond : **émergence et chaos sont structurellement liés**,
  pas découplables. Ce serait un résultat en soi.

## RÉSULTAT du test décisif (`emergence_decouple.py`, 2026-06-01)

RBN N=60, K=3, fonctions contrôlées (parité=synergie vs canalisante=redondance),
synergy_frac 0→1, 84 réseaux.

| synergy_frac | Derrida | Ω | frac_syn(Ω<0) |
|--------------|---------|-----|----------------|
| 0.0 | 0.58 | +0.0001 | 0.07 |
| 0.2 | 0.81 | −0.0002 | 0.50 |
| **0.4** | **1.01** | **−0.0063** | **1.00** |
| 0.6 | 1.28 | −0.0014 | 1.00 |
| 1.0 | 1.75 | −0.0018 | 1.00 |

**Confound Derrida↔synergy_frac = +0.999.** Régression standardisée dégénère
(a=−3.48, b=+3.36, explosent et s'annulent = multicolinéarité). On NE PEUT PAS
découpler — et c'est LE finding, pas un échec.

### Finding : émergence et chaos sont la MÊME propriété (révision de H)

H supposait deux facteurs séparables (émergence ← C, criticalité = révélateur).
**Faux.** Irréductibilité de C et sensibilité dynamique sont **identifiées** :

> Mécanisme : une fonction synergique exige que TOUTE entrée compte (sinon
> réductible) → chaque entrée est pivot → average sensitivity élevée → Derrida
> élevé → chaos. **Synergie (info, statique) ≡ sensibilité (dynamique).** Deux
> angles d'une seule propriété : la non-séparabilité de la règle.

On ne peut pas avoir d'émergence sans instabilité. C'est mécaniste, pas une
coïncidence de RBN.

### Finding : synergie nette MAXIMALE à l'edge of chaos

Ω le plus négatif concentré à **Derrida ≈ 1.0-1.2** (sf≈0.4). Avant (Derrida<0.8) :
pas assez d'irréductibilité, Ω≈0. Après (Derrida>1.4, chaos profond) : Ω remonte
vers 0 — le chaos DILUE la synergie (cohérent avec le finding RBN « chaos profond
= bruit »). **La synergie nette vit dans une fenêtre étroite pile à la frontière.**

### Reformulation finale de la quête

Pas « émergence = criticalité × interaction × irréductibilité » (3 facteurs).
Mais :

> **L'émergence est UNE propriété — la non-séparabilité de la règle — qui se
> manifeste simultanément comme information irréductible (synergie, Ω<0) et
> instabilité dynamique (chaos, Derrida>1). Elle est NETTE maximale exactement à
> l'edge of chaos (Derrida≈1), avant que le chaos ne la dilue.**

Reconnecte l'intuition historique (Langton : calcul max à l'edge) mais reformulée :
ce n'est pas « le calcul » qui pique à l'edge, c'est **l'information irréductible**.
Et elle pique là parce que l'edge est le seuil où l'irréductibilité est assez forte
pour générer de la synergie nette, mais pas assez pour que le chaos la noie.

### Réponse finale à « règles simples → complexité »

Une règle irréductible est intrinsèquement **à la fois émergente et instable** —
c'est le prix. L'émergence et le chaos ne sont pas séparables. La complexité
(synergie nette max) vit à la frontière, juste avant la dilution chaotique. **Pas
d'émergence sans flirt avec l'instabilité.**

### Durcissement (A — test exhaustif fonctions, `synergy_sensitivity.py`)

Énumération EXHAUSTIVE des 256 fonctions booléennes K=3 : pour chacune, synergie
intrinsèque Ω(inputs,output) et average sensitivity s (→ Derrida).

- **corr(Ω, s) = −0.954**, relation quasi-déterministe (chaque s → Ω quasi-unique)
- **Coin « forte synergie + basse sensibilité » strictement VIDE** : Ω<−0.5 (128 fonctions) → 0 à s<1.5 ; Ω<−1.0 (18) → 0. Aucune fonction synergique non-sensible n'existe.
- Identification confirmée AUX DEUX NIVEAUX : fonction (A, quasi-bijection) et réseau (test décisif, r=0.999). Pas propre à la parité.

### Le paramètre unique : degré de conjonction de la règle

> Synergie et sensibilité sont deux mesures d'UNE chose : **le degré auquel
> l'output dépend de la CONJONCTION des entrées vs de leurs contributions
> individuelles.**
> - Canalisation (un input force la sortie) = info réductible = basse sensibilité = ordre
> - Non-canalisation (tous comptent ensemble, XOR) = info irréductible = haute sensibilité = chaos
>
> Un seul paramètre — la conjonctivité de la règle — contrôle simultanément
> l'émergence (synergie Ω<0) ET l'instabilité (chaos, Derrida>1). Ce n'est pas une
> corrélation, c'est une identité structurelle. **L'émergence EST le chaos, vus
> respectivement côté information et côté dynamique.**

**Caveat résiduel** : K=3 exhaustif ; K≥4 (65536 fonctions) à échantillonner pour
confirmer que le coin reste vide. La relation quasi-déterministe à K=3 est forte
évidence. Distinction maintenue : synergie I/O (fonction) ≠ synergie émergente
inter-nœuds (réseau) — mais les deux suivent le même paramètre (conjonctivité).

## Φ, recadré

Φ = information utile persistante. Dans ce cadre :
**Φ = information synergique persistante = ce qui émerge (Ω<0, irréductible) ET
dure (modes Koopman persistants).** Pas « modes persistants » tout court — modes
persistants qui portent de l'information irréductible. Test futur : Koopman × Ω.
