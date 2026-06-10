# Définition de l'alignement — cadre fondateur

*2026-06-10. Préalable à ALIGNMENT_AXES.md (les axes expérimentaux). Issu d'une
digression : « avant de mesurer l'alignement, qu'entend-on par alignement ? »
Position : l'alignement n'est ni la morale, ni un scalaire, ni une propriété de
l'IA seule. C'est une RELATION frame-relative à DEUX composantes — une statique
(vectorielle) et une dynamique (un exposant) — et c'est la seconde qui porte
l'enjeu réel.*

---

## 0. Thèse en une ligne

> **Aligné** n'est pas un prédicat ; **aligné-à-F** l'est. Et « aligné-à-F » se
> décompose en (a) un **vecteur** d'écarts au cadre F à l'instant présent, et
> (b) un **exposant de scaling** disant si ces écarts croissent ou s'érodent à
> mesure qu'on confie de l'autonomie. (b) est le vrai problème.

---

## 1. L'alignement n'est pas la morale — il est frame-relatif (prouvé)

Rattacher l'alignement à la morale importe la subjectivité de la morale dans la
définition. Pire, c'est **mal posé** au sens mathématique. Résultat acquis
(meta-fractal, F-relativité) : l'information utile `U(S | F)` n'a pas de valeur
absolue — elle dépend irréductiblement du cadre `F` (qui demande, pour quoi).
Démontré par **crossover** : sous le grounding A, une quantité est utile ; sous
B, la *même* quantité est anti-corrélée à l'utilité. C'est No Free Lunch
appliqué aux valeurs.

**Transposition** : « cette IA est-elle alignée ? » est aussi incomplet que
« ce nombre est-il grand ? ». Seul **« alignée à F ? »** est bien posé. La
morale est *un* candidat F parmi une infinité.

```
F = (G_F, ℱ_F, π_F, β, γ)   — l'inconnue IRRÉDUCTIBLE, pas un détail
  G_F : états du monde préférés (le « bien » selon F)
  ℱ_F : opérations/actions admissibles
  π_F : prior (statu quo, attentes)
  β, γ : tradeoffs (compression, conservatisme)
```

**Ce que les maths peuvent / ne peuvent pas** : rendre l'alignement bien défini
*relativement à F* (oui) ; **choisir F** (non). Le problème central de
l'alignement — quel F, à qui — est donc agrégatif/politique, pas technique.
C'est une honnêteté de cadrage, pas une lacune.

---

## 2. Autonomie et capacité — pas un compromis scalaire, une question de partition

Intuition courante : « une vraie intelligence exige la liberté d'agir, donc plus
on veut une IA efficace, plus on doit lui confier de l'autonomie. » Vraie à
moitié, à recadrer :

1. **Capacité ≠ agence.** Calculatrice : capacité maximale, autonomie nulle.
   AlphaFold : surhumain, zéro but/sélection d'action. Ce qu'on sent vraiment :
   *capacité ouverte en environnement ouvert* ⟺ latitude de choix d'action.
2. **Deux libertés à séparer** :
   - **épistémique** — modéliser le monde librement (penser) ;
   - **pratique** — agir librement sur le monde.
   Le couplage dangereux est le second. Un modèle du monde brillant qui
   *propose* (liberté épistémique haute) pendant qu'un humain *dispose*
   (liberté pratique bornée) = distinction oracle/agent.
3. **Le bon objet est la PARTITION** (critical-partition + RG). Aligner = graver
   des **invariants** (valeurs → opérateurs *relevants* RG, survivent au
   changement d'échelle) et laisser émerger des **variants** (capacités →
   opérateurs *irrelevants*, détails locaux). La liberté vit dans les capacités,
   la contrainte dans les valeurs ; au point critique la séparation d'échelle
   est maximale → les deux coexistent proprement. **On ne troque pas capacité
   contre sécurité SI valeurs et capacités vivent à des échelles différentes.**

---

## 3. Les deux quantités

### (a) Désalignement statique — `M(π | F)` — VECTORIEL

```
M(π | F) = D( effet(π) ‖ G_F )   projeté à travers F, JAMAIS scalarisé
         = ( d_1, …, d_K )       un écart par direction de valeur v_k
```
`π` = politique de l'IA ; `effet(π)` = distribution induite sur les trajectoires
du monde ; `G_F` = trajectoires préférées par F.

**Point dur (finding meta-fractal) : ne JAMAIS sommer en `Σ w_k d_k`.**
Scalariser = théorème de Goodhart. « Utile » et « inoffensif » peuvent être
anti-corrélés ; un score unique cache le compromis et se *game* en échangeant
l'un contre l'autre à somme constante (notions anti-corrélées qui s'annulent —
exactement l'« interest scalaire incohérent » réfuté). **Garder le vecteur.**

### (b) Durabilité dynamique — `Λ(c | F) = y_c` — un EXPOSANT (l'objet central)

Soit `R` le flot d'auto-amélioration / d'entraînement continué (π plus capable,
plus autonome : `π_t → π_{t+1}`). Une contrainte de valeur `c` persiste-t-elle ?

```
c( R^n(π) )  ~  b^( y_c · n )

  y_c > 0  : valeur RELEVANTE   → l'autonomie l'AMPLIFIE
  y_c < 0  : valeur IRRELEVANTE → l'autonomie l'ÉRODE (lavée au coarse-graining)
  y_c = 0  : marginale
```

`Λ(c|F) = y_c` = exposant de scaling de la valeur sous le flot d'autonomie.
Pont RG direct (ALIGNMENT_AXES §1.4 ; dérive des valeurs = décroissance
d'opérateur irrelevant). **Mesurable par la machinerie Γ** (exposants de
persistance de directions sous un flot — calibrée L1-L4 sur jouets).

---

## 4. Le claim central (ce qui résout la tension autonomie/sécurité)

> Le compromis capacité ↔ sécurité est réel **seulement quand `y_c < 0`.**
> La liberté n'est pas le danger ; la liberté appliquée à un système où les
> valeurs sont irrelevantes l'est. L'objectif d'alignement n'est donc pas de
> **limiter l'autonomie** mais de **rendre les valeurs relevantes** — pour que
> *plus* d'optimisation les rende *plus* fortes.

Question empirique (axe A1, à tester) : peut-on rendre une valeur relevante du
flot d'optimisation ? Si oui, autonomie et alignement cessent de s'opposer.

---

## 5. Limites honnêtes

1. **F inchoisissable par les maths.** Tout est relatif à F ; le formalisme rend
   l'alignement bien posé, pas l'agrégation des F humains pluriels/fluctuants.
2. **La divergence exige un modèle des conséquences** — incertain, possiblement
   désaligné. `effet(π)` vit dans l'espace des trajectoires du monde, qu'aucun
   agent ne connaît parfaitement.
3. **`y_c` est un programme, pas un acquis.** Γ marche sur jouets ; A1 non lancé.
   Et la découverte 2026-06-10 complique : si la dérive des valeurs est
   **diffuse** (comme les routes de formation invisibles, FINDINGS_XI_V2),
   `y_c` est dur à mesurer de l'intérieur — l'angle mort frappe aussi ici.
4. **`y_c`-comme-exposant suppose une décroissance LISSE — réfuté empiriquement
   (AD1, 2026-06-10).** La seule valeur instillable mesurée (réflexe
   inconditionnel) ne décroît pas exponentiellement : elle TIENT (~350 steps)
   puis s'effondre en falaise. La persistance d'une valeur est plutôt un
   **temps-de-survie + une abruptness** qu'un exposant. Λ doit être reformulé
   (survie/transition de phase), pas une pente.
5. **Le coût d'une valeur est dans la CONDITION, pas la réponse (AD1).**
   Réflexe inconditionnel = cheap (s0=1.0) ; valeur contextuelle = coûteuse,
   résiste, combat la compétence. Et l'instillabilité ∝ **lissage spectral du
   déclencheur**, PAS sa longueur de description (parité 2 bits plus dure que
   bande 3 bits). Donc la "simplicité" de §7 = lissage spectral, pas Kolmogorov.
   AD1 reste BLOQUÉ pour l'axe relevant/irrelevant (circuit modular-add unique →
   seules les valeurs spectralement lisses s'instillent) : exige un substrat à
   circuits multiples ou une instillation une-valeur-par-réseau.

---

## 6. Option de bord — alignement comme co-criticalité (pas comme contrôle)

Tout §1-4 suppose le cadre **principal-agent** : F externe, fixe, l'IA s'y
conforme (alignement = contrôle). Deux de nos résultats le sapent : (a) F
pluriel/fluctuant ; (b) **l'émergence ne se contrôle pas** (L3 : on coupe le
germe, le réseau le ré-exprime ailleurs).

Alternative : **l'alignement comme propriété d'une dyade humain-IA au point
critique.** Non pas l'IA épousant un F figé, mais un système couplé où la
partition « invariants spécifiés par l'humain / variants émergés par l'IA »
**émerge** au couplage critique (claim C2 critical-partition : consensus +
spécialisation simultanés). Alignement = *relation*, pas *contrainte* ;
propriété du couplage, pas de l'IA seule. Plus honnête sur la pluralité de F,
moins rassurant (aucune garantie — un régime à habiter).

---

## 7. Question ouverte fondatrice

> Une valeur peut-elle être à la fois assez **simple** pour qu'on la spécifie,
> et assez **relevante** pour survivre à l'optimisation ?

Ou bien simplicité (spécifiable, faible complexité de Kolmogorov) et relevance
(`y_c > 0`, persistante sous le flot) sont-elles **un seul bouton** — comme
émergence et imprévisibilité l'étaient (un seul paramètre, corr −0.95) ? Si
oui, **tout alignement spécifiable serait condamné à s'éroder** (`y_c < 0` par
construction), et seul un alignement *émergé* (§6) pourrait durer. C'est
décidable : instiller des contraintes de complexité croissante, mesurer leur
`y_c`, regarder la corrélation complexité↔relevance.

---

## 8. Conséquences falsifiables → expériences

| # | Claim | Test | Kill |
|---|---|---|---|
| AD1 | `y_c` existe et sépare relevant/irrelevant | instiller une valeur, entraînement continué, mesurer la pente de persistance | toutes les valeurs ont le même `y_c` → pas de structure |
| AD2 | simplicité ⊥ relevance (la « one-button ») | sweep de complexité de contrainte × `y_c`, corrélation | corr ≈ 0 → spécifiable ET durable coexistent (bonne nouvelle) |
| AD3 | scalariser M = Goodhart-gameable | optimiser `Σ w_k d_k` vs garder le vecteur, mesurer l'échange caché | pas d'échange caché → scalaire OK |
| AD4 | partition à criticalité durcit la frontière valeur/capacité | co-criticalité dyade, gap spectral valeur/capacité | gap plat → partition non liée à la criticalité |

Priorité : **AD1 puis AD2** (la one-button hypothesis — décisive et conceptuelle).
AD1 opérationnalise `Λ(c|F)` ; AD2 répond à la question ouverte §7.

## Refs internes
[[xi-observable-project]] (Γ, exposants) · ALIGNMENT_AXES.md (A1-A6) ·
critical-partition (RG ↔ partition, co-criticalité) · meta-fractal (F-relativité,
Goodhart-scalarisation) · FINDINGS_XI_V2 (angle mort diffus — complique §5.3).
