# Germination — la compréhension naissante est-elle mesurable par principe ?

*Suite directe du KILL de xi_nn_grokking (2026-06-09) : la synergie nette ne
monte qu'après la généralisation. La structure naissante existe (Nanda la voit)
mais notre observable ne l'entend pas. Ce doc affûte la question, défend les
deux thèses, et en sort un observable nouveau (Γ) + protocole pré-enregistré.*

---

## 1. La question, affûtée en trois

« La compréhension naissante est informationnellement silencieuse » mélange
trois questions distinctes :

- **Q1 — silencieuse pour QUEL instrument ?** Notre Ω/s_net est : instantané
  (un checkpoint), pondéré par la mesure (binarisation médiane = statistique
  de la distribution d'activations), et libre de base (subsets aléatoires
  d'unités, aucune connaissance du circuit). Trois propriétés, trois suspects.
- **Q2 — le silence est-il contingent (mauvais instrument) ou de principe
  (aucun instrument de cette classe ne peut entendre) ?**
- **Q3 — s'il est de principe pour une classe, quelle classe d'instruments
  y échappe, et que peut-elle entendre au juste ?**

## 2. Thèse B d'abord (le silence de principe) — steelman

Soit un circuit naissant : une direction cohérente d'amplitude ε dans
l'espace d'activations, noyée sous des composantes de mémorisation
d'amplitude O(1).

**Argument d'amplitude.** Tout observable fonction de la distribution
instantanée μ_t (moments, entropies, Ω sur binarisation) reçoit la direction ε
en O(ε²) — indistinguable du plancher de bruit anisotrope. La binarisation
par médiane est pire : une direction sous-médiane ne bascule aucun bit.
À UN instant donné, sans connaître la base, un germe de faible amplitude est
*par construction* dans la classe d'équivalence du bruit. Le silence est
réel et démontrable pour cette classe (instantané + measure-weighted +
basis-free).

**Argument de bifurcation (le noyau dur de B).** Près d'une transition, la
susceptibilité diverge : le système est maximalement sensible au bruit. Si
plusieurs circuits candidats sont en compétition (course de germes —
cf. lottery tickets, induction heads d'Olsson), *lequel* gagne peut être
décidé par le bruit au voisinage de la bifurcation — imprévisible **en
principe**, comme la branche d'une fourche (pitchfork). Plafond dur : on
peut espérer détecter QUE quelque chose germe et QUAND ça basculera, pas
toujours QUOI.

## 3. Thèse A (le silence contingent) — ce que d'autres oreilles entendent déjà

La littérature mécaniste entend le germe — mais en trichant sur la base :

- **Restricted/excluded loss** (Nanda 2023) : projeter les poids sur les
  composantes de Fourier du futur circuit → leur norme croît PENDANT le
  plateau. Le circuit se forme tôt, en sous-amplitude.
- **Cohérence des gradients** (Chatterjee 2020) : les directions de gradient
  qui *s'accordent entre exemples* sont les directions généralisantes.
  Mesurable pendant le plateau, sans connaître le circuit — mais en espace
  gradient, pas activation.
- **Schaeffer 2023 (« mirage »)** : les « emergent abilities » des LLM
  sautent ou pas selon la *métrique* — métrique discontinue = saut, métrique
  continue = croissance lisse. Le sous-jacent croît continûment ; c'est
  l'instrument à seuil qui fabrique la discontinuité. Notre KILL est la
  version info-théorique de ce résultat : Ω-binarisé est une métrique à
  seuil de dominance.

Donc : la formation EST audible. Mais chaque oreille existante est soit
frame-full (il faut connaître la réponse — restricted loss), soit hors de
l'espace d'activations (gradients). Q2 reste ouverte pour un observable
frame-free SUR les activations.

## 4. La résolution : la pertinence est un exposant, pas un coefficient

L'erreur de catégorie de notre instrument, dite en langage RG :

> Un opérateur *relevant* à coupling nu minuscule est invisible à
> l'amplitude-mètre, mais sa dimension de scaling est positive : il CROÎT
> exponentiellement le long du flot. La pertinence se lit dans l'EXPOSANT
> de croissance, jamais dans le coefficient instantané.

Le germe du circuit de Fourier est exactement cela : un mode relevant du
*flot d'entraînement* (le système dynamique = la trajectoire de training,
le geste fondateur d'architrace), de coupling initial ~0. Le bruit de
mémorisation est l'inverse : coefficients énormes, exposant nul ou négatif
sous le flot (le weight decay le ronge).

**Ce qui distingue un germe du bruit, sans connaître la base, c'est le
TEMPS** :

| | amplitude | direction entre checkpoints | exposant |
|---|---|---|---|
| bruit | grande | décorrèle (cos → 0) | ≤ 0 |
| germe | minuscule | **persiste** (cos ≈ 1) | **> 0** |

D'où l'observable manquant, **Γ (germination)** :

```
Γ(t) = Σ_directions  persistance(d, t) × croissance(d, t)
```

- directions : modes DMD (ou PCA appariée) de la *trajectoire* d'activations
  entre checkpoints — pas d'un checkpoint.
- persistance : alignement (cos) du sous-espace entre t et t+Δ, soutenu.
- croissance : taux log d'amplitude le long de la direction (σ Koopman du
  flot d'apprentissage, pas du flot d'inférence — montée d'un méta-niveau).

Γ est frame-free (aucune base supposée), temporel (échappe à l'argument
d'amplitude §2 : la cohérence inter-checkpoints est O(ε)·O(ε) RELATIVE au
bruit qui décorrèle — le contraste est dans la fidélité, pas la force), et
sur activations. C'est l'exposant-mètre.

**Inversion contre-intuitive** : on cherchait un signal *fort* ; il fallait
chercher un signal *fidèle*. Le naissant n'est pas faible-et-cachée version
du dominant — c'est une autre catégorie : cohérent-sous-dominant. Le bruit
est fort et infidèle ; le germe est faible et têtu.

## 5. Hiérarchie des trois canaux précurseurs

Du plus informé au plus générique — chacun répond à une question différente :

1. **Frame-full** (restricted loss, probe basé circuit) : entend QUOI germe.
   Oracle. Exige de connaître la réponse → inutilisable en découverte.
2. **Frame-free temporel** (Γ, cohérence de gradients) : entend QUE quelque
   chose germe + sa vitesse. Le sweet spot. À construire.
3. **Universel critique** (early-warning signals, Scheffer 2009 : montée de
   variance, AR(1), critical slowing down, flickering) : entend QUE ça va
   basculer BIENTÔT, sans rien savoir de quoi. Le plus faible, le plus
   général. Les transitoires s_net du contrôle (max 0.0052 non soutenus)
   ressemblent à du flickering — à vérifier.

Le plafond de principe (§2, bifurcation) ne bloque que le canal 1 en
découverte : près de la fourche, QUOI est décidé par le bruit. QUE et QUAND
restent mesurables — c'est précisément ce que la criticalité *offre* (la
susceptibilité divergente qui rend QUOI imprévisible est elle-même le
signal de QUAND).

## 6. Enjeu au-delà du grokking

Détection de capacités émergentes dans les LLM avant dominance
(safety-relevant : Wei et al. ; Schaeffer). Si Γ marche sur le grokking
(transition jouet propre), le candidat suivant est un fine-tune réel :
Γ sur les activations pendant le training, cherchant les modes cohérents
croissants AVANT que le benchmark ne bouge. L'early-warning de capacité
serait au canal 2, pas au canal 1 — on saurait qu'une structure monte et à
quelle vitesse, pas laquelle. C'est exactement la bonne granularité pour
déclencher une évaluation ciblée (qui, elle, est frame-full).

## 7. Protocole Γ v0 — pré-enregistré, prêt à tourner

Testbed identique (GrokGPT, wd=1.0 ×3 + wd=0 ×2, probe test fixe 512).

1. Checkpoints tous les 50 steps : activations A_t (512×128), centrées.
2. DMD entre paires (A_t, A_{t+50}) → modes ; OU variante pauvre robuste :
   PCA de A_t, appariement des sous-espaces consécutifs par cos principal.
3. Pour chaque direction appariée : persistance p = cos soutenu sur 3
   fenêtres ; croissance g = pente log-amplitude. Γ(t) = Σ max(0,g)·p sur
   les directions HORS top-3 dominantes (exclure la dominance acquise).
4. **H-Γ : Γ(t) monte de façon soutenue ≥ 300 steps avant t_grok dans ≥ 2/3
   seeds wd=1.0, et reste plat (ou non soutenu) dans les contrôles wd=0.**
5. Baselines à battre : weight norm totale (détecte le grokking trivialement
   — Γ doit faire MIEUX en avance ou en spécificité), TC plateau (0.05 vs
   0.034, l'exploratoire d'hier, à pré-enregistrer ici), early-warning AR(1)
   sur val_loss.
6. Kill : Γ plat jusqu'au grok, OU Γ monte autant dans le contrôle →
   l'exposant-mètre n'entend pas mieux que l'amplitude-mètre → la version
   forte de B gagne sur les activations, et le germe ne vit que dans les
   poids/gradients.

## 7bis. RÉSOLUTION EMPIRIQUE (2026-06-10) — la thèse A gagne, par le temps

Le protocole §7 a tourné (3 actes, voir FINDINGS_XI_V2.md) :

- v0 (croissance absolue) : ARTEFACT — le mémorisant gonfle aussi.
  L'amplitude absolue ne discrimine pas, conformément à §2.
- v1 (part de variance) : structure à 2 vagues — burst initial universel
  (formation de la mémorisation : elle aussi est une structure qui germe),
  puis vague 2 pré-grok spécifique aux runs qui généraliseront.
- v2 (fenêtre pré-enregistrée, seeds frais) : **KEEP** — Γ_window
  wd1 = [0.013, 0.027, 0.094] vs contrôles = [0.000, 0.000] exact,
  séparation par seed totale.

**Verdict sur Q2 : le silence était contingent.** Un observable frame-free,
temporel, sur activations seules, entend le germe ~500-650 steps avant la
dominance. La classe condamnée par §2 (instantané + measure-weighted +
basis-free) l'est bien — mais il suffit de deux photos pour en sortir.
La pertinence est un exposant : mesurée comme telle, elle parle.

Le plafond résiduel de la thèse B (§2, bifurcation : QUOI germe près de la
fourche) reste non testé — c'est la question de l'individualité des modèles
(germes gagnants coïncident-ils entre seeds ?).

~~Caveat ouvert : wd ↔ destin confondus par design~~ — **LEVÉ (v3,
2026-06-10)** : à wd=1.0 fixe, frac ∈ {0.25..0.70}, Γ_window sépare
binairement les destins 8/8 (min grokkers 0.0219 > max no-grok 0.0142) avec
dose-réponse monotone en fraction de données. Γ signe le germe, pas le
réglage. Réfuté au passage : Γ prédicteur de DATE (Spearman +0.4 — destin
oui, date non).

**L3 causal : KILL (même jour)** — ablation en ligne des directions
Γ-positives (chasse adaptative k ≤ 8, contrôles matchés aléatoire/non-croissant) :
retard +100/+150 steps seulement, sous la barre pré-enregistrée (+300).
**Le germe vit dans les poids ; les directions d'activations sont son ombre
— on coupe l'ombre, le réseau ré-exprime le circuit dans le complément
orthogonal en ~50 steps.** Raffinement de §4 : la « fidélité » que Γ mesure
est la trace en activations d'un objet poids-spatial. Γ = sismographe, pas
barrage : il prédit (L1+L2) sans tenir le levier (L3-activations). Pour
l'usage early-warning (canal 2, §6), c'est suffisant et c'est l'essentiel.
Reste : L3-poids (intervention en espace poids), L4 généralité,
QUOI près de la bifurcation.

## 8. Coda — être et devenir

Ξ mesure l'émergence *accomplie* (la dominance) ; Γ mesurerait l'émergence
*en train de se faire* (la fidélité croissante). Le silence n'était pas une
propriété du monde — c'était une propriété de l'instantané. Une distribution
à un instant est de l'être ; un germe est du devenir ; aucune statistique de
l'être ne contient le devenir, mais deux instants suffisent à l'entendre.
La question du programme (« comment des règles simples font-elles émerger la
complexité ? ») se mesure donc en deux temps irréductibles : Γ pour le
surgissement, Ξ pour ce qui en survit.
