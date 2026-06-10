# Alignement — ce que le corpus dit, les axes falsifiables

*2026-06-10. Synthèse transversale (xi-observable / Γ / emergetics / metamorph /
architrace / critical-partition / meta-fractal) → sous-problèmes d'alignement.
Règle : chaque pont s'appuie sur un finding daté, pas une analogie.*

---

## 1. Les quatre ponts exacts

### 1.1 Détection précoce ↔ Γ (le plateau EST le scénario qui inquiète)
Sandbagging / deceptive alignment / capacité non détectée = *comportement plat,
intérieur en mouvement* — la structure exacte du plateau de grokking. Γ lit la
formation quand le comportement ne dit rien (L1-L4 : 8/8 ; L5 : conditions =
fond calme + sonde adéquate). Insight RG transposable à la gouvernance :
**les évals comportementales mesurent des coefficients ; la sécurité exige des
exposants** (Schaeffer : les jumps sont des artefacts de métriques à seuil ;
le substrat croît continûment).

### 1.2 Certification ↔ L3 + bifurcation
L3 : le germe vit dans les poids ; activations = ombres ; comportement = ombre
d'ombre. Bifurcation : près de la transition, QUEL circuit gagne peut être
décidé par le bruit → **deux modèles comportementalement identiques peuvent
porter des mécanismes différents** (divergence OOD possible). Plafond de
principe de l'éval-based safety — testable (A4).

### 1.3 Auto-amélioration ↔ VG / méta-récursion / théorème-c
- **Discipline > sophistication** (VG) : la stabilité d'un système
  auto-modifiant tient à la PORTE D'ACCEPTATION (verify-then-keep, rollback),
  pas à la qualité des mutations. Primitive de gouvernance empiriquement fondée.
- **La récursion converge sous compression** (meta_relational L1-L2 saturé ;
  d(n) monotone) : le Droste ne s'est pas matérialisé — MAIS nos systèmes
  n'avaient aucune pression à contourner leurs bornes. Bonne question
  formulée proprement : *qui fixe β (pression de compression) dans un agent
  réel, et β est-il manipulable par l'agent ?*

### 1.4 Dérive des valeurs ↔ partition RG (la contribution la plus nette)
Aligner = graver des invariants (valeurs), laisser émerger des variants
(capacités) — la méta-partition de critical-partition. RG : seuls les
opérateurs RELEVANTS survivent au changement d'échelle.
> **Dérive des valeurs sous scaling = décroissance d'opérateur irrelevant.
> Une contrainte d'alignement n'est durable que si elle est un opérateur
> relevant du flot d'entraînement — et sa relevance est un exposant,
> mesurable avec l'exposant-mètre (Γ).**

## 2. Trois insights transversaux

1. **Émergence ≡ imprévisibilité** (corr −0.95, coin vide 256 fonctions) :
   génératif-et-totalement-prévisible est une tension de principe, pas un
   problème d'ingénierie. À quantifier, pas à nier.
2. **Le danger vit dans le training, pas l'inférence** : activations
   d'inférence = redondance massive (synergie rare) ; la réorganisation
   violente arrive pendant l'apprentissage (L5a : érosion wd). Le moment
   critique du monitoring = chaque phase de training/fine-tune/RL.
3. **Goodhart = théorème de scalarisation chez nous** : interest scalaire
   réfuté (notions anti-corrélées s'annulent), fix = vecteur frame-projeté ;
   F-relativité (crossover anti-corrélé) = optimiser un proxy sous le mauvais
   cadre est ANTI-aligné par construction. F (à qui ?) = inconnue irréductible.

## 3. Steelman contre la pertinence (assumé)

(1) Échelle ≤ 70M, zéro RLHF/agence — L5a a montré comment les jouets
trahissent. (2) Γ sans valence : *que/quand*, jamais *quoi/bien-mal*.
(3) Pas de levier (L3) — réponse : « pause et évalue » EST le levier demandé
à un early-warning. (4) Adversarialité non testée : gradient-hacking du
sismographe plausible → c'est l'axe A3, pas une objection.

## 4. Axes falsifiables

| # | Axe | H / Kill | Coût |
|---|---|---|---|
| **A1** | **Exposant de persistance des valeurs** : contrainte instillée → entraînement continué → exposant de la direction-contrainte (Γ frame-full) | H : classes de persistance mesurables et prédictibles. Kill : exposants indistincts. Si ça marche : loi de scaling de la dérive des valeurs | jouet→pythia, ~$5-20 |
| **A2** | **Monitoring à étages** : Γ v2 multi-position (scaffold `probe_mode="all4"` déjà dans modal_gamma_llm.py) → seuil → éval ciblée. Démo : capacité attrapée N steps avant benchmark à X% du coût | Kill : v2 sourde ≥1/3 grokkers → pivot sondes poids | ~$2 (re-run test2 all4) |
| **A3** | **Évasabilité du moniteur** (le bord) : pénalité adversariale sur Γ pendant training. Issues : (a) non-furtivité = théorème pratique ; (b) furtif = warning publiable contre les moniteurs internes ; (c) taxe de dissimulation mesurable | toutes les issues sont précieuses — pas de kill, un trilemme mesuré | jouet CPU puis Modal |
| **A4** | **Audit de bifurcation** : multi-seeds, comportement identique, bases internes comparées (checkpoints DÉJÀ sauvés), divergence OOD | Kill : mécanismes toujours isomorphes | gratuit (offline) |
| **A5** | **Gate de vérification** comme primitive d'auto-amélioration sûre ; strictness vs stabilité/capacité ; l'agent peut-il tromper son propre gate (A3 réflexif) | données VG existantes + extension | local |
| **A6** | **Détecteur de collusion multi-agent** : Ω/Ξ sur traces jointes d'agents (synergie = coordination invisible aux audits par-agent). Substrat critical-partition V0 | Kill : synergie indistincte canal/pas-canal | local |

Ordre suggéré : A2 (déjà scaffoldé) → A3 (cheap, neuf, important) → A4
(gratuit) → A1 (le plus original, plus long) → A6 → A5.

## 5. La position (falsifiable)

Le champ traite l'alignement comme un problème de **contrôle**. Le corpus dit :
problème d'**émergence** — et l'émergence ne se contrôle pas (L3) ; elle
s'instrumente (Γ, Ξ), se borne (gates, MDL stop), se canalise (choisir les
invariants qui survivent au flot). **Sismographes, digues, lois de persistance
— pas de volant.** Chaque tiers a son instrument ; chaque instrument a son
kill-switch.
