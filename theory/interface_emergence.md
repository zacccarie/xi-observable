# L'émergence aux interfaces — ouverture théorique

Déclenchée par une observation viz (2026-06-01) : les motifs CML ressemblent à de la
réaction-diffusion. Question : produire une dynamique « lampe à lave » (fronts mobiles
lisses) comme le paramètre *Evolution* du bruit fractal After Effects. Identifier le
mécanisme, en faire un apport.

## Mécanisme mathématique

**AE « Evolution » (bruit fractal / turbulence)** : le bruit de Perlin/simplex est un
champ 3D `N(x,y,z)`. L'image 2D = coupe `z = const`. Animer `z = t` = traverser le
volume. Continuité C¹ selon z ⟹ métamorphose lisse. **Pas une dynamique** : navigation
dans un champ figé. Zéro émergence.

**Lampe à lave (vraie dynamique)** = **mouvement d'interface**. Une isosurface
`∂{f = c}` se déplace à vitesse normale
```
v_n = − ∂_t f / |∇f|
```
Gouvernée par courbure + forçage (mean-curvature flow, Allen–Cahn, Gray–Scott) :
```
∂_t u = ε² ∇²u − W'(u)            (Allen–Cahn, paroi de domaine)
∂_t u = Du ∇²u − uv² + F(1−u)     (Gray–Scott, réaction-diffusion)
∂_t v = Dv ∇²v + uv² − (F+k)v
```
Diffusion = tension de surface (lisse les fronts). Réaction = instabilité (les crée).
Séparation d'échelles `Du ≠ Dv` (Turing) = blobs stables mobiles → look lampe à lave.

Le CML scintille car update local **sans diffusion dominante** (pas de tension de
surface) → bruit temporel par cellule. Gray–Scott lisse par diffusion → fronts continus.

## Conjecture INTERFACE (apport au programme)

Un motif réaction-diffusion = **domaines** (régions homogènes) + **interfaces** (fronts).

> **L'émergence — synergie irréductible (Ω<0) — est localisée aux INTERFACES, pas dans
> le bulk.**
> - **Bulk** : cellules identiques, chacune prédit ses voisines → **redondance** (Ω>0)
> - **Interface** : la frontière encode la *relation* entre domaines, irréductible à
>   chaque côté pris seul → **synergie** (Ω<0)

### Unification

L'interface est le concept général ; l'edge of chaos en est un cas particulier :

| Phénomène du programme | Interface dans la dimension… |
|------------------------|-------------------------------|
| Edge of chaos (Ising Tc, RBN K_c, Kuramoto K_c) | espace des paramètres de contrôle |
| Transition générative LLM (T≈1.2) | température de sampling |
| Couche critique (profondeur LLM) | profondeur du réseau |
| Domain wall (réaction-diffusion) | espace physique |

**Même objet.** L'émergence vit aux frontières, quelle que soit la dimension (temps,
espace, paramètre, profondeur). Réconcilie :
- **A** (émergence ≡ chaos) : le chaos spatio-temporel = prolifération d'interfaces
- **B** (Φ aux structures cohérentes) : les structures cohérentes SONT les domaines ;
  l'émergence est à leurs bords, pas en leur cœur

### Prédiction falsifiable

Ω locale (patchs) corrélée négativement au gradient |∇field| : synergie pique aux
fronts mobiles, redondance dans le bulk. Test : `interface_emergence.py`.

### Conséquences testables additionnelles

- **#11** : la densité d'interface (longueur totale de front / surface) devrait suivre
  la synergie globale d'un système — un système « plus émergent » a plus d'interface.
- **#12** : à l'edge of chaos, la densité d'interface est maximale (fronts à toutes
  échelles = scale-invariance) ; en ordre (un domaine) ou chaos profond (décorrélé,
  pas de domaine net) elle chute. Reconnecte la synergie-pique-à-l'edge.
- **#13** : dans un LLM, les « interfaces » = positions de tokens où la représentation
  bascule (changement de sujet, frontière syntaxique). Y mesurer la synergie locale.

## RÉSULTAT — conjecture RÉFUTÉE, finding récupéré (`interface_emergence.py`)

Gray–Scott L=128 (F=0.037, k=0.06, labyrinthe), Ω par patch 2×2 sur 240 frames.

- corr(Ω, |∇|) = **+0.146** (positif, pas négatif)
- Ω bulk = +1.59 | Ω interface = **+1.95** (PLUS redondant)
- **frac_synergie = 0 PARTOUT** — Ω>0 partout, aucune synergie

**Conjecture interface FAUSSE.** À l'échelle d'un patch, l'interface = cellules qui
traversent le front ENSEMBLE (mouvement commun) = redondance, pas synergie. 6e
généralisation prématurée du programme réfutée par test.

### Finding récupéré (fort) : complexité visuelle ≠ émergence info-théorique

Gray–Scott produit un labyrinthe magnifique ET info-théoriquement **redondant** (Ω>0
partout). Pourquoi : le couplage est le **laplacien** (diffusion) = moyenne pondérée
locale = **monotone/canalisant**. Or A a montré : émergence ⟸ non-monotonie de C.
La diffusion est monotone ⟹ redondance, malgré la beauté du pattern.

> **L'œil est trompé.** Un système peut être visuellement complexe (patterns Turing,
> lampe à lave) et info-théoriquement redondant. La complexité spatiale perçue ≠
> émergence (synergie). Confirme A par un nouveau système : la réaction-diffusion, icône
> de l'auto-organisation, ne porte PAS d'émergence synergique — son couplage moyennant
> est canalisant.

### Conséquence pour la viz

Mettre Gray–Scott (lampe à lave) dans une viz « émergence » serait trompeur : le pattern
est redondant. Honnêteté requise — soit message « beau mais redondant » (pédagogique,
contre-intuitif), soit garder le CML (qui a une vraie bascule redondance→synergie en
régime chaotique, cf data.json). La beauté du front mobile ≠ l'émergence.

### Ce qui survit du mécanisme

L'identification mathématique tient (Evolution AE = coupe bruit 3D ; lampe à lave =
mouvement d'interface par courbure). Mais le pont « interface = émergence » est coupé.
L'edge of chaos reste une interface dans l'espace des PARAMÈTRES — c'est là que la
synergie bascule (A, RBN, CML chaos) — mais les interfaces SPATIALES d'un champ
diffusif ne portent pas de synergie. Distinction : interface paramétrique (émergente)
vs interface spatiale diffusive (redondante).
