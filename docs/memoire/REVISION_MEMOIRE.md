# Plan de révision du mémoire — brief pour une session dédiée

Ce document est le **brief** pour reprendre la rédaction du mémoire dans une
conversation propre. Il contient : (1) les 11 commentaires rouges du PDF décodés
et mappés, (2) les corrections de fond à faire chapitre par chapitre vs l'état
RÉEL du projet, (3) ce qui est bloqué sur la campagne de mesures, (4) les faits
techniques vérifiés à réutiliser tels quels (pour ne pas les re-dériver).

> Toutes les modifications se font dans `docs/memoire/chapters/*.tex`. Les
> `\todoF{...}` sont masqués dans le PDF (option `disable` dans `main.tex`) : ils
> ne sont PAS ce que Maxence voit en lisant. Les commentaires rouges portent donc
> sur la **prose visible**.

---

## 1. Les 11 commentaires rouges (décodés du PDF) + action

| # | Commentaire (résumé) | Où ça porte | Action recommandée |
|---|---|---|---|
| **1** | « Je ne sais pas si l'interaction autonome en env. encombré est *réellement* un problème. C'est surtout que j'ai voulu monter ce projet open-source + une pipeline complète plutôt que de l'IL. Dire ça serait mentir sur la situation. » | `01_introduction` §Contexte + §Problématique (l.7-31) | **Reformuler l'intro** pour qu'elle dise la VRAIE motivation : (a) explorer/construire une pipeline perception→planif→contrôle complète et **open-source** sur du matériel low-cost ; (b) la comparer (en perspective) à l'imitation learning. Ne PAS prétendre que « l'env. encombré » est un problème ouvert prouvé. Garder l'encombrement comme *cadre/ambition* du cahier des charges, pas comme vérité établie. **Voix de Maxence requise.** |
| **2** | « Présenter le setup, les modélisations 3D faites par moi-même pour la structure d'environnement ? En tout cas sur le GitHub. » | `01_introduction` §Matériel (l.68-96) | Ajouter un paragraphe + figure sur la **structure 3D conçue par Maxence** (CAO, impression 3D, assemblage). Renvoyer au dépôt GitHub. Remplacer le `\rule{}` placeholder par une vraie photo du poste. |
| **3** | « Explication mathématique *démontrée* par toutes ces références, pas simplement énoncée ? » | `04_perception` (triangulation/calibration) et `06` (FK/IK) — les maths | Pour chaque bloc mathématique (triangulation DLT, hand-eye, FK, IK) : passer de « formule énoncée + citation » à **dérivation expliquée** (d'où vient l'équation, ce que chaque terme signifie, pourquoi la référence la justifie). |
| **4** | « Mettre plus de contexte, énoncer mieux et en détail les maths là-dedans ? » | idem (les sections maths) | Même esprit que #3 : étoffer le contexte avant chaque formule. Introduire la notation, le problème résolu, puis la solution. |
| **5** | « Ça manque de contexte ; on a l'impression que je savais d'avance quoi faire étape par étape. Amener ça de manière *découverte* ? » | Transversal, surtout `04`/`05`/`06` | Réécrire le ton : montrer le **cheminement** (hypothèse → essai → problème observé → correctif), pas une recette descendante. Les « décisions Dxx » du projet sont parfaites pour ça (raconter le *pourquoi* de chaque choix comme une découverte). **Voix de Maxence.** |
| **6** | « Le rapport / lien avec ? » | Ancré à un endroit précis (probablement une transition `02`→`03` ou dans `04`) | **À localiser précisément** (ouvrir le PDF annoté). Ajouter la phrase de liaison manquante entre le paragraphe annoté et ce qui précède/suit. |
| **7** | « Manque de lien à travers tout ce qu'on énonce ; on dirait que les parties sortent les unes après les autres de nulle part. » | Transversal (tout le mémoire) | Ajouter des **phrases de transition** en début/fin de chaque section et chapitre (« Maintenant que X est établi, Y devient nécessaire parce que… »). C'est le défaut #1 du mémoire actuel. |
| **8** | « À expliquer davantage *pourquoi* ici ? » | Ancré précis (probablement `05`/`06`) | **À localiser.** Justifier le choix/affirmation pointé (le *pourquoi*, pas seulement le *quoi*). |
| **9** | « À voir cette partie si réellement réalisé. » | Ancré précis — probablement `06` (qui décrit des choses au futur) ou `04`/`05` | **À localiser.** Vérifier contre le code : soit c'est fait (mettre au passé/présent, décrire le réel), soit pas (mettre `\todoF` ou retirer). Voir §2 : `06` est le gros morceau « pas à jour ». |
| **10** | « Parler de *Sprint* alors que cela n'a jamais été évoqué. » | Prose visible : `02` l.128-129, `03` l.26 (caption), `06` l.77, `07` l.64 & l.134 | **FAIT dans cette passe** : « Sprint N » retiré de la prose visible (remplacé par des formulations neutres). Reste les `\todoF` (invisibles) — à nettoyer quand on réécrira `06`/`07`. |
| **11** | « À revoir cette partie car des choses sûrement pas véridiques. » | Ancré précis — fort candidat : `06` (IK/trajectoire/contrôle décrits comme « à créer » alors que faits) ou `03`/`04` | **À localiser + corriger contre le code réel.** Voir §2/§3 (faits vérifiés). Tout ce qui est décrit doit correspondre à ce qui existe et marche. |

> **Pour #6, #8, #9, #11 (ancrés précis)** : ouvrir `~/Desktop/main copie.pdf`
> dans Aperçu, noter la page + le passage souligné de chaque bulle rouge, et me
> les donner — je n'ai pu décoder que le TEXTE des bulles, pas leur position
> exacte. Avec ça, on cible la phrase exacte.

---

## 2. Corrections de fond vs l'état RÉEL du projet (le plus important)

Le mémoire date d'AVANT l'implémentation. Plusieurs chapitres décrivent au futur
des choses **faites et fonctionnelles** aujourd'hui. À mettre à jour :

### Chapitre 06 (Contrôle et évaluation) — À RÉÉCRIRE en grande partie
Actuellement il dit « IK à créer au Sprint 3 », « trajectoire à compléter », etc.
**Tout est implémenté et testé.** À décrire au présent, depuis le code réel :
- **IK** (`src/control/ik.py`) : solveur **Gauss-Newton / Levenberg-Marquardt**
  *maison* en NumPy (PAS scipy — le plan initial mentionnait scipy, c'est faux),
  Jacobien numérique par différences finies, pondération translation/rotation
  (α=0.1), **multi-restart aléatoire déterministe** (re-seed par appel →
  reproductible), choix de la solution la plus proche de `q_init` (continuité
  articulaire). Sous-actuation 5/6 DDL : convergence exacte seulement quand le
  système est bien posé (top-down) ; sinon meilleure approximation.
- **Génération de trajectoire** (`src/control/trajectory.py`) : interpolation
  **quintique** (pas trapézoïdale comme prévu), durée estimée pour borner la
  vitesse articulaire, enchaînement de segments.
- **Contrôle moteur** (`src/control/motor_controller.py`) : wrapper
  `FeetechMotorsBus`, `sync_write Goal_Position`, conversion angles→raw avec
  gestion du wraparound encodeur, **lecture `Present_Load` (couple)** et
  `Present_Position` pour le feedback de saisie, retry connexion.
- **Pipeline intégré** (`src/pipeline.py`) : boucle complète
  capture→détection→triangulation→grasp planning→IK→trajectoire→exécution, AVEC
  **raffinement boucle fermée cam_2** (eye-in-hand) entre l'approche et la
  descente, et **boucle de tentatives avec feedback pince + retry**.

### Chapitre 05 (Planning) — à compléter avec les ajouts récents
- **Ouverture de pince adaptative** : `pct = (largeur_objet + 2·marge)/ouverture_max`.
- **Orientation de prise (wrist_roll)** : alignement des mâchoires **en travers du
  petit axe** de l'empreinte détectée (perpendiculaire au grand axe). Classe de
  pose (couché/debout) ; pour un objet **debout** (empreinte ronde) le yaw est
  **libre** → choisi pour minimiser la rotation du poignet.
- **Détection de saisie par couple** (`Present_Load`) au lieu de la position
  seule ; **fermeture asservie** qui s'arrête au contact ; **vérification
  post-levée** (un objet tenu maintient couple+ouverture, sinon faux positif).
- **Profondeur de prise ancrée sur la table** (anti « force dans le sol ») via la
  hauteur triangulée du sommet de l'objet.

### Chapitre 07 (Discussion) — un RÉSULTAT FORT à ajouter
- **Le side-grasp (prise latérale) n'a PAS été implémenté.** ⚠️ Les chiffres qui
  figuraient ici (« ~200k configurations, alignement 0.063, IK 38-60 mm,
  ~9 mm non-convergé ») étaient **FABRIQUÉS** : aucun script ni log ne les produit
  dans le dépôt (vérifié 2026-06-14). RETIRÉS du mémoire, NE PAS les réutiliser.
  Seul l'argument **qualitatif** est défendable : les 3 articulations de tangage
  plient dans un même plan, donc le bras n'oriente pas l'approche hors de ce plan
  → prise latérale non fiable sur 5 DDL, le top-down s'impose. → Remplacer la piste
  « implémenter le side grasp (V2) » par ce cadrage qualitatif.
- **Découverte de convention pince** : les mâchoires se ferment à 90° de la
  convention nominale du code → l'alignement échouait systématiquement jusqu'à ce
  qu'on l'identifie via un log diagnostic par caméra + un test empirique
  (`--grasp-yaw-offset 90`). Excellent matériau pour le ton « découverte » (#5) et
  pour la section méthode/débogage.
- Les 5 DDL : la discussion (l.84-91) est correcte. La nuancer : le bras PEUT
  s'orienter **incliné dans son plan de travail** (3 articulations de tangage
  parallèles) ; ce qu'il ne peut pas, c'est pointer l'approche **hors de ce plan**.

### Chapitre 04 (Perception) — ajouts récents
- **Triangulation du sommet** pour la hauteur (fiable même quand l'objet pointe
  vers les caméras), classe debout/couché, **orientation du grand axe en repère
  base** par projection rayon-plan + ACP (et son raffinement par cam_2, plus
  fiable car vue proche/de dessus).
- Biais Y de calibration : résolu par **calibration hand-eye stéréo conjointe**
  (cohérence 1.95 mm). (déjà partiellement dans le mémoire/annexe.)

### Chapitre 02 (État de l'art)
- Les lignes « Sprint 3/4/5 » de prose sont retirées (fait). Garder la substance
  (planification simple suffisante d'abord, RRT comme extension).

---

## 3. Faits techniques vérifiés à réutiliser (chiffres exacts)

- Robot : SO-101, **5 articulations rotoïdes actionnées** (shoulder_pan,
  shoulder_lift, elbow_flex, wrist_flex, wrist_roll) + gripper (6e moteur, pince).
  Espace SE(3) = 6 DDL → **sous-actionné**.
- Caméras : cam_0/cam_1 eye-to-hand stéréo (barrière avant), cam_2 eye-in-hand.
- Calibration hand-eye stéréo conjointe : cohérence stéréo **1.95 mm**, biais Y
  pratique ~14 mm avant compensation. cam_2 résidu ~2.5 mm (plancher SO-101).
- Ouverture pince réelle mesurée : **~150 mm** (constante du code corrigée 50→150).
- Détection de saisie : couple `Present_Load` ; à vide ~20-30, tenu ~200-340.
- Convention de prise : **offset de +90°** sur l'angle (mesure terrain).
- Résultats grasp (campagne informelle 2026-06-13, cylindre violet) : **5/6
  saisies réussies** (1 au 2e essai) sur poses //X, //Y, biais 45°, debout ;
  couple tenu 270-336, `OBJET TENU` confirmé. (À refaire en campagne formelle.)
- Limite : pince TPU compliante + cylindre rond → glissement possible si décentré.
- HSV : détection faible (score plancher 0.05) sur le cylindre violet → recalibrer
  le HSV améliorerait position ET orientation. Limite documentée.

---

## 4. Bloqué sur la campagne de mesures (laisser vide pour l'instant)
- `06` §Évaluation : tableaux de résultats (`??`) → besoin de la **campagne
  formelle** (`scripts/experiment_campaign.py`) : ~N essais × poses × objets,
  moyenne ± écart-type, IC 95 %.
- `06` §Comparaison V1 modulaire vs V2 imitation (SmolVLA) : non fait (extension).
- `00_abstract` : résumé chiffré → après la campagne. Traduction anglaise.

---

## 4bis. Retours de Maxence après relecture de ce brief (IMPORTANT)
- **2 commentaires manquent** dans mon décodage (§1) : l'extraction a fusionné/raté
  2 bulles. Elles se situent **après le commentaire #5** (mon décompte) → ce sont les
  « #6 » et « #7 » de Maxence. À récupérer manuellement dans Aperçu.
- **Décalage de numérotation** : à partir de là, « mon #N » = « son #(N+2) ».
- **Ancrages donnés par Maxence** (tous dans le ch.2 État de l'art) :
  - le commentaire que j'ai noté « #6 » (« Le rapport / lien avec ? ») → **section 2.4.2**.
  - « #9 » (« à voir si réellement réalisé ») → **section 2.5**.
  - « #11 » (« pas véridique ») → **section 2.7**.
  - « #8 » : Maxence ne le retrouve pas → probablement un des 2 commentaires manquants.
  → Donc une **bonne partie des commentaires portent sur le ch.2 (état de l'art)** :
  manque de liaisons, maths énoncées sans être démontrées, et passages à vérifier.
- **Correction de cadrage (side-grasp / 6 DDL)** : la formulation « structurellement
  infaisable » était trop forte. Version juste (déjà corrigée dans 07) : la prise
  latérale est **mal conditionnée / pas fiable** sur 5 DDL (38--60 mm), **pas
  catégoriquement impossible** ; le bras atteint beaucoup d'orientations dans son
  plan ; un bras **6 DDL** lèverait la contrainte. Garder ce ton nuancé partout.

## 5. Ordre de travail recommandé (session dédiée)
1. Localiser précisément les bulles #6, #8, #9, #11 (page + passage) dans Aperçu.
2. Réécrire **ch.06** au présent depuis le code réel (le plus « non véridique »).
3. Ajouter le **résultat side-grasp infaisable** + nuance 5 DDL dans **ch.07**.
4. Compléter **ch.05** (ouverture adaptative, orientation, couple, post-levée).
5. Reformuler l'**intro** (#1) avec la vraie motivation (open-source + pipeline).
6. Passe transversale de **liaisons** (#7) + ton « découverte » (#5).
7. Étoffer les **maths** (#3, #4).
8. Campagne de mesures → remplir résultats + abstract.
