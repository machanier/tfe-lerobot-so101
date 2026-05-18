# Reprendre le projet — guide pour la prochaine session

Document à lire EN PREMIER quand tu reprends le projet (toi ou un nouvel
agent Claude Code dans une nouvelle conversation).

## 0. TL;DR pour un nouvel agent — lire dans cet ordre

1. **`docs/PROJECT_STATUS.md`** : décisions techniques D1-D16 (le quoi & le pourquoi).
2. **`docs/REPERE_BASE.md`** : conventions repère robot (CRUCIAL : `base_link`
   = centre plaque de base à plat sur la table ; X devant le robot ; Y à
   GAUCHE positif / Y à DROITE négatif ; Z au-dessus de la table).
3. **`src/pipeline.py`** : c'est le module central. Toute la logique
   `perception → planning → contrôle` y est. `PipelineConfig` en haut
   liste tous les paramètres ajustables.
4. **`configs/scene.json`** : position de la boîte de dépose et zones
   d'exclusion. La position de la boîte (`drop_box.center_base_m`) est
   en **mètres** dans le repère base et désigne le **centre du FOND** de
   la boîte (le code calcule le dessus automatiquement).
5. **`configs/perception/bias_correction.json`** : compensation dy=+30mm
   appliquée à chaque détection 3D pour corriger le biais hand-eye.

### Ce qui marche (au 2026-05-18)
- Calibration Sprint 1 (intrinsèques + hand-eye 3 cams) : solide.
- Perception 3D avec **HF (OWL-ViTv2)** : robuste, ~0.3 fps détection
  initiale + raffinement cam_2.
- Pipeline pick-and-place : saisie réussie ~70% en HF, dépose dans la
  boîte si scene.json correct.
- Retour à la pose de départ.

### Ce qui ne marche pas / fragile
- **HSV** : détecte la pince orange comme cube orange même après
  recalibration. Marche mal en pratique. Privilégier HF.
- **Saisie irrégulière** : la pince fixe peut percuter l'objet (pas de
  décalage smart du grasp pour positionner l'objet du bon côté).
- **Refinement cam_2 varie 38-50 mm** d'un essai à l'autre (plancher de
  bruit + mouvement de la cam pendant capture).
- **Approche figée en TOP-DOWN** : impossible de saisir par le côté.

### Comment Maxence préfère collaborer
- Ne PAS retirer de fonctionnalités sans demander (ex: bbox vertes dans
  le live display).
- Justifier les compromis avec des chiffres / sources.
- Demander avant de modifier `configs/scene.json` (il a des modifs
  locales avec ses mesures).
- Privilégier 1 fix qui marche à 3 fix qui s'annulent.

## 1. État du projet au 2026-05-18 (session 9)

✅ **Sprint 1 (Calibration)** : complet. Intrinsèques + hand-eye OK.
✅ **Sprint 2 (Perception)** : V1 HSV + V2 OWL-ViTv2 fonctionnels.
✅ **Sprint 3 (Planning + Contrôle)** : grasp top-down + IK + trajectoire.
✅ **Sprint 4 (Boucle fermée)** : raffinement cam_2 actif.
✅ **PREMIÈRE SAISIE RÉUSSIE** le 17 mai 2026 avec `pick_and_place.py`.

### Session 9 (2026-05-18) — diverses améliorations pipeline

Refactos et fixes appliqués dans `src/pipeline.py` (commits 79b5899,
86e647f, 621b5cf, et celui-ci) :

1. **IK dédié pour la dépose** : `_ik_drop = IKSolver(rotation_weight=0.05)`.
   Pour `drop_above`/`drop_release` on accepte une orientation pince
   approximative (≤15° de travers) pour gagner en précision position
   (~2 mm vs 47 mm avant avec rotation_weight=0.1 standard).
   Compromis 0.05 = équilibre entre précision (0.01 trop tordu, faisait
   se cogner le bras) et conformité top-down (0.1 ratait la cible).

2. **`home_from_session_start: bool = True`** : robot revient à la pose
   du lancement du script (pas une pose hardcodée).

3. **Convention scene.json clarifiée** : `center_base_m` = centre du
   **FOND** de la boîte (face posée sur la table). Le dessus est calculé
   auto = `center_base_m[2] + dimensions_m[2]`. Log détaillé au démarrage
   + WARN si position semble incohérente (proche base / hors workspace).

4. **`home_gripper_pct: float = 5.0`** : pince quasi-fermée à la fin
   (au lieu de grand ouverte). Évite les accrochages cables.

5. **Live display avec WORKER THREAD asynchrone** : la détection HF
   tourne dans un thread séparé (queue size=1 pour les frames). Le
   callback principal n'est jamais bloqué → trajectoire fluide même en
   HF. Cache pré-peuplé avec les détections initiales pour que les bbox
   vertes apparaissent dès la première frame du live.

6. **Pince s'ouvre PROGRESSIVEMENT** durant le segment courant → approach
   (au lieu d'être grand ouverte dès le démarrage).

7. **Sortie verticale de la boîte** : nouveau segment
   `drop_release → drop_above` avant `→ safe` pour éviter que la pince
   fixe ne cogne le bord intérieur de la boîte en remontant en diagonale.

8. **Ordre display caméras** : `cam_1 | cam_0 | cam_2` (au lieu de
   `cam_0 | cam_1 | cam_2`) pour respecter la perspective du robot.
   Tile agrandi à 960×540 (1.5×). Bandeau noir avec nom de la cam.

9. **Bug `_build_full_trajectory`** corrigé (NameError q_home/dur_home en
   mode `--no-closed-loop`/`--dry-run`).

### Limites identifiées (en cours, à investiguer)

- **HSV fragile** : détecte la pince orange comme cube orange. À
  recalibrer en cliquant uniquement sur les pixels au CENTRE du cube
  (pas les bords ni les ombres). Et augmenter `score_threshold` du HF
  pour le mode --detector hf si trop de faux positifs.
- **Refinement cam_2 varie 38-50 mm d'un essai à l'autre** : c'est le
  plancher de bruit de la détection HF combiné au mouvement de cam_2
  pendant l'exposition. Pas trivial à régler.
- **Saisie qui rate parfois** : la pince fixe percute parfois l'objet
  avant la fermeture. Solution V2 : décaler le grasp de demi-largeur
  pour que l'objet soit du côté pince fixe (non implémenté, demande de
  connaître la convention pince fixe/mobile dans l'URDF).
- **Approche top-down rigide** : ne peut pas saisir un cube par le côté
  ou un objet posé contre un mur. À étendre (V2, hors scope bachelor ?).

🟡 **À continuer** : amélioration de la robustesse, recalibration HSV,
évitement obstacles (Sprint 4.5), SmolVLA comparatif (Sprint 5),
rédaction LaTeX.

## 2. Quick-start après git clone (ou nouvelle session)

```bash
git pull origin main
source venv/bin/activate

# Verifie que tout est OK
python scripts/check_calibration.py
# → doit afficher [OK] partout

# Voir où on en est sur le pipeline pick-and-place
python scripts/pick_and_place.py --target orange_cube --detector hf --dry-run
# (dry-run = simule, ne bouge pas le robot)

# Test LIVE (le robot bouge !)
python scripts/pick_and_place.py --target orange_cube --detector hf --display
# (--display = affiche les cameras pendant l'execution)
```

## 3. Documents-clés à lire pour comprendre

| Fichier | Quand le lire |
|---|---|
| [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md) | Document VIVANT. Toutes les décisions D1-D16. À lire en 1er. |
| [`docs/REPERE_BASE.md`](REPERE_BASE.md) | Référence sur l'origine du repère robot. À lire AVANT toute mesure physique. |
| [`docs/bachelor_chanier_25_26.pdf`](bachelor_chanier_25_26.pdf) | Cahier des charges Partie I. La référence ultime. |
| [`docs/memoire/`](memoire/) | Brouillon LaTeX du mémoire (33 pages déjà rédigées). |

## 4. Configuration spécifique du poste de Maxence

### Matériel
- SO-101 follower + leader, 6 moteurs Feetech STS3215 chacun (différents gear ratios).
- **Pince TPU XLRobot avec grip de tennis** ajouté pour la saisie (meilleure adhérence sur les surfaces lisses).
- 3 caméras USB 1920×1080 :
  - cam_0 + cam_1 : barrière avant, baseline ~109 mm en Y (stéréo eye-to-hand).
  - cam_2 : eye-in-hand sur la pince.
- 2 hubs USB séparés (D12) pour répartir cam_0+1 d'un côté, cam_2+robot de l'autre.
- MacBook Pro M4, macOS, Python 3.12.

### Repère base — origine (0, 0, 0)
Centre géométrique de la plaque inférieure (`base_so101_v2.stl`), à plat sur la table.
- shoulder_pan motor à `(+38.8, 0, +62.4)` mm depuis base_link.
- Pose home (config zéro FK) : effecteur attendu à `(+391, 0, +227)` mm.
- Voir `docs/REPERE_BASE.md` pour les détails.

### Compensation systématique du biais
- `configs/perception/bias_correction.json` : **dy = 30 mm** (mesuré expérimentalement).
- Soustrait à chaque détection 3D. Permet de ne pas modifier `gt_test.json` à chaque test.
- À réajuster si la calibration hand-eye est refaite.

### Objets utilisés (testés)
4 primitives colorées (HSV V1) :
- orange_cube (cube 30 mm) — **objet de référence pour les tests**
- blue_rectangular_box
- purple_cylinder
- black_triangular_prism (mode `color_mode: black`)

5 objets quotidien (HF V2, non calibrés HSV) :
- white_mug (imprimé 3D)
- pen (stylo Bic)
- rubiks_cube
- tissue_box (paquet mouchoirs)
- tall_plastic_cup (gobelet)

### Boîte de dépose
`configs/scene.json` → `drop_box.center_base_m`. À mesurer physiquement après chaque
déplacement de la boîte.

## 5. Bugs résolus et limites connues (à mentionner dans le mémoire)

### Résolus durant les itérations
1. HSV ne gère pas noir/blanc/gris (D8) → résolu par `color_mode`.
2. Faux positifs robot orange ↔ cube orange → résolu par exclusion_zones (D9).
3. 3 caméras 1080p saturent USB → résolu par MJPG (D10) + 2 hubs (D12).
4. Bus moteur "Lock id_=N" → retry 5x avec délai progressif.
5. **Repère base** mal compris (D11 corrigée) → docs/REPERE_BASE.md.
6. IK plages divisées par 2 (bug π vs 2π) → corrigé, +marge 3°.
7. Biais Y systématique stéréo +30 mm → compensation auto (D11 + bias_correction.json).
8. Compensation Y créait faux reproj_error → corrigé : reproj **avant** compensation.

### Limites identifiées (à documenter dans la discussion)
- OWL-ViTv2 = ~3-5 sec par image sur M4 (trop lent pour servoing temps réel).
- HSV échoue sur noir/blanc/transparent + sensibilité éclairage.
- Sous-actuation SO-101 5/6 DDL (toutes orientations pas atteignables).
- Pas de détection de collision (le robot frappe les obstacles s'il y en a).
- Faux positifs sémantiques OWL-ViTv2 (gourde → "purple_cylinder", essuie-tout → "white_mug").

## 6. Prochaines étapes (priorité décroissante)

### A) Validation expérimentale rigoureuse (Sprint 5 partiel)
20 essais de pick-and-place avec mesures :
- Taux de réussite (success/total)
- Précision finale (mm)
- Nombre de "retouches" (cas où le cube est déplacé puis ressaisi)
- Temps moyen par essai
→ Pour le tableau de résultats du chapitre 6 du mémoire.

### B) Évitement obstacles (Sprint 4.5, objectif 3 du cahier des charges)
Détecter et **éviter** les obstacles dans la trajectoire :
1. Le `pose_estimator` peut déjà détecter les objets autres que la cible.
2. Le grasp planner / trajectoire devrait les contourner.
3. Méthode simple : ajouter "marge" autour des autres détections, vérifier que
   la trajectoire ne traverse pas. Sinon : RRT (LaValle 2006) → c'est plus complexe.

### C) Recalibration HSV (5 min côté hardware)
Refaire `python scripts/calibrate_hsv.py` sous l'éclairage du jour pour avoir
des détections HSV fiables (utile pour la comparaison V1/V2 du mémoire).

### D) SmolVLA comparatif (Sprint 5 final)
1. Téléopérer le pick-and-place ~50 fois (datasets LeRobot).
2. `lerobot-train --policy=smolvla` (~quelques heures GPU).
3. Comparer SmolVLA vs notre pipeline modulaire sur les mêmes scènes.
4. Tableau dans le mémoire chapitre 6.

### E) Rédaction LaTeX
Compléter les sections vides de `docs/memoire/chapters/06_control_evaluation.tex`
et `07_discussion.tex` avec les chiffres expérimentaux obtenus.

## 7. Réponses aux questions récurrentes

**"Pourquoi le pipeline appelle l'IK et pas la FK ?"**
Les deux sont utilisées :
- **FK** : "j'ai les angles, où est la pince" → utilisé pour calculer `T_base_cam2` en temps réel (cam_2 eye-in-hand bouge avec le bras).
- **IK** : "j'ai la pose désirée, quels angles" → utilisé pour calculer les angles à envoyer aux moteurs pour atteindre approach/grasp/retract/drop.

**"Pourquoi `gt_test.json` si le robot détecte l'objet seul ?"**
- `pick_and_place.py` : **détecte** l'objet en live, n'utilise PAS gt_test.json.
- `check_perception.py` : **valide** la précision en comparant à un ground truth → utilise gt_test.json.

**"Comment le robot 'sait' où sont ses pinces ?"**
URDF définit `gripper_frame_link` = point virtuel au centre de la pince. FK calcule
sa position pour n'importe quelle config articulaire. Le grasp planner positionne ce
repère au centre du cube → pince attrape. **Pas de détection de collision physique**.

## 8. Comment poursuivre dans une nouvelle session Claude Code

Quand tu démarres une nouvelle conversation :
1. Le nouvel agent ne se souviendra pas des détails précédents.
2. **Donne-lui** le chemin `/Users/maxencechanier/Projects/tfe-lerobot-so101/docs/NEXT_SESSION.md` pour qu'il se mette à jour.
3. Il pourra lire automatiquement `PROJECT_STATUS.md`, `REPERE_BASE.md`, et le brouillon LaTeX.
4. La mémoire utilisateur (`~/.claude/projects/.../memory/project_tfe.md`) sera lue automatiquement et résumera la situation.

## 9. Fichiers tracés sur git à ne PAS oublier de commit

```bash
git status
# Si tu vois des modifs locales sur :
#   - scripts/run_perception.py (FPS counter perso)
#   - configs/perception/gt_test.json (ground truth de tes essais)
#   - configs/scene.json (boîte de dépose)
# → commit-les avec git add + git commit -m "..."
```
