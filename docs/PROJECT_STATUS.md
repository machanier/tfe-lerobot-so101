# État du projet — TFE LeRobot SO-101

Document vivant qui résume où en est le projet, les décisions techniques
importantes prises au fil du parcours, les pièges connus, et ce qui reste à
faire. À garder à jour quand le périmètre évolue.

Pour reprendre rapidement, sauter à [§Quick-start](#quick-start).

## 1. Contexte

**TFE** : Saisie d'objets en environnement encombré pour un bras robotique
assisté par vision. Bachelor informatique UNIGE 2025-2026.
**Étudiant** : Maxence Chanier. **Encadrant** : Guido Bologna.

**Partie I** (cahier des charges, objectifs détaillés) : **ACQUIS** le
21.01.2026. Voir `docs/bachelor_chanier_25_26.pdf`.

**Partie II** (mise en œuvre, en cours) : architecture **perception ↔
planification ↔ contrôle** multi-caméras (`cam_0`/`cam_1` stéréo eye-to-hand
sur la barrière avant, `cam_2` eye-in-hand sur l'effecteur), avec :

- sélection dynamique de point de vue (active vision),
- évitement d'obstacles,
- replanification en boucle fermée perception → action,
- stratégies de saisie adaptées à la géométrie.

Approche V1 : règles heuristiques + critères géométriques. Extensions
possibles : imitation learning (références dans `docs/references/`).

**Métriques d'évaluation imposées** par le cahier des charges :
taux de réussite, nombre de replans, collisions évitées, précision de la prise.

## 2. État au 2026-05-15 (Sprint 2 — fin Phase calibration + démarrage perception)

### Hardware
- SO-101 leader + follower assemblés, téléopération opérationnelle.
- 3 caméras USB 1920×1080 fixées par la structure 3D imprimée définitive.
- Boîte fixée au sol comme point de dépose pour le pick-and-place.

### Phase 1 — Calibration : COMPLÈTE

Validable à tout moment via :

```bash
python scripts/check_calibration.py
```

Résultats actuels (`mean / max`) :

| Bloc | Mesure | Verdict |
|---|---|---|
| Intrinsèques 3 caméras | 0.14 – 0.20 px reprojection | OK |
| Plages moteur vs URDF | conformes | OK |
| Hand-eye `cam_0` (eye-to-hand) | 5.82 / 12.78 mm — 2.1° / 3.4° | ACCEPTABLE |
| Hand-eye `cam_1` (eye-to-hand) | 6.80 / 15.04 mm — 1.9° / 3.0° | ACCEPTABLE |
| Hand-eye `cam_2` (eye-in-hand) | 2.48 / 4.77 mm — 0.4° / 0.8° | BON |
| Baseline stéréo `cam_0` ↔ `cam_1` | 109 mm (typique 60-250) | OK |
| Self-tests modules cinématiques | tous OK | OK |

Les résidus `cam_0`/`cam_1` correspondent au **plancher de bruit du SO-101**
(~5-10 mm de répétabilité d'après la spec constructeur). Ils seront compensés
par la replanification en boucle fermée (objectif 4 du cahier des charges).
`cam_2` est meilleure parce que l'eye-in-hand a un setup physique plus rigide
(damier fixe sur la scène, pas de wobble de collage sur la pince).

## 3. Décisions techniques importantes

### D1 — Recalage du `Homing_Offset` de `wrist_roll`

**Problème découvert** : la calibration LeRobot d'origine avait laissé
`wrist_roll` dans un état où sa course physique (~334°) chevauchait la
couture 0/4095 de l'encodeur 12 bits. Conséquence : les angles calculés
par `motor_to_angle.py` sautaient de ±360° pour certaines positions, et la
cinématique directe donnait des poses incohérentes.

**Solution** :
1. `scripts/measure_wrist_roll.py` — enregistrement continu d'un balayage
   butée à butée, déroulage de l'encodeur, mesure du vrai centre.
2. `scripts/fix_wrist_roll_calibration.py` — calcule le `Homing_Offset` pour
   que ce centre lise `Present=2047` (milieu encodeur), l'écrit dans le
   servo, met à jour `configs/calibration_follower.json`
   (`wrist_roll: range=[150, 3944]`, `homing_offset=1494`).
3. `scripts/verify_wrist_roll.py` — affichage en direct (torque coupé) pour
   confirmer que centre → 0° et butées → ±167°.

`motor_to_angle.py` est désormais wraparound-aware par défaut (no-op pour les
joints sains, défensif si jamais un autre joint dérivait).

### D2 — Damier asymétrique 9×6 à la place de 7×7

**Problème** : `cv2.findChessboardCorners` détecte un damier 7×7 carré dans
**4 orientations équivalentes** (rotations 0°/90°/180°/270° du « (0,0) »).
Le solveur hand-eye reçoit alors des poses incohérentes entre elles, ce qui
explose les résidus.

**Solution** : `scripts/generate_chessboard.py` génère désormais par défaut
un **PNG 9×6 asymétrique** à 300 DPI (avec DPI embarqué dans la métadonnée
PNG pour que l'imprimante respecte les millimètres). L'asymétrie supprime la
4-fold. Le solveur hand-eye gère encore la 2-fold résiduelle de l'asymétrique.

### D3 — Solveur hand-eye « robuste » (alignment + outlier rejection)

**Pipeline** (`src/calibration/handeye.py`) :

1. Solve initial sur toutes les poses (PARK).
2. Identification du cluster majoritaire de `T_gripper_target` (eye-to-hand)
   ou `T_base_target` (eye-in-hand).
3. Réalignement de chaque pose : choix parmi les corrections géométriques
   (rotation + décalage d'origine du damier) celle qui rapproche du cluster.
4. Rejet itératif des 10 % de poses au plus grand résidu, jusqu'à
   stabilisation ou plancher de 20 poses.

C'est ce qui permet d'atteindre le plancher de bruit du SO-101. Typiquement
on retient 20-25 poses sur 60-70 capturées.

### D4 — FK cinématique directe en numpy, depuis l'URDF

`src/calibration/forward_kinematics.py` parse `configs/so101_new_calib.urdf`
(récupéré depuis le repo officiel TheRobotStudio/SO-ARM100) avec
`xml.etree.ElementTree` (stdlib) et compose la chaîne cinématique en numpy.
**L'URDF est la seule source de vérité géométrique** : pour modifier le
modèle on remplace le fichier, pas le code.

Alternative écartée : `placo` + `pinocchio` (le stack utilisé par LeRobot).
Trop lourd (C++ bindings, met à jour numpy), boîte noire pour la rédaction.
À reconsidérer si on a besoin d'IK avancée pour la planification (Sprint 3).

### D5 — Convention de calibration moteur « new_calib »

Le SO-ARM100 a deux URDF (`new_calib` et `old_calib`). On utilise
`new_calib` : zéro de chaque joint = milieu de sa plage. Correspond exactement
à la formule `(raw - mid) * 360 / 4095` qu'applique LeRobot dans `_normalize`
(cf `lerobot/motors/motors_bus.py:858`).

### D6 — Architecture perception modulaire (Sprint 2)

**Décision** : on isole la perception dans `src/perception/`, derrière une
interface stable (`Scene` / `ObjectInstance`), au lieu de coder en monolithe.

**Pourquoi** :
- Le cahier des charges (Partie I) impose la décomposition
  **perception → planification → contrôle**. Toute architecture monolithique
  serait incohérente avec l'évaluation.
- On veut pouvoir **comparer méthodes** dans le mémoire (HSV classique vs
  détecteur appris) : ça impose que `ObjectDetector` soit une ABC.

**Choix de baseline V1** : `HSVDetector` (seuillage HSV + contours OpenCV)
sur les 4 primitives colorées. Méthode classique, déterministe, reproductible,
zéro dépendance ML. **Permet d'isoler la contribution de la géométrie**
(calibration + triangulation) avant d'introduire les incertitudes d'un
détecteur appris. Standard en robotique de manipulation pour benchmarker une
chaîne de calibration. Référence : Forsyth & Ponce, *Computer Vision: A
Modern Approach*, ch. 6.

**Choix de triangulation** : DLT linéaire (Hartley & Zisserman 2018, ch. 12)
sur centres de détection, à partir de `T_base_cam` connus (hand-eye). Avec
un test d'erreur de reprojection comme garde-fou. Précision sub-mm sur le
synthétique (cf `tests/perception/test_pipeline.py`).

**Fallback monoculaire** : PnP (cv2.solvePnP / IPPE_SQUARE) avec les
dimensions métriques connues, pour le cas où seule `cam_2` voit l'objet.
Limite connue : ambiguïté planaire pour 4 coins coplanaires alignés au plan
image (Lepetit et al. 2009, *EPnP*, sec. "Planar case"). Acceptable : la
validation Sprint 2 repose sur la stéréo.

### D8 — HSV : distinction `color_mode` chromatic / black / white / gray (Sprint 2)

**Problème découvert (2026-05-16)** : la première calibration HSV de Maxence sur 5 objets a montré que **noir et blanc ne sont pas des couleurs en HSV**. Pour le noir, V≈0 rend H et S indéterminés (mathématiquement) ; pour le blanc, S≈0 rend H indéterminé. Conséquence : le seuillage par H sur un objet noir donnait `H ∈ [103, 168]`, couvrant simultanément le bleu et le violet → confusions massives entre objets sombres.

**Solution** : `HSVRange` a maintenant un champ `color_mode` qui contrôle quelles bornes sont appliquées :
- `"chromatic"` (défaut) : seuillage H + S + V comme avant.
- `"black"` : V ≤ v_hi seulement (H, S ignorés).
- `"white"` : S ≤ s_hi ET V ≥ v_lo (H ignoré).
- `"gray"` : S ≤ s_hi ET v_lo ≤ V ≤ v_hi (H ignoré).

`calibrate_hsv.py` détecte automatiquement le bon mode en regardant la médiane de S et V des pixels échantillonnés (cf `_detect_color_mode`).

**Référence** : c'est la propriété fondamentale du modèle HSV décrite dans tout manuel de Computer Vision (Forsyth & Ponce 2012, ch. 6.1.2 ; Smith 1978 *Color Gamut Transform Pairs*, papier original HSV).

### D9 — Zones d'exclusion + bornes workspace dans `configs/scene.json` (Sprint 2)

**Problème découvert (2026-05-16)** : le filament orange du robot SO-101 est physiquement la même teinte que le cube orange. HSV ne peut PAS les distinguer (par définition).

**Solution** : `configs/scene.json` déclare des **zones d'exclusion spatiales** (cylindre autour de la base robot) et des **bornes workspace**. Le `pose_estimator` filtre toute estimation 3D qui tombe dedans → la "détection" du robot orange est rejetée car la position triangulée tombe sur la base, hors workspace utile.

C'est une approche **prior géométrique** : on injecte la connaissance "le robot est ici, donc ce n'est pas un objet à saisir". Économique en compute, robuste, justifiable.

`configs/scene.json` contient aussi la position de la **boîte de dépose** (fixée au sol, mesurée une fois). Elle servira au Sprint 3 (cible du retract) et au Sprint 4 (obstacle d'évitement).

### D10 — Codec MJPG pour 3 caméras USB 1080p (Sprint 2)

**Problème découvert (2026-05-16)** : sur macOS, ouvrir 3 caméras USB 1080p simultanément sature la bande passante USB → la 3ème caméra (cam_2) échoue en boucle au `grab()`.

**Cause** : OpenCV demande par défaut un flux YUYV non-compressé = ~125 Mo/s par caméra à 1080p30 → 375 Mo/s total, dépasse USB 3.0.

**Solution** : forcer le codec **MJPG** (Motion-JPEG) qui compresse l'image dans la caméra avant l'envoi USB. Coût : ~12 Mo/s par caméra (×10 moins). Effet sur la calibration intrinsèque : nul (même modèle pinhole, légère compression JPEG invisible à 1080p).

Combiné à un **warmup** (5 frames par caméra après open) pour stabiliser l'autoexposure et l'allocation USB, cela résout le problème cam_2.

### D7 — Extension V2 prévue : détecteur HF dans l'écosystème LeRobot

**Décision** : la classe `HFDetector` (stub) reste dans `detector.py`,
prête à être implémentée plus tard avec `transformers.AutoModelForZeroShotObjectDetection`
(OWL-ViTv2 ou Grounding-DINO).

**Pourquoi cet alignement** :
- Le projet est bâti sur **LeRobot** (Hugging Face) : on garde la cohérence
  d'écosystème (les détecteurs HF utilisent le même `transformers` que les
  policies HF type SmolVLA / ACT).
- L'open-vocabulary detection couvre les **objets quotidien** (rubik's cube,
  paquet de mouchoir, stylo, tasse, gobelet) que le HSV ne peut pas atteindre.
- C'est une **contribution comparative** valorisable dans le mémoire :
  approche classique HSV vs approche moderne foundation-model, sur le même
  pipeline géométrique. Référence biblio : SmolVLA (Shukor et al. 2025).

**Périmètre V2 exclu pour l'instant** : pose 6D (FoundationPose), VLA
end-to-end. Mentionnés comme ouverture du mémoire, pas comme contribution.

## 2bis. Sprint 2 — Détail

État au 2026-05-15 : **modules implémentés et auto-testés**. Reste à
calibrer les couleurs sous l'éclairage du poste et à mesurer l'erreur de
localisation 3D avec le ground truth pied à coulisse.

### Modules livrés (tous auto-testés via `python scripts/check_calibration.py`)

| Fichier | Rôle | Test |
|---|---|---|
| [`src/perception/scene.py`](../src/perception/scene.py) | Dataclasses (`Frame`, `Detection2D`, `ObjectInstance`, `Scene`) | self-test |
| [`src/perception/robot_state.py`](../src/perception/robot_state.py) | Lecture moteurs + FK (3 modes : live / raw / angles) | self-test |
| [`src/perception/camera_io.py`](../src/perception/camera_io.py) | `MultiCamera` (live) + `ReplayCamera` (offline), compose `T_base_cam` selon eye-to/eye-in-hand | self-test |
| [`src/perception/detector.py`](../src/perception/detector.py) | `ObjectDetector` (ABC) + `HSVDetector` (V1) + `HFDetector` (stub V2) | self-test |
| [`src/perception/pose_estimator.py`](../src/perception/pose_estimator.py) | Triangulation stéréo DLT + raffinement + PnP mono fallback | self-test |
| [`tests/perception/test_pipeline.py`](../tests/perception/test_pipeline.py) | Test d'intégration synthétique du pipeline complet | 4 cas |

### Scripts opérationnels

| Script | Quand l'utiliser |
|---|---|
| `scripts/calibrate_hsv.py` | **À lancer une fois sous l'éclairage final** : pointe la cam_0 sur chaque primitive, clique des pixels, sauve `configs/perception/hsv_specs.json`. |
| `scripts/record_perception_frames.py` | Enregistre un dataset de validation (3 caméras + moteurs) au format manifest pour `ReplayCamera`. |
| `scripts/run_perception.py` | Pipeline complet, 3 modes : `--mode live` (boucle vidéo), `--mode replay` (rejoue un dataset), `--mode oneshot` (1 trame + sauvegarde JSON). |
| `scripts/check_perception.py` | **Validation chiffrée** : pose des objets à des positions mesurées au pied à coulisse (`--gt FILE` ou `--interactive`), compare, sort un rapport mm. |

### Validation expérimentale à mener (procédure)

1. `python scripts/calibrate_hsv.py --camera 0` → calibrer les 4 primitives sous l'éclairage du poste.
2. Poser un cube rouge à une position mesurée précisément (X, Y, Z en mm depuis la base, mesure au pied à coulisse) et noter le triplet.
3. Créer `outputs/perception/gt_test.json` :
   ```json
   {"objects":[{"label":"red_cube","position_base_mm":[X,Y,Z]}]}
   ```
4. `python scripts/check_perception.py --gt outputs/perception/gt_test.json`
5. **Critère de succès** (cf cahier des charges + plancher de bruit hand-eye) :
   erreur moyenne ≤ 10 mm. Au-delà : recalibrer HSV ou re-vérifier hand-eye.

### Roadmap perception (V2, à venir)

- Implémenter `HFDetector` (OWL-ViTv2) → couverture des 5 objets quotidien (rubik's, mouchoir, stylo, tasse, gobelet).
- Mesurer mAP / précision de localisation 3D sur dataset enregistré par `record_perception_frames.py` : ce sera une **section comparative** du mémoire.
- Étudier l'extension active vision (le module `camera_io` est déjà compatible : un seul provider pour les 3 caméras).

## 4. Roadmap forward

| Sprint | Contenu | État | Livrables principaux |
|---|---|---|---|
| 1 | Calibration hand-eye | **✅ FAIT** | `configs/handeye_cam_*.json`, `scripts/check_calibration.py` |
| 2 | Perception : détection + triangulation 3D + scène | **🟡 CODE LIVRÉ, validation à mener** | `src/perception/` (5 modules, tous auto-testés), 4 scripts CLI |
| 3 | Planification + contrôle : grasp + IK + interface LeRobot Python | À faire | `src/planning/`, `src/control/`, `src/pipeline.py` |
| 4 | Boucle fermée : replanification + active vision minimale | À faire | enrichissement `src/pipeline.py` |
| 5 | Évaluation expérimentale + rédaction | À faire | rapport TFE + campagne d'expériences avec les 4 métriques |

Validation Sprint 2 (à faire) : cube colorée posé à une position mesurée au
pied à coulisse → triangulation doit retomber dessus à ~10 mm près (procédure
détaillée en [§2bis](#2bis-sprint-2--détail)).

## 5. Pièges connus (à se rappeler)

- **Damier symétrique → 4-fold ambiguity**. Toujours utiliser un damier
  asymétrique pour toute future calibration.
- **`wrist_roll` seam crossing** : si un servo est remplacé ou la
  calibration moteur refaite, relancer `measure_wrist_roll.py` +
  `fix_wrist_roll_calibration.py`.
- **Le solveur hand-eye retient ~30 % des poses** par design (rejet
  d'outliers). Capturer plus n'aide pas si la diversité angulaire est faible.
  Viser >65° d'écart moyen entre poses (vérifié par
  `check_extrinsic_capture.py`).
- **`calibration_follower.json` doit rester en phase avec le hardware**. La
  fonction `sync_calibration_to_configs` (dans `scripts/calibrate.py`) le
  fait après `lerobot-calibrate`. Si on écrit directement au servo (comme
  `fix_wrist_roll_calibration.py`), penser à mettre à jour le JSON.
- **Les anciennes captures extrinsèques (avant 2026-05-15) sont incompatibles**
  avec la calibration moteur actuelle (le `homing_offset` de `wrist_roll` a
  changé). Reconstituer des poses antérieures donnerait un mauvais repère.

## 6. Structure du repo

```
tfe-lerobot-so101/
  configs/
    so101.yaml                                  config robot high-level
    so101_new_calib.urdf                        modèle cinématique (URDF)
    calibration_cam_{0,1,2}.json                intrinsèques
    calibration_{leader,follower}.json          calibration moteurs
    extrinsic_capture_cam_{0,1,2}.json          captures hand-eye brutes
    handeye_cam_{0,1,2}.json                    résultats hand-eye
    perception/
      hsv_specs.json                            plages HSV (généré par calibrate_hsv.py)
  src/
    utils/transforms.py                         SE(3) helpers
    calibration/
      motor_to_angle.py                         encoder → radians (wraparound-aware)
      forward_kinematics.py                     FK SO-101 depuis URDF
      handeye.py                                solve {eye_to_hand,eye_in_hand}{,_robust}
    perception/
      scene.py                                  Frame, Detection2D, ObjectInstance, Scene
      robot_state.py                            lecture moteurs + FK (live/raw/angles)
      camera_io.py                              MultiCamera (live) + ReplayCamera
      detector.py                               ABC ObjectDetector + HSVDetector + HFDetector (stub)
      pose_estimator.py                         triangulation stéréo + PnP mono fallback
  scripts/
    config.py                                   ports USB + caméras
    calibrate.py                                wrapper lerobot-calibrate
    calibrate_intrinsic.py                      calibration intrinsèque
    calibrate_extrinsic.py                      captures pour hand-eye
    measure_wrist_roll.py                       one-shot : centre de wrist_roll
    fix_wrist_roll_calibration.py               one-shot : écrit Homing_Offset
    verify_wrist_roll.py                        live : vérifie wrist_roll
    detect_cameras.py                           liste les caméras
    preview_camera.py                           prévisualisation caméra
    generate_chessboard.py                      damier PNG imprimable
    check_motor_calibration.py                  valide la calibration moteur
    check_extrinsic_capture.py                  valide une capture extrinsèque
    check_calibration.py                        validation globale (+ modules perception)
    solve_handeye_cam.py                        résout hand-eye d'une caméra
    calibrate_hsv.py                            échantillonne couleurs → hsv_specs.json
    record_perception_frames.py                 enregistre dataset 3 cams + moteurs
    run_perception.py                           pipeline complet (live / replay / oneshot)
    check_perception.py                         validation chiffrée (pied à coulisse)
    teleoperate.py, record_dataset.py, train.py wrappers LeRobot CLI
  tests/
    perception/test_pipeline.py                 tests d'intégration synthétiques
  docs/
    bachelor_chanier_25_26.pdf                  cahier des charges (Partie I)
    PROJECT_STATUS.md                           CE FICHIER
    references/tfe_zotero.bib/                  bibliographie Zotero
  lerobot/                                      (gitignored) clone éditable de LeRobot
  venv/                                         (gitignored)
  outputs/, data/                               (gitignored) artefacts non-tracés
```

Chaque fichier source a une docstring expliquant son rôle.

## 7. Quick-start

Pour reprendre la main :

```bash
source venv/bin/activate
python scripts/check_calibration.py
```

Doit afficher `[OK]` partout et conclure par
`Toute la chaine de calibration est validee.`

Si quelque chose est rouge :

- **Intrinsèque dégradée** d'une caméra → `python scripts/calibrate_intrinsic.py --index N` (15-20 captures du damier sous différents angles).
- **Calibration moteur incohérente** → `python scripts/calibrate.py --follower`. Si `wrist_roll` ressort avec span > 345°, lance ensuite `measure_wrist_roll.py` + `fix_wrist_roll_calibration.py`.
- **Extrinsèque dégradée** d'une caméra :
  ```bash
  python scripts/calibrate_extrinsic.py --index N --cols 9 --rows 6 --square-size 22.0
  python scripts/solve_handeye_cam.py --index N
  ```

Pour la prochaine étape, voir [§4 Roadmap](#4-roadmap-forward).

## 8. Bibliographie utile

`docs/references/tfe_zotero.bib/` contient le `.bib` Zotero du TFE. Les PDF
des articles sont gitignorés (taille + copyright). Références centrales :

- **Hand-eye** : Tsai & Lenz 1989 ; Horaud & Dornaika 1995 ; Li et al. 2025.
- **Multiple view geometry / triangulation** : Hartley & Zisserman 2018, ch. 12 (DLT, triangulation linéaire). Utilisé pour `pose_estimator.triangulate_stereo`.
- **PnP monoculaire** : Lepetit et al. 2009, *EPnP*. Mode fallback de `pose_estimator`.
- **Détection couleur classique** : Forsyth & Ponce, *Computer Vision: A Modern Approach*, ch. 6 (color-based segmentation). Référence pour `HSVDetector` (V1).
- **Asservissement visuel** : Chaumette & Hutchinson 2006/2007 ; Flandin et al. 2000.
- **Grasp planning** : Bohg et al. 2014 (survey). Inspire la structure `ObjectInstance` / `Scene`.
- **Planification trajectoires** : LaValle 2006.
- **Active vision / extensions ML** : Diffusion Policy (Chi et al.), SmolVLA (Shukor et al.), ACT. Cités pour justifier la V2 (`HFDetector` open-vocabulary, cohérent avec l'écosystème LeRobot/HF).
