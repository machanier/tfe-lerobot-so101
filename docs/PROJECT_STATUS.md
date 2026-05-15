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

## 2. État au 2026-05-15

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

## 4. Roadmap forward

| Sprint | Contenu | État | Livrables principaux |
|---|---|---|---|
| 1 | Calibration hand-eye | **✅ FAIT** | `configs/handeye_cam_*.json`, `scripts/check_calibration.py` |
| 2 | Perception : détection + triangulation 3D + scène | À FAIRE | `src/perception/{camera_io,detector,pose_estimator,scene}.py` |
| 3 | Planification + contrôle : grasp + IK + interface LeRobot Python | À faire | `src/planning/`, `src/control/`, `src/pipeline.py` |
| 4 | Boucle fermée : replanification + active vision minimale | À faire | enrichissement `src/pipeline.py` |
| 5 | Évaluation expérimentale + rédaction | À faire | rapport TFE + campagne d'expériences avec les 4 métriques |

### Sprint 2 — détail

Cible : pipeline minimal `objet visible → position 3D dans le repère base
du robot`.

Fichiers à créer (signatures fixes, corps à remplir) :

- `src/perception/camera_io.py` : `MultiCamera.grab()` → dict `{cam_key:
  Frame}` avec `Frame = {image, K, dist, T_base_cam, timestamp}`. Synchronise
  les 3 caméras.
- `src/perception/detector.py` : `ObjectDetector.detect(frames, specs)` →
  liste de `Detection`. V1 : seuillage HSV + contours OpenCV.
- `src/perception/pose_estimator.py` : `triangulate(det_cam_0, det_cam_1)` →
  position 3D dans le repère base. Triangulation stéréo classique avec les
  `T_base_cam` connues (Hartley-Zisserman ch. 12).
- `src/perception/scene.py` : `@dataclass Scene` = liste d'objets + obstacles
  exprimés dans le repère base, prête pour la planification.

Validation Sprint 2 : un cube de couleur posé à une position mesurée au pied
à coulisse → `triangulate` retombe dessus à ~10 mm près (cohérent avec le
plancher de bruit calibration).

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
  src/
    utils/transforms.py                         SE(3) helpers
    calibration/
      motor_to_angle.py                         encoder → radians (wraparound-aware)
      forward_kinematics.py                     FK SO-101 depuis URDF
      handeye.py                                solve {eye_to_hand,eye_in_hand}{,_robust}
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
    check_calibration.py                        validation globale
    solve_handeye_cam.py                        résout hand-eye d'une caméra
    teleoperate.py, record_dataset.py, train.py wrappers LeRobot CLI
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
- **Multiple view geometry / triangulation** : Hartley & Zisserman 2018.
- **Asservissement visuel** : Chaumette & Hutchinson 2006/2007 ; Flandin et al. 2000.
- **Grasp planning** : Bohg et al. 2014 (survey).
- **Planification trajectoires** : LaValle 2006.
- **Active vision / extensions ML** : Diffusion Policy (Chi et al.), SmolVLA (Shukor et al.), ACT, papers active-vision divers.
