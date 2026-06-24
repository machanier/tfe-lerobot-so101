# TFE — Saisie d'objets assistée par vision avec SO-101

**Travail de fin d'études — Bachelor informatique — Université de Genève (2025-2026)**

| | |
|---|---|
| **Étudiant** | Maxence Chanier |
| **Encadrant** | Guido Bologna |
| **Robot** | SO-101 (Feetech STS3215, 6 DOF × 2 bras) |
| **Caméras** | 3 USB 1920×1080 — `cam_0`/`cam_1` stéréo eye-to-hand + `cam_2` eye-in-hand |
| **Machine** | MacBook Pro M4, 24 GB RAM, macOS |

## Objectif

Concevoir, implémenter et évaluer une architecture **perception ↔ planification
↔ contrôle** pour le SO-101, capable de saisir un objet dans un environnement
encombré : sélection dynamique de point de vue (active vision), évitement
d'obstacles, replanification en boucle fermée. V1 par règles heuristiques /
géométriques ; extensions possibles par imitation learning (Diffusion Policy,
ACT, SmolVLA — voir biblio).

Cahier des charges complet (Partie I, acquise le 21.01.2026) :
[`docs/bachelor_chanier_25_26.pdf`](docs/bachelor_chanier_25_26.pdf).

## Documentation principale

**[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)** — document vivant : état
actuel détaillé, décisions techniques importantes (avec leurs raisons),
pièges connus, feuille de route. C'est le premier fichier à lire pour
reprendre le projet.

## État d'avancement

- [x] Hardware assemblé (SO-101 + 3 caméras + structure 3D imprimée + boîte de dépose)
- [x] Téléopération opérationnelle (`lerobot-teleoperate`)
- [x] Calibration intrinsèque des 3 caméras (reproj. 0.14–0.20 px)
- [x] Calibration moteur du follower (avec recalage spécifique de `wrist_roll`)
- [x] Calibration hand-eye des 3 caméras (résidus 2.5–7 mm moyens) — voir [PROJECT_STATUS](docs/PROJECT_STATUS.md#2-état-au-2026-05-15)
- [x] **Sprint 2 — Perception (code)** : 5 modules `src/perception/` + 4 scripts CLI, tous auto-testés (triangulation sub-mm sur synthétique)
- [ ] Sprint 2 — Validation expérimentale (calibrer HSV sous éclairage + mesurer erreur 3D pied à coulisse)
- [x] **Sprint 3** : grasp planning (`src/planning/grasp.py`) + trajectoire (`src/control/`) + interface LeRobot (`pick_and_place.py`)
- [x] **Sprint 4** : raffinement en boucle fermée par `cam_2` eye-in-hand (`src/control/closed_loop.py`)
- [~] **Sprint 5** : évaluation expérimentale (en cours) + rédaction du mémoire (`docs/memoire/`)
  - Saisie **fiable en objet seul** (2026-06-23) : `orange_cube` et `purple_cylinder` (debout, couché //X, couché //Y) saisis et déposés, souvent du 1ᵉʳ coup. Repose sur deux offsets de prise (cf section *Saisie*) qui compensent le décalage entre le repère outil et le point de préhension. Reste : scènes encombrées / occlusion (objectif du cahier des charges encore ouvert), et robustesse perception sous éclairage variable.

## Structure du projet

```
tfe-lerobot-so101/
├── configs/                        # Calibrations + modèle robot (versionnés)
│   ├── so101.yaml, so101_new_calib.urdf
│   ├── calibration_cam_{0,1,2}.json          # Intrinsèques
│   ├── calibration_{leader,follower}.json    # Calibration moteurs
│   ├── extrinsic_capture_cam_{0,1,2}.json    # Captures hand-eye brutes
│   ├── handeye_cam_{0,1,2}.json              # Résultats hand-eye
│   └── perception/hsv_specs.json             # Plages HSV (généré)
├── src/
│   ├── utils/transforms.py         # SE(3) helpers
│   ├── calibration/                # Hand-eye + FK + motor_to_angle
│   ├── perception/                 # Sprint 2
│   │   ├── scene.py                # Dataclasses Frame, Detection2D, ObjectInstance, Scene
│   │   ├── robot_state.py          # Lecture moteurs + FK
│   │   ├── camera_io.py            # MultiCamera (live) + ReplayCamera (offline)
│   │   ├── detector.py             # ABC + HSVDetector (V1) + HFDetector (stub V2)
│   │   └── pose_estimator.py       # Triangulation stéréo + PnP mono fallback
│   ├── planning/grasp.py           # Sprint 3 : TopDownGrasp + AdaptiveGrasp (0/45/90 par zone)
│   ├── control/                    # Sprint 3-4
│   │   ├── ik.py, trajectory.py, motor_controller.py  # IK 5-DOF + traj quintiques + envoi moteur
│   │   └── closed_loop.py          # Sprint 4 : raffinement cam_2 eye-in-hand
│   └── pipeline.py                 # Orchestration perception -> planning -> control
├── scripts/
│   ├── (calibration: calibrate*.py, solve_handeye_cam.py, check_calibration.py …)
│   ├── calibrate_hsv.py            # Échantillonne couleurs → hsv_specs.json
│   ├── record_perception_frames.py # Enregistre dataset 3 cams + moteurs (replay)
│   ├── run_perception.py           # Pipeline perception (live / replay / oneshot)
│   ├── check_perception.py         # Validation chiffrée (ground truth pied à coulisse)
│   └── pick_and_place.py           # Pick-and-place complet (Sprint 3-4)
├── tests/
│   ├── perception/test_pipeline.py # Tests d'intégration synthétiques
│   └── planning/test_adaptive_grasp_selection.py  # Sélection d'angle + reorient cam_2
├── docs/PROJECT_STATUS.md          # Document vivant
└── requirements.txt, setup_env.sh
```

## Installation

```bash
git clone https://github.com/machanier/tfe-lerobot-so101.git
cd tfe-lerobot-so101
./setup_env.sh
source venv/bin/activate
```

## Quick-start

```bash
# Activer le venv
source venv/bin/activate

# Vérifier que toute la calibration + modules perception passent
python scripts/check_calibration.py

# Téléopérer le robot
python scripts/teleoperate.py

# Prévisualiser une caméra (cam_0 par défaut, --camera N pour les autres)
python scripts/preview_camera.py

# Sprint 2 : pipeline perception multi-caméras
python scripts/calibrate_hsv.py             # une fois, sous l'éclairage final
python scripts/run_perception.py            # boucle live des 3 caméras
python scripts/check_perception.py --gt outputs/perception/gt_test.json   # validation chiffrée

# Sprint 3-4 : pick-and-place complet (perception -> saisie -> dépose)
python scripts/pick_and_place.py --target orange_cube
```

Les **snapshots de diagnostic** (vues caméras + vues `cam_2` au moment de la prise)
sont sauvés dans `outputs/perception/` **par défaut, sans `--display`**. Réserver
`--display` à la fenêtre temps réel : elle capture les 3 caméras en continu pendant
le mouvement et peut faire **décrocher `cam_2`** sur le hub USB. `--no-snapshots`
coupe la sauvegarde.

## Saisie (pick-and-place)

Chaîne `src/pipeline.py` :

1. **Perception** — détection HSV (ou HF) dans `cam_0`/`cam_1`, triangulation
   stéréo de la position 3D (repère base, **z=0 = la plaque**) ; la hauteur vient
   d'une triangulation du sommet. Un biais de calibration est soustrait
   (`configs/perception/bias_correction.json`).
2. **Planning** (`src/planning/grasp.py`) — `AdaptiveGrasp` propose plusieurs
   angles d'attaque — **top-down (0°)**, **diagonale (45°)**, **face avant (90°)** —
   dans l'ordre de préférence selon la **zone** (distance + hauteur ;
   `GRASP_ZONE_*`), et garde le 1ᵉʳ angle **atteignable** par l'IK. Pour un objet
   allongé, l'approche s'aligne sur le grand axe → mâchoires en travers du petit côté.
3. **Raffinement `cam_2`** (`src/control/closed_loop.py`) — à ~8 cm au-dessus de
   l'objet, la caméra eye-in-hand recale la **position** (résiduel stéréo) et
   réaligne les **mâchoires** sur le grand axe, sous garde-fous (taille du blob,
   plafond de correction).
4. **Offsets de prise** (appliqués après `cam_2`, en repère image `cam_2`) — ils
   compensent le fait que le repère outil commandé (`gripper_frame_link`, sur l'axe
   du poignet) **n'est pas** le point où les mâchoires serrent (le mécanisme est
   monté ~3 cm à côté de cet axe). Deux composantes horizontales :
   - **latéral** (`--grasp-lateral-offset`, par défaut **adaptatif = ½ largeur +
     marge**) : amène le doigt FIXE à fleur de l'arête, quelle que soit la taille ;
   - **profondeur** (`--grasp-forward-offset`, défaut 15 mm, constant) : avance la
     prise vers le point de fermeture réel des doigts.

   La composante verticale (le long de la pince) est, elle, gérée par la hauteur de
   descente — pas d'offset nécessaire.
5. **Saisie** — descente + **fermeture asservie au couple**, vérification après
   levée, **retry** si fermeture à vide, puis dépose dans la boîte.

Pour les procédures détaillées (recalibrer, déboguer un capteur, etc.) et les
prochaines étapes, voir [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot) — la stack
  Hugging Face pour téléopération / datasets / politiques.
- [Tutoriel SO-101 LeRobot](https://huggingface.co/docs/lerobot/so101)
- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100) — repo
  hardware officiel, source de l'URDF.
- Bibliographie académique : [`docs/references/tfe_zotero.bib/`](docs/references/).
