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
- [ ] **Sprint 3** : grasp planning + planification trajectoire + interface LeRobot Python
- [ ] **Sprint 4** : replanification boucle fermée + active vision
- [ ] **Sprint 5** : évaluation expérimentale + rédaction du mémoire

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
│   └── perception/                 # Sprint 2 (NOUVEAU)
│       ├── scene.py                # Dataclasses Frame, Detection2D, ObjectInstance, Scene
│       ├── robot_state.py          # Lecture moteurs + FK
│       ├── camera_io.py            # MultiCamera (live) + ReplayCamera (offline)
│       ├── detector.py             # ABC + HSVDetector (V1) + HFDetector (stub V2)
│       └── pose_estimator.py       # Triangulation stéréo + PnP mono fallback
├── scripts/
│   ├── (calibration: calibrate*.py, solve_handeye_cam.py, check_calibration.py …)
│   ├── calibrate_hsv.py            # Échantillonne couleurs → hsv_specs.json
│   ├── record_perception_frames.py # Enregistre dataset 3 cams + moteurs (replay)
│   ├── run_perception.py           # Pipeline complet (live / replay / oneshot)
│   └── check_perception.py         # Validation chiffrée (ground truth pied à coulisse)
├── tests/
│   └── perception/test_pipeline.py # Tests d'intégration synthétiques (4 cas)
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
```

Pour les procédures détaillées (recalibrer, déboguer un capteur, etc.) et les
prochaines étapes, voir [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot) — la stack
  Hugging Face pour téléopération / datasets / politiques.
- [Tutoriel SO-101 LeRobot](https://huggingface.co/docs/lerobot/so101)
- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100) — repo
  hardware officiel, source de l'URDF.
- Bibliographie académique : [`docs/references/tfe_zotero.bib/`](docs/references/).
