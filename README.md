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
- [ ] **Sprint 2 — Perception** : détection + triangulation 3D + modèle de scène
- [ ] **Sprint 3** : grasp planning + planification trajectoire + interface LeRobot Python
- [ ] **Sprint 4** : replanification boucle fermée + active vision
- [ ] **Sprint 5** : évaluation expérimentale + rédaction du mémoire

## Structure du projet

```
tfe-lerobot-so101/
├── configs/                        # Calibrations + modèle robot (versionnés)
│   ├── so101.yaml                  # Config robot high-level
│   ├── so101_new_calib.urdf        # URDF (TheRobotStudio/SO-ARM100)
│   ├── calibration_cam_{0,1,2}.json     # Intrinsèques
│   ├── calibration_{leader,follower}.json   # Calibration moteurs
│   ├── extrinsic_capture_cam_{0,1,2}.json   # Captures hand-eye brutes
│   └── handeye_cam_{0,1,2}.json    # Résultats hand-eye (T_base_cam / T_gripper_cam)
├── src/
│   ├── utils/transforms.py         # SE(3) helpers
│   └── calibration/
│       ├── motor_to_angle.py       # Encoder → radians (wraparound-aware)
│       ├── forward_kinematics.py   # FK SO-101 depuis URDF (zéro dépendance lourde)
│       └── handeye.py              # solve_eye_to_hand{,_robust} + solve_eye_in_hand{,_robust}
├── scripts/                        # Scripts CLI
│   ├── config.py                   # Ports USB + caméras
│   ├── calibrate.py                # Wrapper lerobot-calibrate
│   ├── calibrate_intrinsic.py      # Calibration intrinsèque caméra
│   ├── calibrate_extrinsic.py      # Captures pour hand-eye
│   ├── solve_handeye_cam.py        # Résolution hand-eye d'une caméra (mode robuste)
│   ├── measure_wrist_roll.py       # One-shot : mesure du centre de wrist_roll
│   ├── fix_wrist_roll_calibration.py    # One-shot : recale le Homing_Offset wrist_roll
│   ├── verify_wrist_roll.py        # Live : vérifie la calibration wrist_roll
│   ├── check_motor_calibration.py  # Valide la calibration moteur
│   ├── check_extrinsic_capture.py  # Valide une capture extrinsèque
│   ├── check_calibration.py        # Validation globale de toute la chaîne
│   ├── generate_chessboard.py      # Génère un damier PNG imprimable
│   ├── detect_cameras.py           # Liste les caméras connectées
│   └── preview_camera.py           # Prévisualisation
├── docs/
│   ├── PROJECT_STATUS.md           # Document vivant : état du projet
│   ├── bachelor_chanier_25_26.pdf  # Cahier des charges Partie I
│   └── references/                 # Bibliographie Zotero
├── lerobot/                        # Clone éditable de LeRobot (gitignored)
├── venv/                           # Environnement Python (gitignored)
├── outputs/, data/                 # Artefacts non-tracés (gitignored)
├── requirements.txt
└── setup_env.sh
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

# Vérifier que toute la calibration est OK
python scripts/check_calibration.py

# Téléopérer le robot
python scripts/teleoperate.py

# Prévisualiser une caméra (cam_0 par défaut, --camera N pour les autres)
python scripts/preview_camera.py
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
