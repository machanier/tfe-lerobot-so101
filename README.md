# TFE – Saisie d'objets assistee par vision avec SO-101

**Travail de fin d'etudes – Bachelor en informatique – Universite de Geneve (2025-2026)**

| | |
|---|---|
| **Auteur** | Maxence Chanier |
| **Encadrant** | Guido Bologna |
| **Robot** | SO-101 (Feetech STS3215, 6 DOF x 2 bras) |
| **Camera** | USB eye-to-hand (1920x1080 @ 30fps) |
| **Machine** | MacBook Pro M4, 24GB RAM, macOS |

## Objectif

Concevoir une architecture perception-planification-controle pour un bras robotique SO-101 equipe de cameras, capable de saisir des objets dans un environnement encombre. Le projet utilise [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face) pour la teleoperation et l'imitation learning.

## Etat d'avancement

- [x] Robot assemble et calibre (leader + follower)
- [x] Teleoperation fonctionnelle
- [x] Camera eye-to-hand installee et mise au point
- [x] Calibration intrinseque (erreur: 0.31px)
- [x] Calibration extrinseque (20 poses)
- [ ] Enregistrement de demonstrations (pick and place)
- [ ] Entrainement d'une politique ACT
- [ ] Evaluation en autonomie

## Structure du projet

```
tfe-lerobot-so101/
├── scripts/                        # Scripts Python
│   ├── config.py                   # Configuration centralisee (ports, camera)
│   ├── teleoperate.py              # Teleoperation leader -> follower
│   ├── calibrate.py                # Calibration des bras (moteurs)
│   ├── preview_camera.py           # Apercu camera temps reel
│   ├── calibrate_intrinsic.py      # Calibration intrinseque camera
│   ├── calibrate_extrinsic.py      # Calibration extrinseque camera-robot
│   ├── record_dataset.py           # Enregistrement de demonstrations
│   └── train.py                    # Entrainement de politiques
├── configs/                        # Donnees de calibration
│   ├── so101.yaml                  # Config robot
│   ├── calibration_leader.json     # Calibration bras leader
│   ├── calibration_follower.json   # Calibration bras follower
│   ├── calibration_cam_0.json      # Calibration intrinseque camera
│   └── extrinsic_cam_0.json        # Calibration extrinseque camera
├── lerobot/                        # Clone de LeRobot (gitignored)
├── venv/                           # Environnement Python (gitignored)
├── outputs/                        # Images de calibration, modeles (gitignored)
├── data/                           # Datasets (gitignored)
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

## Utilisation

Tous les parametres hardware sont dans `scripts/config.py`.

```bash
# Teleoperation
python scripts/teleoperate.py

# Apercu camera
python scripts/preview_camera.py

# Calibration camera (intrinseque puis extrinseque)
python scripts/calibrate_intrinsic.py
python scripts/calibrate_extrinsic.py

# Enregistrer des demonstrations
python scripts/record_dataset.py --task "pick_and_place" --episodes 50

# Entrainer une politique
python scripts/train.py --policy act --dataset maxence/pick_and_place
```

## Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Tutoriel SO-101](https://huggingface.co/docs/lerobot/so101)
- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
