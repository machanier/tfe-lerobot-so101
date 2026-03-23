# TFE – Saisie d'objets assistee par vision avec SO-101

**Travail de fin d'etudes – Bachelor en informatique – Universite de Geneve (2025-2026)**
**Auteur :** Maxence Chanier | **Encadrant :** Guido Bologna

## Description

Architecture perception-planification-controle pour un bras robotique **SO-101** equipe de cameras, capable de saisir des objets dans un environnement encombre. Utilise **[LeRobot](https://github.com/huggingface/lerobot)** (Hugging Face).

## Structure du projet

```
tfe-lerobot-so101/
├── scripts/                      # Scripts Python (tout se lance depuis ici)
│   ├── config.py                 # Config centralisee (ports USB, camera)
│   ├── teleoperate.py            # Teleoperation leader -> follower
│   ├── calibrate.py              # Calibration des bras (moteurs)
│   ├── preview_camera.py         # Apercu camera en temps reel
│   ├── calibrate_intrinsic.py    # Calibration intrinseque camera
│   ├── calibrate_extrinsic.py    # Calibration extrinseque camera
│   ├── record_dataset.py         # Enregistrement de demonstrations
│   └── train.py                  # Entrainement de politiques
├── configs/                      # Fichiers de configuration
│   ├── so101.yaml                # Config robot (documentation)
│   ├── calibration_leader.json   # Calibration bras leader
│   └── calibration_follower.json # Calibration bras follower
├── lerobot/                      # Clone de LeRobot (gitignored, genere par setup_env.sh)
├── venv/                         # Environnement Python (gitignored)
├── requirements.txt
└── setup_env.sh                  # Setup automatique
```

## Installation

```bash
git clone https://github.com/machanier/tfe-lerobot-so101.git
cd tfe-lerobot-so101
./setup_env.sh
source venv/bin/activate
```

## Utilisation

```bash
# Teleoperation
python scripts/teleoperate.py

# Apercu camera
python scripts/preview_camera.py

# Calibration camera
python scripts/calibrate_intrinsic.py --generate    # Generer un damier
python scripts/calibrate_intrinsic.py --index 0     # Calibrer
python scripts/calibrate_extrinsic.py               # Position camera/robot

# Enregistrer des demonstrations
python scripts/record_dataset.py --task "pick_and_place" --episodes 50

# Entrainer une politique
python scripts/train.py --policy act --dataset maxence/pick_and_place
```

## Configuration

Tous les parametres hardware (ports USB, camera) sont dans **`scripts/config.py`**.
Pour trouver les ports : `ls /dev/tty.usbmodem*`

## Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [Tutoriel SO-101](https://huggingface.co/docs/lerobot/so101)
