# TFE – Robot SO-101 avec LeRobot 🤖

**Travail de fin d'études – Bachelor – Université de Genève (2025-2026)**  
**Auteur :** Maxence Chanier

---

## 📌 Description

Ce projet vise à explorer et étendre les capacités du robot **SO-101** (bras robotique open-source) en utilisant la bibliothèque **[LeRobot](https://github.com/huggingface/lerobot)** de Hugging Face.

**État actuel :** Le robot est assemblé, calibré et capable de téléopération (leader → follower).

**Objectifs :**
- Enregistrer des démonstrations via téléopération
- Entraîner des politiques d'imitation learning (ACT, Diffusion Policy, etc.)
- Évaluer les performances du robot sur des tâches de manipulation
- (Optionnel) Ajouter une caméra pour la politique visuo-motrice

## 🛠️ Hardware

| Composant | Détail |
|-----------|--------|
| Robot | SO-101 (follower + leader arm) |
| Moteurs | STS3215 (Feetech) |
| Machine | MacBook Pro M4, 24GB RAM, 1TB SSD |
| OS | macOS |

## 📁 Structure du projet

```
tfe-lerobot-so101/
├── configs/           # Configurations robot (calibration, ports...)
├── scripts/           # Scripts principaux (téléop, enregistrement, entraînement)
├── notebooks/         # Jupyter notebooks (exploration, visualisation)
├── data/              # Datasets enregistrés (gitignored – lourd)
├── outputs/           # Modèles entraînés, logs (gitignored – lourd)
├── docs/              # Documentation, rapport TFE
├── tests/             # Tests unitaires
├── requirements.txt   # Dépendances Python
└── setup_env.sh       # Script de setup de l'environnement
```

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/<ton-username>/tfe-lerobot-so101.git
cd tfe-lerobot-so101

# 2. Créer et activer l'environnement virtuel
./setup_env.sh

# 3. Activer le venv
source venv/bin/activate
```

## 🎮 Utilisation rapide

```bash
# Téléopération
python scripts/teleoperate.py

# Enregistrer un dataset
python scripts/record_dataset.py

# Entraîner un modèle
python scripts/train.py
```

## 📚 Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [Tutoriel SO-101](https://huggingface.co/docs/lerobot/so101)
- [Getting Started with Real-World Robots](https://huggingface.co/docs/lerobot/il_robots)

## 📄 Licence

Ce projet est réalisé dans un cadre académique (TFE). Le code s'appuie sur LeRobot (Apache 2.0).
