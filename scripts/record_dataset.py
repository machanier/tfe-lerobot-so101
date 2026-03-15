#!/usr/bin/env python3
"""
record_dataset.py – Enregistrer un dataset via téléopération

Usage:
    python scripts/record_dataset.py --task "pick_and_place" --episodes 50

Ce script enregistre les actions du leader et les états du follower
pour créer un dataset utilisable pour l'imitation learning.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Enregistrer un dataset de téléopération")
    parser.add_argument("--task", type=str, required=True, help="Nom de la tâche (ex: pick_and_place)")
    parser.add_argument("--episodes", type=int, default=50, help="Nombre d'épisodes à enregistrer")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo id (ex: maxence/so101_pick)")
    args = parser.parse_args()

    repo_id = args.repo_id or f"maxence/{args.task}"

    # Utilise la CLI LeRobot pour l'enregistrement
    cmd = [
        sys.executable, "-m", "lerobot.scripts.control_robot",
        "--robot.type=so101",
        "--control.type=record",
        f"--control.repo_id={repo_id}",
        f"--control.num_episodes={args.episodes}",
        f"--control.single_task={args.task}",
    ]

    print(f"🎬 Enregistrement de {args.episodes} épisodes pour la tâche '{args.task}'")
    print(f"   Repo: {repo_id}")
    print(f"   Commande: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
