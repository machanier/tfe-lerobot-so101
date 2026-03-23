#!/usr/bin/env python3
"""
record_dataset.py – Enregistrer un dataset via teleoperation

Usage:
    python scripts/record_dataset.py --task "pick_and_place" --episodes 50

Ce script enregistre les actions du leader et les etats du follower
pour creer un dataset utilisable pour l'imitation learning.
"""

import argparse
import subprocess
import sys

from config import FOLLOWER_ID, FOLLOWER_PORT, LEADER_ID, LEADER_PORT


def main():
    parser = argparse.ArgumentParser(description="Enregistrer un dataset de teleoperation")
    parser.add_argument("--task", type=str, required=True, help="Nom de la tache (ex: pick_and_place)")
    parser.add_argument("--episodes", type=int, default=50, help="Nombre d'episodes a enregistrer")
    parser.add_argument(
        "--repo-id", type=str, default=None, help="HuggingFace repo id (ex: maxence/so101_pick)"
    )
    args = parser.parse_args()

    repo_id = args.repo_id or f"maxence/{args.task}"

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",
        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_ID}",
        f"--repo-id={repo_id}",
        f"--num-episodes={args.episodes}",
        f"--single-task={args.task}",
    ]

    print(f"Enregistrement de {args.episodes} episodes pour la tache '{args.task}'")
    print(f"  Repo: {repo_id}")
    print(f"  Commande: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nEnregistrement arrete.")
    except FileNotFoundError:
        print("Commande 'lerobot-record' non trouvee.")
        print("  Verifie que le venv est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
