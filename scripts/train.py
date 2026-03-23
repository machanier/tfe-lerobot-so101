#!/usr/bin/env python3
"""
train.py – Entrainer une politique d'imitation learning

Usage:
    python scripts/train.py --policy act --dataset maxence/so101_pick

Politiques disponibles dans LeRobot :
    - act       : Action Chunking with Transformers
    - diffusion : Diffusion Policy
    - tdmpc     : TD-MPC
    - vqbet     : VQ-BeT
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Entrainer une politique LeRobot")
    parser.add_argument(
        "--policy", type=str, default="act", help="Type de politique (act, diffusion, ...)"
    )
    parser.add_argument("--dataset", type=str, required=True, help="ID du dataset HuggingFace")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Dossier de sortie")
    args = parser.parse_args()

    cmd = [
        "lerobot-train",
        f"--policy.type={args.policy}",
        f"--dataset.repo_id={args.dataset}",
        f"--training.num_epochs={args.epochs}",
        f"--output_dir={args.output_dir}",
    ]

    print(f"Entrainement de la politique '{args.policy}'")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    print(f"  Commande: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nEntrainement arrete.")
    except FileNotFoundError:
        print("Commande 'lerobot-train' non trouvee.")
        print("  Verifie que le venv est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
