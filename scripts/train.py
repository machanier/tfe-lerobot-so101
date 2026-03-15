#!/usr/bin/env python3
"""
train.py – Entraîner une politique d'imitation learning

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
    parser = argparse.ArgumentParser(description="Entraîner une politique LeRobot")
    parser.add_argument("--policy", type=str, default="act", help="Type de politique (act, diffusion, ...)")
    parser.add_argument("--dataset", type=str, required=True, help="ID du dataset HuggingFace")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Dossier de sortie")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--policy.type={args.policy}",
        f"--dataset.repo_id={args.dataset}",
        f"--training.num_epochs={args.epochs}",
        f"--output_dir={args.output_dir}",
    ]

    print(f"🧠 Entraînement de la politique '{args.policy}'")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Output: {args.output_dir}")
    print(f"   Commande: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
