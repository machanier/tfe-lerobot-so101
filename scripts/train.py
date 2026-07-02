#!/usr/bin/env python3
"""
Entrainement d'une politique d'imitation learning avec LeRobot.

Usage avec les valeurs par defaut issues de config.py (ACT, dataset orange, MPS) :
    python scripts/train.py

Personnalisation des principaux parametres :
    python scripts/train.py --policy act --dataset maxence/so101_orange_cube --steps 100000

Politiques LeRobot disponibles : act (defaut), diffusion, tdmpc, vqbet, etc.

ACT se compte en steps (et non en epochs) ; le defaut LeRobot est de 100 000
steps, soit quelques heures sur un GPU NVIDIA. Sur MPS (Apple Silicon)
l'entrainement est plus lent, et un operateur non supporte peut necessiter
PYTORCH_ENABLE_MPS_FALLBACK=1 ; le notebook Colab officiel ACT constitue alors
une alternative. Repli local possible avec --device cpu.

Le modele entraine reste local (checkpoints dans outputs/). L'option
--push-to-hub permet de le publier sur le Hub (necessite `hf auth login`).
"""

import argparse
import os
import subprocess
import sys

from config import (
    IL_BATCH_SIZE,
    IL_POLICY_DEVICE,
    IL_POLICY_TYPE,
    IL_REPO_ID,
    IL_STEPS,
)


def main():
    parser = argparse.ArgumentParser(description="Entrainer une politique LeRobot (IL/ACT)")
    parser.add_argument("--policy", type=str, default=IL_POLICY_TYPE,
                        help=f"Type de politique (defaut: {IL_POLICY_TYPE})")
    parser.add_argument("--dataset", type=str, default=IL_REPO_ID,
                        help=f"ID du dataset (defaut: {IL_REPO_ID})")
    parser.add_argument("--steps", type=int, default=IL_STEPS,
                        help=f"Nombre de steps (defaut: {IL_STEPS})")
    parser.add_argument("--batch-size", type=int, default=IL_BATCH_SIZE,
                        help=f"Taille de batch (defaut: {IL_BATCH_SIZE})")
    parser.add_argument("--device", type=str, default=IL_POLICY_DEVICE,
                        help=f"mps | cpu | cuda (defaut: {IL_POLICY_DEVICE})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dossier de sortie (defaut: outputs/train/<job>)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Envoyer le modele sur le Hub (defaut: local seulement)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre l'entrainement depuis le dernier checkpoint et poursuivre jusqu'a --steps (cible totale, non cumulative). Exemple : un entrainement arrete a 30000 relance avec --resume --steps 50000 effectue 20000 steps supplementaires.")
    args = parser.parse_args()

    # Nom de job lisible derive du dataset (ex: act_so101_orange_cube)
    dataset_slug = args.dataset.split("/")[-1]
    job_name = f"{args.policy}_{dataset_slug}"
    output_dir = args.output_dir or f"outputs/train/{job_name}"

    if args.resume:
        # Reprise : toute la configuration est relue depuis le dernier
        # checkpoint ; seul --steps est resurcharge (cible totale a atteindre).
        ckpt_config = f"{output_dir}/checkpoints/last/pretrained_model/train_config.json"
        if not os.path.exists(ckpt_config):
            print(f"Checkpoint introuvable pour reprendre : {ckpt_config}")
            print("  (aucun entrainement n'a encore ete lance dans ce dossier ?)")
            sys.exit(1)
        cmd = [
            "lerobot-train",
            f"--config_path={ckpt_config}",
            "--resume=true",
            f"--steps={args.steps}",
        ]
        print(f"Reprise de l'entrainement -> cible totale : {args.steps} steps")
        print(f"  Checkpoint: {ckpt_config}")
    else:
        cmd = [
            "lerobot-train",
            f"--dataset.repo_id={args.dataset}",
            f"--policy.type={args.policy}",
            f"--policy.device={args.device}",
            f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}",
            f"--batch_size={args.batch_size}",
            f"--steps={args.steps}",
            f"--output_dir={output_dir}",
            f"--job_name={job_name}",
        ]
        print(f"Entrainement '{args.policy}' sur '{args.dataset}'")
        print(f"  Steps: {args.steps}   Batch: {args.batch_size}   Device: {args.device}")
        print(f"  Output: {output_dir}")
    print(f"  Commande: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nEntrainement arrete.")
    except FileNotFoundError:
        print("Commande 'lerobot-train' introuvable.")
        print("  Verifier que l'environnement virtuel est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
