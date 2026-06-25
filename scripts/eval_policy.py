#!/usr/bin/env python3
"""
eval_policy.py – Evaluer une politique entrainee sur le VRAI robot (LeRobot)

Usage (defaut : dernier checkpoint du job ACT sur l'objet orange) :
    python scripts/eval_policy.py

    ou en personnalisant :
    python scripts/eval_policy.py \
        --policy-path outputs/train/act_so101_orange_cube/checkpoints/last/pretrained_model \
        --episodes 10

C'est la MEME commande que l'enregistrement, mais SANS leader : c'est le reseau
qui genere les actions (--policy.path). On enregistre N episodes d'evaluation
pour mesurer le taux de succes.

ATTENTION COHERENCE : le decor (positions cameras, eclairage, calibration, objet)
DOIT etre identique a celui de l'enregistrement des demos, sinon la policy
echoue (hors distribution). Memes cles cameras "front"/"wrist" -> garanties
par config.il_cameras_flag().
"""

import argparse
import subprocess
import sys

from config import (
    FOLLOWER_ID,
    IL_POLICY_TYPE,
    IL_REPO_ID,
    IL_TASK,
    il_cameras_flag,
    pick_ports,
)


def main():
    dataset_slug = IL_REPO_ID.split("/")[-1]
    default_ckpt = f"outputs/train/{IL_POLICY_TYPE}_{dataset_slug}/checkpoints/last/pretrained_model"

    parser = argparse.ArgumentParser(description="Evaluer une politique LeRobot sur le robot")
    parser.add_argument("--policy-path", type=str, default=default_ckpt,
                        help=f"Chemin du checkpoint (defaut: {default_ckpt})")
    parser.add_argument("--task", type=str, default=IL_TASK,
                        help=f"Instruction (defaut: '{IL_TASK}')")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Nombre d'episodes d'evaluation (defaut: 10)")
    parser.add_argument("--repo-id", type=str, default=f"{IL_REPO_ID.split('/')[0]}/eval_{dataset_slug}",
                        help="Repo id du dataset d'eval (prefixe 'eval_')")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Envoyer le dataset d'eval sur le Hub (defaut: local)")
    parser.add_argument("--display", action="store_true",
                        help="Afficher les cameras via rerun (utile pour debug, mais ralentit la boucle de controle)")
    args = parser.parse_args()

    follower_port, _ = pick_ports()
    if not follower_port:
        print("Port USB du follower introuvable.")
        print("  Liste : ls /dev/tty.usbmodem*   ou   lerobot-find-port")
        sys.exit(1)

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        il_cameras_flag(),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--display_data={'true' if args.display else 'false'}",
        f"--policy.path={args.policy_path}",
    ]

    print(f"Evaluation de la policy : {args.policy_path}")
    print(f"  Episodes: {args.episodes}   Eval dataset: {args.repo_id}")
    print(f"  Cameras:  {il_cameras_flag()}")
    print("  (pas de leader : c'est le reseau qui pilote)")
    print("  Clavier : -> episode suivant | Echap = stop\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nEvaluation arretee.")
    except FileNotFoundError:
        print("Commande 'lerobot-record' non trouvee.")
        print("  Verifie que le venv est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
