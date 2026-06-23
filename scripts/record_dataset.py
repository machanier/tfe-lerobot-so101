#!/usr/bin/env python3
"""
record_dataset.py – Enregistrer un dataset via teleoperation (LeRobot)

Usage (valeurs par defaut = config.py : objet orange, 50 episodes) :
    python scripts/record_dataset.py

    ou en personnalisant :
    python scripts/record_dataset.py --task "Grab the orange cube" --episodes 50

Enregistre, par image (30 fps), les actions du leader, l'etat du follower et
les 2 cameras (front + wrist) -> dataset LeRobot reutilisable pour entrainer ACT.

Controles clavier PENDANT l'enregistrement :
    Fleche droite  = episode termine -> passe au suivant
    Fleche gauche  = annuler et re-enregistrer l'episode courant (a utiliser !)
    Echap          = stop, encode les videos, (et upload si --push-to-hub)

Par defaut le dataset reste LOCAL (~/.cache/huggingface/lerobot/<repo-id>).
Ajoute --push-to-hub pour l'envoyer sur le Hub (necessite `hf auth login`).
"""

import argparse
import subprocess
import sys

from config import (
    FOLLOWER_ID,
    IL_NUM_EPISODES,
    IL_REPO_ID,
    IL_TASK,
    LEADER_ID,
    il_cameras_flag,
    pick_ports,
)


def main():
    parser = argparse.ArgumentParser(description="Enregistrer un dataset de teleoperation (IL/ACT)")
    parser.add_argument("--task", type=str, default=IL_TASK,
                        help=f"Instruction de la tache (defaut: '{IL_TASK}')")
    parser.add_argument("--episodes", type=int, default=IL_NUM_EPISODES,
                        help=f"Nombre d'episodes (defaut: {IL_NUM_EPISODES})")
    parser.add_argument("--repo-id", type=str, default=IL_REPO_ID,
                        help=f"HuggingFace repo id (defaut: {IL_REPO_ID})")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Envoyer le dataset sur le Hub (defaut: local seulement)")
    args = parser.parse_args()

    follower_port, leader_port = pick_ports()
    if not follower_port or not leader_port:
        print("Ports USB introuvables (follower et/ou leader).")
        print("  Liste : ls /dev/tty.usbmodem*   ou   lerobot-find-port")
        sys.exit(1)

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        il_cameras_flag(),
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={LEADER_ID}",
        "--display_data=true",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.single_task={args.task}",
        f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}",
    ]

    print(f"Enregistrement de {args.episodes} episodes -- tache : '{args.task}'")
    print(f"  Repo:    {args.repo_id}  (push_to_hub={args.push_to_hub})")
    print(f"  Cameras: {il_cameras_flag()}")
    print(f"  Follower: {follower_port}   Leader: {leader_port}")
    print("  Clavier : -> suivant | <- re-enregistrer | Echap = stop\n")

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
