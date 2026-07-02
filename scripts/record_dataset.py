#!/usr/bin/env python3
"""
Enregistrement d'un dataset de teleoperation via LeRobot.

Usage (valeurs par defaut issues de config.py) :
    python scripts/record_dataset.py

    ou en personnalisant :
    python scripts/record_dataset.py --task "Grab the orange cube" --episodes 50

Enregistre, image par image (30 fps), les actions du leader, l'etat du follower
et les deux cameras (front + wrist) pour produire un dataset LeRobot reutilisable
a l'entrainement d'ACT.

Controles clavier durant l'enregistrement :
    Fleche droite  = episode termine, passe au suivant
    Fleche gauche  = annule et re-enregistre l'episode courant
    Echap          = stop, encode les videos (et upload si --push-to-hub)

Par defaut le dataset reste local (~/.cache/huggingface/lerobot/<repo-id>).
Ajouter --push-to-hub pour l'envoyer sur le Hub (necessite `hf auth login`).
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
    parser.add_argument("--no-display", action="store_true",
                        help="Desactiver la visualisation rerun (reduit la charge en cas de time-out camera)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre ou completer un dataset --repo-id existant (par exemple apres une interruption)")
    args = parser.parse_args()

    follower_port, leader_port = pick_ports()
    if not follower_port or not leader_port:
        print("Ports USB introuvables (follower et/ou leader).")
        print("  Pour les lister : ls /dev/tty.usbmodem*   ou   lerobot-find-port")
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
        f"--display_data={'false' if args.no_display else 'true'}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.single_task={args.task}",
        f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--resume={'true' if args.resume else 'false'}",
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
        print("Commande 'lerobot-record' introuvable.")
        print("  Verifier que l'environnement virtuel est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
