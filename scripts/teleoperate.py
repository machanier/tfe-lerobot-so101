#!/usr/bin/env python3
"""
teleoperate.py – Teleoperation du SO-101 (leader -> follower)

Usage:
    python scripts/teleoperate.py

    Arreter avec Ctrl+C.

But : verifier que leader, follower ET les 2 cameras IL (front + wrist)
fonctionnent, AVANT d'enregistrer un dataset. Le decor que tu vois ici doit
etre celui de l'enregistrement (memes cameras, meme cadrage).

Note:
    Les ports USB changent a chaque branchement sur macOS !
    Si ca ne marche pas : `lerobot-find-port`, puis corrige scripts/config.py.
"""

import subprocess
import sys

from config import (
    FOLLOWER_ID,
    LEADER_ID,
    il_cameras_flag,
    pick_ports,
)


def main():
    follower_port, leader_port = pick_ports()

    if not follower_port or not leader_port:
        print("Ports USB introuvables (follower et/ou leader).")
        print("  Le robot est-il branche en USB ET alimente sur secteur ?")
        print("  Liste : ls /dev/tty.usbmodem*   ou   lerobot-find-port")
        sys.exit(1)

    cmd = [
        "lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        il_cameras_flag(),
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={LEADER_ID}",
        "--display_data=true",
    ]

    print("Lancement de la teleoperation SO-101 (2 cameras IL)...")
    print(f"  Follower: {follower_port}")
    print(f"  Leader:   {leader_port}")
    print(f"  Cameras:  {il_cameras_flag()}")
    print("  Arreter avec Ctrl+C\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTeleoperation arretee.")
    except FileNotFoundError:
        print("Commande 'lerobot-teleoperate' non trouvee.")
        print("  Verifie que le venv est active : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
