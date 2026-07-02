#!/usr/bin/env python3
"""
teleoperate.py — téléopération du SO-101 (le bras leader pilote le follower).

Usage :
    python scripts/teleoperate.py     # arrêter avec Ctrl+C

Sert à vérifier que le leader, le follower et les deux caméras d'imitation
(frontale + poignet) fonctionnent avant d'enregistrer un dataset. Le décor
visible ici doit être celui de l'enregistrement (mêmes caméras, même cadrage).

Note : sur macOS, les ports USB changent à chaque branchement. En cas de
problème, lancer `lerobot-find-port` puis corriger scripts/config.py.
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
        print("  Le robot est-il branché en USB et alimenté sur secteur ?")
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

    print("Lancement de la téléopération SO-101 (2 caméras).")
    print(f"  Follower : {follower_port}")
    print(f"  Leader   : {leader_port}")
    print(f"  Caméras  : {il_cameras_flag()}")
    print("  Arrêter avec Ctrl+C\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTéléopération arrêtée.")
    except FileNotFoundError:
        print("Commande 'lerobot-teleoperate' introuvable.")
        print("  Vérifier que le venv est activé : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
