#!/usr/bin/env python3
"""
teleoperate.py – Téléopération du SO-101 (leader → follower)

Usage:
    python scripts/teleoperate.py

    Arrêter avec Ctrl+C.

Note:
    Les ports USB changent à chaque branchement sur macOS !
    Si ça ne marche pas, modifie scripts/config.py
    ou lance : ls /dev/tty.usbmodem*
"""

import glob
import subprocess
import sys

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    LEADER_ID,
    LEADER_PORT,
)


def pick_ports():
    """Choisit automatiquement les ports si ceux configurés ne sont pas tous présents."""
    ports_detectes = glob.glob("/dev/tty.usbmodem*")

    if not ports_detectes:
        print("Aucun port USB detecte !")
        print("  Le robot est-il branche en USB ET alimente sur secteur ?")
        return None, None

    follower = FOLLOWER_PORT if FOLLOWER_PORT in ports_detectes else None
    leader = LEADER_PORT if LEADER_PORT in ports_detectes else None

    if follower and leader:
        return follower, leader

    if len(ports_detectes) >= 2:
        follower, leader = ports_detectes[:2]
        print(f"Ports auto-selectionnes : {follower}, {leader}")
        return follower, leader

    if len(ports_detectes) == 1:
        print(f"Un seul port detecte : {ports_detectes[0]}")
        print("  Branche aussi le leader ou corrige scripts/config.py")
        return ports_detectes[0], None

    return follower, leader


def main():
    follower_port, leader_port = pick_ports()

    if not follower_port or not leader_port:
        sys.exit(1)

    cmd = [
        "lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        f'--robot.cameras={{"front": {{"type": "opencv", "index_or_path": {CAMERA_INDEX}, "width": {CAMERA_WIDTH}, "height": {CAMERA_HEIGHT}, "fps": {CAMERA_FPS}}}}}',
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={LEADER_ID}",
        "--display_data=true",
    ]

    print(f"Lancement de la teleoperation SO-101...")
    print(f"  Follower: {follower_port}")
    print(f"  Leader:   {leader_port}")
    print(f"  Camera front: index {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS})")
    print(f"  Arreter avec Ctrl+C\n")

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
