#!/usr/bin/env python3
"""
calibrate.py – Recalibrer le leader et/ou le follower du SO-101

Usage:
    python scripts/calibrate.py           # Calibrer les deux (leader puis follower)
    python scripts/calibrate.py --leader   # Calibrer seulement le leader
    python scripts/calibrate.py --follower # Calibrer seulement le follower

Rappel des articulations (de la base vers la pince) :
    1. shoulder_pan   = Rotation de la base         (gauche/droite)
    2. shoulder_lift  = Lever/baisser l'epaule      (haut/bas)
    3. elbow_flex     = Plier/deplier le coude      (plie/tendu)
    4. wrist_flex     = Plier le poignet            (haut/bas)
    5. wrist_roll     = Tourner le poignet          (comme une cle)
    6. gripper        = Ouvrir/fermer la pince      (ouvert/ferme)

Pendant la calibration :
    - Etape 1 : Mettre CHAQUE articulation au MILIEU de sa course -> ENTER
    - Etape 2 : Bouger CHAQUE articulation (les 6) a FOND dans les DEUX sens -> ENTER
"""

import argparse
import glob
import subprocess
import sys

from config import FOLLOWER_ID, FOLLOWER_PORT, LEADER_ID, LEADER_PORT


def check_ports():
    """Verifie que les ports USB sont disponibles."""
    ports = glob.glob("/dev/tty.usbmodem*")
    if not ports:
        print("Aucun port USB detecte ! Le robot est-il branche et alimente ?")
        sys.exit(1)
    missing = []
    if LEADER_PORT not in ports:
        missing.append(f"Leader ({LEADER_PORT})")
    if FOLLOWER_PORT not in ports:
        missing.append(f"Follower ({FOLLOWER_PORT})")
    if missing:
        print(f"Ports non trouves : {', '.join(missing)}")
        print(f"  Ports detectes : {ports}")
        print("  Modifie scripts/config.py ou lance : ls /dev/tty.usbmodem*")
        sys.exit(1)


def calibrate_leader():
    """Calibre le bras leader (celui que vous bougez a la main)."""
    print()
    print("=" * 60)
    print("  CALIBRATION DU LEADER (bras que vous bougez a la main)")
    print("=" * 60)
    print()
    print("  Articulations a bouger (dans l'ordre) :")
    print("    1. shoulder_pan   -> Tourner la base a fond GAUCHE puis DROITE")
    print("    2. shoulder_lift  -> Lever le bras tout en HAUT puis tout en BAS")
    print("    3. elbow_flex     -> Plier le coude a FOND puis deplier a FOND")
    print("    4. wrist_flex     -> Plier le poignet en HAUT puis en BAS")
    print("    5. wrist_roll     -> Tourner le poignet, a fond GAUCHE puis DROITE")
    print("    6. gripper        -> OUVRIR la pince a fond puis FERMER a fond")
    print()
    input("  Appuyez sur ENTER quand vous etes pret...")
    print()

    cmd = [
        "lerobot-calibrate",
        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_ID}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n  Calibration du leader terminee !")
    except subprocess.CalledProcessError:
        print("\n  Erreur pendant la calibration du leader.")
        sys.exit(1)


def calibrate_follower():
    """Calibre le bras follower (celui qui imite / le robot)."""
    print()
    print("=" * 60)
    print("  CALIBRATION DU FOLLOWER (bras robot qui imite)")
    print("=" * 60)
    print()
    print("  Le follower a normalement le torque (force) active.")
    print("  LeRobot va le desactiver pour la calibration.")
    print("  Tenez le bras pour qu'il ne tombe pas !")
    print()
    print("  Meme chose : bougez CHAQUE articulation (les 6) a FOND")
    print("  dans les DEUX sens.")
    print()
    input("  Appuyez sur ENTER quand vous etes pret...")
    print()

    cmd = [
        "lerobot-calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n  Calibration du follower terminee !")
    except subprocess.CalledProcessError:
        print("\n  Erreur pendant la calibration du follower.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Calibrer le robot SO-101")
    parser.add_argument("--leader", action="store_true", help="Calibrer seulement le leader")
    parser.add_argument("--follower", action="store_true", help="Calibrer seulement le follower")
    args = parser.parse_args()

    check_ports()

    do_leader = args.leader or (not args.leader and not args.follower)
    do_follower = args.follower or (not args.leader and not args.follower)

    if do_leader:
        calibrate_leader()

    if do_follower:
        calibrate_follower()

    print()
    print("=" * 60)
    print("  Calibration terminee !")
    print("  Vous pouvez maintenant teleoperer :")
    print("    python scripts/teleoperate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
