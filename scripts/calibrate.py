#!/usr/bin/env python3
"""Recalibration du bras leader et/ou du bras follower du SO-101.

Usage :
    python scripts/calibrate.py            # Calibrer les deux (leader puis follower)
    python scripts/calibrate.py --leader   # Calibrer seulement le leader
    python scripts/calibrate.py --follower # Calibrer seulement le follower

La calibration ecrit les fichiers configs/calibration_leader.json et
configs/calibration_follower.json utilises par le reste du projet.

Rappel des articulations (de la base vers la pince) :
    1. shoulder_pan   = Rotation de la base    (gauche/droite)
    2. shoulder_lift  = Lever/baisser l'epaule (haut/bas)
    3. elbow_flex     = Plier/deplier le coude (plie/tendu)
    4. wrist_flex     = Plier le poignet       (haut/bas)
    5. wrist_roll     = Tourner le poignet     (comme une cle)
    6. gripper        = Ouvrir/fermer la pince (ouvert/ferme)

Deroulement de la calibration :
    - Etape 1 : placer chaque articulation au milieu de sa course, puis valider.
    - Etape 2 : bouger chaque articulation (les six) a fond dans les deux sens, puis valider.
"""

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

from config import FOLLOWER_ID, FOLLOWER_PORT, LEADER_ID, LEADER_PORT

REPO_ROOT = Path(__file__).resolve().parents[1]


def sync_calibration_to_configs(kind):
    """Recopie la calibration generee par LeRobot vers configs/.

    LeRobot ecrit la calibration dans son cache
    (~/.cache/huggingface/lerobot/calibration/...). Le nom du sous-dossier
    depend de la version de LeRobot (so_follower, so101_follower, ...), donc
    on prend le fichier {id}.json le plus recemment ecrit. Le reste du projet
    lit configs/calibration_{leader,follower}.json : on y recopie le resultat
    juste apres la calibration.
    """
    try:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    except ImportError:
        print("  Avertissement : LeRobot introuvable, copie de la calibration ignoree.")
        return

    if kind == "leader":
        search_dir = HF_LEROBOT_CALIBRATION / "teleoperators"
        motor_id = LEADER_ID
        dst = REPO_ROOT / "configs" / "calibration_leader.json"
    else:
        search_dir = HF_LEROBOT_CALIBRATION / "robots"
        motor_id = FOLLOWER_ID
        dst = REPO_ROOT / "configs" / "calibration_follower.json"

    candidates = list(search_dir.glob(f"*/{motor_id}.json"))
    if not candidates:
        print(f"  Avertissement : aucune calibration LeRobot trouvee dans {search_dir}")
        print(f"  Copiez le fichier manuellement vers {dst}")
        return

    # le fichier le plus recent = celui que la calibration vient d'ecrire
    src = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copyfile(src, dst)
    print(f"  Calibration synchronisee : {src.parent.name}/{src.name} -> configs/{dst.name}")


def check_ports(need_leader, need_follower):
    """Verifie que les ports USB necessaires sont disponibles.

    On ne verifie que les ports utiles a la calibration demandee : inutile
    d'avoir le leader branche pour `--follower` (et inversement).
    """
    ports = glob.glob("/dev/tty.usbmodem*")
    if not ports:
        print("Aucun port USB detecte ! Le robot est-il branche et alimente ?")
        sys.exit(1)
    missing = []
    if need_leader and LEADER_PORT not in ports:
        missing.append(f"Leader ({LEADER_PORT})")
    if need_follower and FOLLOWER_PORT not in ports:
        missing.append(f"Follower ({FOLLOWER_PORT})")
    if missing:
        print(f"Ports non trouves : {', '.join(missing)}")
        print(f"  Ports detectes : {ports}")
        print("  Ajustez scripts/config.py ou lancez : ls /dev/tty.usbmodem*")
        sys.exit(1)


def calibrate_leader():
    """Calibre le bras leader (celui que l'operateur deplace a la main)."""
    print()
    print("=" * 60)
    print("  Calibration du leader (bras deplace a la main)")
    print("=" * 60)
    print()
    print("  Articulations a bouger (dans l'ordre) :")
    print("    1. shoulder_pan   -> Tourner la base a fond a gauche puis a droite")
    print("    2. shoulder_lift  -> Lever le bras tout en haut puis tout en bas")
    print("    3. elbow_flex     -> Plier le coude a fond puis le deplier a fond")
    print("    4. wrist_flex     -> Plier le poignet vers le haut puis vers le bas")
    print("    5. wrist_roll     -> Tourner le poignet a fond a gauche puis a droite")
    print("    6. gripper        -> Ouvrir la pince a fond puis la fermer a fond")
    print()
    input("  Appuyez sur Entree lorsque vous etes pret...")
    print()

    cmd = [
        "lerobot-calibrate",
        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_ID}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n  Calibration du leader terminee.")
        sync_calibration_to_configs("leader")
    except subprocess.CalledProcessError:
        print("\n  Erreur pendant la calibration du leader.")
        sys.exit(1)


def calibrate_follower():
    """Calibre le bras follower (le bras robot qui reproduit les mouvements)."""
    print()
    print("=" * 60)
    print("  Calibration du follower (bras robot)")
    print("=" * 60)
    print()
    print("  Le follower a normalement le couple (torque) active.")
    print("  LeRobot va le desactiver le temps de la calibration.")
    print("  Tenez le bras pour qu'il ne retombe pas.")
    print()
    print("  Comme pour le leader : bougez chaque articulation (les six)")
    print("  a fond dans les deux sens.")
    print()
    input("  Appuyez sur Entree lorsque vous etes pret...")
    print()

    cmd = [
        "lerobot-calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n  Calibration du follower terminee.")
        sync_calibration_to_configs("follower")
    except subprocess.CalledProcessError:
        print("\n  Erreur pendant la calibration du follower.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Calibrer le robot SO-101")
    parser.add_argument("--leader", action="store_true", help="Calibrer seulement le leader")
    parser.add_argument("--follower", action="store_true", help="Calibrer seulement le follower")
    args = parser.parse_args()

    do_leader = args.leader or (not args.leader and not args.follower)
    do_follower = args.follower or (not args.leader and not args.follower)

    check_ports(do_leader, do_follower)

    if do_leader:
        calibrate_leader()

    if do_follower:
        calibrate_follower()

    print()
    print("=" * 60)
    print("  Calibration terminee.")
    if do_follower:
        print("  Etape suivante : verifier la calibration moteur avec")
        print("    python scripts/check_motor_calibration.py")
    else:
        print("  Vous pouvez maintenant teleoperer avec")
        print("    python scripts/teleoperate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
