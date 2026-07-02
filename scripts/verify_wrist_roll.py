#!/usr/bin/env python3
"""Affichage en direct de l'angle wrist_roll pour valider la calibration moteur.

Le torque est coupe : l'articulation wrist_roll se deplace a la main afin de
visiter trois reperes de reference.

    Centre mecanique : angle proche de 0 deg, Present proche de 2047
    Premiere butee   : angle proche de -167 deg, Present proche de 150
    Seconde butee    : angle proche de +167 deg, Present proche de 3944

L'objectif est de verifier que les transitions entre ces reperes sont continues,
sans saut de 360 deg au passage de la zone morte oppose. Si ces trois points
apparaissent comme prevu, la calibration moteur est validee et les calibrations
extrinseques peuvent partir de mesures saines.

Usage :
    python scripts/verify_wrist_roll.py
    (touche ENTER pour arreter)

Entrees : configs/calibration_follower.json (plage calibree de wrist_roll).
Sortie  : affichage console de l'angle courant et des reperes atteints.
"""

import sys
import time
from pathlib import Path

import numpy as np

from calibrate_extrinsic import connect_robot
from config import FOLLOWER_PORT

JOINT = "wrist_roll"
REPO_ROOT = Path(__file__).resolve().parents[1]
# Tolerance d'affichage du repere "centre/butee", exprimee en counts encodeur.
TAG_TOL = 80


def main():
    try:
        from lerobot.utils.utils import enter_pressed
    except ImportError:
        print("ERREUR : LeRobot introuvable. Activer l'environnement virtuel : source venv/bin/activate")
        sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT))
    from src.calibration.motor_to_angle import load_motor_calibration, raw_to_radians

    calib = load_motor_calibration(REPO_ROOT / "configs" / "calibration_follower.json")
    wrist = calib[JOINT]
    lo, hi = wrist["range_min"], wrist["range_max"]
    mid = (lo + hi) // 2
    angle_at_lo = np.rad2deg(raw_to_radians(lo, wrist))
    angle_at_hi = np.rad2deg(raw_to_radians(hi, wrist))

    bus, _ = connect_robot(FOLLOWER_PORT)
    try:
        print()
        print("=" * 62)
        print("  Verification de wrist_roll  (lecture directe, torque coupe)")
        print("=" * 62)
        print(f"  Plage calibree : [{lo}, {hi}]   centre : {mid}")
        print("  Attendu :  centre -> 0 deg")
        print(f"             Present={lo}  -> {angle_at_lo:+.1f} deg")
        print(f"             Present={hi}  -> {angle_at_hi:+.1f} deg")
        print()
        print("  Deplacer wrist_roll a la main : passer par le centre puis les")
        print("  deux butees, en verifiant que les transitions sont continues")
        print("  (pas de saut de 360 deg).")
        print("  Touche ENTER pour arreter.")
        print()

        while not enter_pressed():
            raw = int(bus.sync_read("Present_Position", normalize=False)[JOINT])
            angle = np.rad2deg(raw_to_radians(raw, wrist))
            if abs(raw - mid) < TAG_TOL:
                tag = "  <-- centre"
            elif raw - lo < TAG_TOL:
                tag = "  <-- butee (range_min)"
            elif hi - raw < TAG_TOL:
                tag = "  <-- butee (range_max)"
            else:
                tag = ""
            print(f"\r    Present={raw:>4d}   angle={angle:+7.2f} deg{tag:<28}",
                  end="", flush=True)
            time.sleep(0.05)
        print()
        print()
        print("  Si les trois reperes apparaissent comme attendu et que les")
        print("  transitions sont continues, wrist_roll est valide et la")
        print("  calibration de cam_0 peut suivre.")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
