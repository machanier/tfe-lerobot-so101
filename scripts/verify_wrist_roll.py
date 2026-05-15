#!/usr/bin/env python3
"""
verify_wrist_roll.py - Affichage en direct de wrist_roll pour valider la
calibration apres fix_wrist_roll_calibration.py.

Torque coupe : tu bouges wrist_roll a la main et tu visites 3 reperes :

    Centre mecanique : angle ~  0 deg, Present ~ 2047
    Butee 1          : angle ~ -167 deg, Present ~  150
    Butee 2          : angle ~ +167 deg, Present ~ 3944

Et tu observes que les transitions entre ces reperes sont continues (pas de
saut de 360 deg quand tu passes par la zone morte de l'autre cote). Si ces 3
points sortent comme prevu, la calibration moteur est validee et les
calibrations extrinsèques peuvent partir de mesures saines.

Usage :
    python scripts/verify_wrist_roll.py
    (ENTER pour arreter)
"""

import sys
import time
from pathlib import Path

import numpy as np

from calibrate_extrinsic import connect_robot
from config import FOLLOWER_PORT

JOINT = "wrist_roll"
REPO_ROOT = Path(__file__).resolve().parents[1]
# tolerance d'affichage du tag "centre/butee" (en counts encodeur)
TAG_TOL = 80


def main():
    try:
        from lerobot.utils.utils import enter_pressed
    except ImportError:
        print("ERREUR : LeRobot introuvable. Active le venv : source venv/bin/activate")
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
        print(f"  VERIFICATION DE wrist_roll  (lecture directe, torque coupe)")
        print("=" * 62)
        print(f"  Plage calibree : [{lo}, {hi}]   centre : {mid}")
        print(f"  Attendu :  centre -> 0 deg")
        print(f"             Present={lo}  -> {angle_at_lo:+.1f} deg")
        print(f"             Present={hi}  -> {angle_at_hi:+.1f} deg")
        print()
        print(f"  Bouge wrist_roll a la main : passe par le centre puis les")
        print(f"  deux butees. Verifie que les transitions sont continues")
        print(f"  (pas de saut de 360 deg).")
        print(f"  ENTER pour arreter.")
        print()

        while not enter_pressed():
            raw = int(bus.sync_read("Present_Position", normalize=False)[JOINT])
            angle = np.rad2deg(raw_to_radians(raw, wrist))
            if abs(raw - mid) < TAG_TOL:
                tag = "  <-- CENTRE"
            elif raw - lo < TAG_TOL:
                tag = "  <-- BUTEE (range_min)"
            elif hi - raw < TAG_TOL:
                tag = "  <-- BUTEE (range_max)"
            else:
                tag = ""
            print(f"\r    Present={raw:>4d}   angle={angle:+7.2f} deg{tag:<28}",
                  end="", flush=True)
            time.sleep(0.05)
        print()
        print()
        print("  Si les 3 reperes sortent comme attendu et les transitions sont")
        print("  continues -> wrist_roll est valide, tu peux enchainer cam_0.")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
