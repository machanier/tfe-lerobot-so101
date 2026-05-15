#!/usr/bin/env python3
"""
fix_wrist_roll_calibration.py - Recale le Homing_Offset de wrist_roll a partir
de la mesure faite par measure_wrist_roll.py.

Avant : wrist_roll a `range=[0, 4095]` dans calibration_follower.json
(artefact de la couture 0/4095 traversee pendant la calibration moteur).
Apres : wrist_roll a une plage propre centree autour de 2047, sans toucher
la couture. Le servo recoit un nouveau Homing_Offset, et le fichier de
calibration est mis a jour pour rester en accord.

Effet :
  - servo wrist_roll : Homing_Offset modifie (ecriture standard via le bus
    LeRobot, comme le ferait `lerobot-calibrate`)
  - configs/calibration_follower.json : entree wrist_roll mise a jour
  - configs/encoder_unwrap.json : supprime (plus necessaire)

Pre-requis :
  - measure_wrist_roll.py a deja ete lance (encoder_unwrap.json existe)
  - le follower n'a pas ete recalibre depuis la mesure (le script verifie)

Usage :
    python scripts/fix_wrist_roll_calibration.py
"""

import json
import sys
from pathlib import Path

from calibrate_extrinsic import connect_robot
from config import FOLLOWER_PORT

JOINT = "wrist_roll"
ENCODER_MAX = 4095          # STS3215 : valeur max (0..4095)
HALF_TURN = ENCODER_MAX // 2  # 2047 : cible Present apres recalage

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = REPO_ROOT / "configs" / "calibration_follower.json"
UNWRAP_PATH = REPO_ROOT / "configs" / "encoder_unwrap.json"


def signed_wrap(value, modulus=4096):
    """Ramene une valeur dans (-modulus/2, modulus/2]."""
    half = modulus // 2
    return ((value + half) % modulus) - half


def main():
    if not UNWRAP_PATH.exists():
        print(f"ERREUR : {UNWRAP_PATH} introuvable.")
        print("  Lance d'abord : python scripts/measure_wrist_roll.py")
        sys.exit(1)

    info = json.load(open(UNWRAP_PATH))[JOINT]
    measured_center = info["unwrap_center"]
    measured_span_deg = info["reachable_span_deg"]
    measured_homing = info["homing_offset_when_measured"]

    calib = json.load(open(CALIB_PATH))
    current_homing = calib[JOINT]["homing_offset"]

    if current_homing != measured_homing:
        print(f"ERREUR : la calibration moteur a change depuis la mesure.")
        print(f"  homing actuel               : {current_homing}")
        print(f"  homing au moment de la mesure : {measured_homing}")
        print(f"  -> relance python scripts/measure_wrist_roll.py")
        sys.exit(1)

    # Actual_at_center = Present_at_center + Homing  (Present = Actual - Homing)
    # On veut new_Present_at_center = 2047 -> new_Homing = Actual_at_center - 2047
    actual_at_center = (measured_center + current_homing) % 4096
    new_homing = signed_wrap(actual_at_center - HALF_TURN)

    # Nouvelle plage : centree sur 2047, de demi-course = mesure / 2
    half_span_counts = round(measured_span_deg * ENCODER_MAX / 360 / 2)
    new_range_min = HALF_TURN - half_span_counts
    new_range_max = HALF_TURN + half_span_counts

    print()
    print("=" * 62)
    print("  RECALAGE DU Homing_Offset DE wrist_roll")
    print("=" * 62)
    print(f"  Mesure utilisee :")
    print(f"    centre de la course (Present) : {measured_center}")
    print(f"    course                        : {measured_span_deg} deg")
    print(f"    homing au moment de la mesure : {measured_homing}")
    print()
    print(f"  Avant -> apres :")
    print(f"    homing_offset : {current_homing:>6} -> {new_homing:>6}")
    print(f"    range         : [{calib[JOINT]['range_min']:>4}, "
          f"{calib[JOINT]['range_max']:>4}] -> [{new_range_min:>4}, {new_range_max:>4}]")
    print(f"    -> Present au centre = {HALF_TURN} (milieu encodeur,")
    print(f"       couture 0/4095 dans la zone morte du joint)")
    print()

    bus, _ = connect_robot(FOLLOWER_PORT)
    try:
        before = int(bus.sync_read("Present_Position", normalize=False)[JOINT])

        print(f"  Ecriture du nouveau Homing_Offset...")
        bus.write("Homing_Offset", JOINT, new_homing, normalize=False)

        readback = int(bus.read("Homing_Offset", JOINT, normalize=False))
        after = int(bus.sync_read("Present_Position", normalize=False)[JOINT])
        # Apres ecriture : Present devrait avoir bouge de (current_homing - new_homing)
        expected = (before + current_homing - new_homing) % 4096
        delta = signed_wrap(after - expected)

        print(f"    Homing_Offset relu : {readback}")
        print(f"    Present_Position : avant={before} -> apres={after} (attendu ~{expected})")

        if readback != new_homing:
            print(f"  ERREUR : la relecture du Homing_Offset ne correspond pas.")
            sys.exit(1)
        if abs(delta) > 100:
            print(f"  ATTENTION : Present apres differe de l'attendu de {delta} counts.")
            print(f"  (peut etre normal si le joint a derive sous gravite, torque coupe)")
    finally:
        bus.disconnect()

    # Mise a jour de calibration_follower.json
    calib[JOINT]["homing_offset"] = int(new_homing)
    calib[JOINT]["range_min"] = int(new_range_min)
    calib[JOINT]["range_max"] = int(new_range_max)
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=4)
    print(f"\n  configs/{CALIB_PATH.name} : entree {JOINT} mise a jour")

    UNWRAP_PATH.unlink()
    print(f"  configs/{UNWRAP_PATH.name} : supprime (plus necessaire)")

    print()
    print("=" * 62)
    print("  Termine. Verifie avec : python scripts/check_motor_calibration.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
