#!/usr/bin/env python3
"""Mesure la course angulaire reelle du moteur wrist_roll.

wrist_roll dispose d'une grande course angulaire (environ 330 deg) qui chevauche
la couture 0/4095 de l'encodeur 12 bits. Plutot que de le re-homer physiquement
pour eviter cette couture (peu fiable vu l'amplitude), le script mesure sa course
reelle et laisse le logiciel derouler l'encodeur autour du centre de cette course
(voir src/calibration/motor_to_angle.py).

Le robot n'est jamais commande : le couple est desactive et l'operateur deplace
wrist_roll a la main pendant que le script lit les positions en continu.

Usage :
    python scripts/measure_wrist_roll.py

Procedure :
    1. Appuyer sur ENTER pour demarrer l'enregistrement.
    2. Balayer wrist_roll lentement d'une butee a l'autre, sur 2 a 3
       allers-retours complets.
    3. Appuyer sur ENTER pour arreter.
    4. Le script calcule le centre de la course et l'ecrit dans
       configs/encoder_unwrap.json.

Entree : positions lues sur le port du follower (configs/calibration_follower.json
pour le homing_offset courant).
Sortie : configs/encoder_unwrap.json (centre de course et amplitude mesuree).

A relancer uniquement apres une recalibration des moteurs du follower : le centre
mesure depend du homing courant. Le homing_offset du moment est enregistre afin de
detecter une mesure devenue obsolete.
"""

import json
import sys
import time
from pathlib import Path

from calibrate_extrinsic import connect_robot
from config import FOLLOWER_PORT

ENCODER_FULL = 4096      # STS3215 : encodeur 12 bits, 0..4095
HALF = ENCODER_FULL // 2
JOINT = "wrist_roll"
REPO_ROOT = Path(__file__).resolve().parents[1]
UNWRAP_PATH = REPO_ROOT / "configs" / "encoder_unwrap.json"
CALIB_PATH = REPO_ROOT / "configs" / "calibration_follower.json"


def wrap(d):
    """Ramene une difference d'encodeur dans (-2048, 2048] (encodeur circulaire)."""
    return ((d + HALF) % ENCODER_FULL) - HALF


def main():
    try:
        from lerobot.utils.utils import enter_pressed
    except ImportError:
        print("Erreur : LeRobot introuvable. Activez l'environnement : source venv/bin/activate")
        sys.exit(1)

    bus, _ = connect_robot(FOLLOWER_PORT)
    try:
        print()
        print("=" * 62)
        print("  Mesure de la course de wrist_roll")
        print("=" * 62)
        print("  Couple desactive : deplacez wrist_roll a la main.")
        print()
        print("  1. Appuyez sur ENTER pour demarrer l'enregistrement.")
        print("  2. Balayez wrist_roll lentement d'une butee a l'autre,")
        print("     sur 2 a 3 allers-retours complets (butee a butee).")
        print("  3. Appuyez de nouveau sur ENTER pour arreter.")
        print()
        input("  ENTER pour demarrer...")
        print()

        # Echantillonnage continu : chaque pas etant petit, wrap() leve
        # l'ambiguite de la couture et le deroulage reste fiable.
        prev = float(bus.sync_read("Present_Position", normalize=False)[JOINT])
        unwrapped = prev
        lo = hi = unwrapped

        while not enter_pressed():
            cur = float(bus.sync_read("Present_Position", normalize=False)[JOINT])
            unwrapped += wrap(cur - prev)
            prev = cur
            lo = min(lo, unwrapped)
            hi = max(hi, unwrapped)
            span_deg = (hi - lo) * 360.0 / (ENCODER_FULL - 1)
            print(f"\r    course balayee : {span_deg:6.1f} deg    (ENTER pour arreter)",
                  end="", flush=True)
            time.sleep(0.03)
        print()
        print()

        span_deg = (hi - lo) * 360.0 / (ENCODER_FULL - 1)
        center_raw = round((lo + hi) / 2) % ENCODER_FULL

        print(f"  Course mesuree      : {span_deg:.1f} deg")
        print(f"  Centre de la course : encodeur {center_raw}")
        print()

        if span_deg < 90:
            print("  Attention : course inferieure a 90 deg. Verifiez que le balayage")
            print("  atteint bien les deux butees. Rien n'a ete enregistre, relancez le")
            print("  script.")
            return
        if span_deg >= 358:
            print("  Anomalie : course superieure ou egale a 358 deg. wrist_roll ferait")
            print("  quasiment un tour complet, le centre ne peut pas etre defini sans")
            print("  ambiguite. Verifiez la mesure avant de relancer.")
            return

        calib = json.load(open(CALIB_PATH))
        homing = calib[JOINT]["homing_offset"]

        result = {
            JOINT: {
                "unwrap_center": center_raw,
                "reachable_span_deg": round(span_deg, 1),
                "homing_offset_when_measured": homing,
                "note": ("Centre de la course reelle de wrist_roll, mesure par "
                         "scripts/measure_wrist_roll.py. wrist_roll a une course large "
                         "qui chevauche la couture 0/4095 de l'encodeur 12 bits ; "
                         "motor_to_angle.py deroule les angles autour de ce centre. "
                         "A re-mesurer si le follower est recalibre (le homing change)."),
            }
        }
        with open(UNWRAP_PATH, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  -> enregistre dans configs/{UNWRAP_PATH.name}")
        print()
        print("=" * 62)
        print("  Termine.")
        print("  Inutile de relancer calibrate.py : wrist_roll est gere en logiciel.")
        print("=" * 62)
    except KeyboardInterrupt:
        print("\n  Annule (rien enregistre).")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
