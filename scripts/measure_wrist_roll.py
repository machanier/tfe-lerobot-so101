#!/usr/bin/env python3
"""
measure_wrist_roll.py - Mesure la course reelle de wrist_roll.

wrist_roll a une grande course angulaire (~330 deg) qui chevauche la couture
0/4095 de l'encodeur 12 bits. Plutot que de tenter de le re-homer physiquement
pour eviter la couture (fragile vu la course), on mesure sa course reelle et
on laisse le logiciel "derouler" l'encodeur autour du centre de cette course
(cf src/calibration/motor_to_angle.py).

Le robot n'est jamais commande : torque desactive, tu bouges wrist_roll a la
main pendant que le script lit les positions en continu.

Usage :
    python scripts/measure_wrist_roll.py

Procedure :
    1. ENTER pour demarrer l'enregistrement.
    2. Balaye wrist_roll LENTEMENT d'une butee a l'autre, 2-3 allers-retours
       COMPLETS.
    3. ENTER pour arreter.
    4. Le script calcule le centre de la course et l'ecrit dans
       configs/encoder_unwrap.json.

A relancer uniquement si tu recalibres les moteurs du follower : le centre
mesure depend du homing courant. Le script enregistre le homing_offset du
moment, pour pouvoir detecter une mesure devenue obsolete.
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
        print("ERREUR : LeRobot introuvable. Active le venv : source venv/bin/activate")
        sys.exit(1)

    bus, _ = connect_robot(FOLLOWER_PORT)
    try:
        print()
        print("=" * 62)
        print("  MESURE DE LA COURSE DE wrist_roll")
        print("=" * 62)
        print("  Torque desactive : tu bouges wrist_roll a la main.")
        print()
        print("  1. ENTER pour demarrer l'enregistrement.")
        print("  2. Balaye wrist_roll LENTEMENT d'une butee a l'autre,")
        print("     fais 2-3 allers-retours COMPLETS (butee a butee).")
        print("  3. ENTER de nouveau pour arreter.")
        print()
        input("  ENTER pour demarrer...")
        print()

        # Echantillonnage continu : chaque pas est petit, donc wrap() leve
        # l'ambiguite de la couture sans risque, et le deroulage est fiable.
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
            print("  ATTENTION : course < 90 deg. As-tu bien balaye jusqu'aux DEUX")
            print("  butees ? Rien n'a ete enregistre, relance le script.")
            return
        if span_deg >= 358:
            print("  PROBLEME : course >= 358 deg. wrist_roll ferait quasiment un tour")
            print("  complet -> centre non definissable sans ambiguite. Montre-moi ca.")
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
        print("  Termine. Montre-moi la sortie ci-dessus.")
        print("  (inutile de relancer calibrate.py : wrist_roll est gere en logiciel)")
        print("=" * 62)
    except KeyboardInterrupt:
        print("\n  Annule (rien enregistre).")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
