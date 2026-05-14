#!/usr/bin/env python3
"""
center_wrist_roll.py - Positionne wrist_roll au centre de sa course.

wrist_roll a une grande course angulaire qui, mal centree, chevauche la
couture 0/4095 de l'encodeur 12 bits : les angles "sautent" alors de 360 deg
d'une mesure a l'autre. Ce script mesure la course reelle de wrist_roll et te
guide pour le placer exactement au centre, AVANT de relancer la calibration
moteur (de cette facon la couture tombe dans la zone que le joint n'atteint
jamais).

Le robot n'est jamais commande : le script lit seulement les positions
(torque desactive, tu bouges le bras a la main).

Usage :
    python scripts/center_wrist_roll.py

Procedure :
    1. Le script lit wrist_roll a ses deux butees, puis au milieu.
    2. Il calcule le centre de la course.
    3. Il affiche en direct l'ecart au centre : tu tournes wrist_roll
       jusqu'a "CENTRE".
    4. Tu laisses wrist_roll la et tu relances la calibration moteur :
         python scripts/calibrate.py --follower
       (au temps 1, centre les AUTRES joints, NE TOUCHE PAS wrist_roll)
"""

import sys
import time

from calibrate_extrinsic import connect_robot
from config import FOLLOWER_PORT

ENCODER_FULL = 4096      # STS3215 : encodeur 12 bits, 0..4095
HALF = ENCODER_FULL // 2
TOL_COUNTS = 100         # +-100 counts ~ +-9 deg : tolerance de centrage
STABLE_ITERS = 30        # ~1.5 s stable avant de valider le centrage
JOINT = "wrist_roll"


def wrap(d):
    """Ramene une difference d'encodeur dans (-2048, 2048]."""
    return ((d + HALF) % ENCODER_FULL) - HALF


def read_wrist_roll(bus):
    """Lit la position brute (raw encoder) de wrist_roll."""
    return float(bus.sync_read("Present_Position", normalize=False)[JOINT])


def main():
    bus, _ = connect_robot(FOLLOWER_PORT)
    try:
        print()
        print("=" * 60)
        print("  CENTRAGE DE wrist_roll")
        print("=" * 60)
        print("  Le bras est manipulable a la main (torque desactive).")
        print()

        # --- Phase 1 : mesurer la course ---
        input("  1/3  Tourne wrist_roll A FOND d'un cote, puis ENTER...")
        a = read_wrist_roll(bus)
        input("  2/3  Tourne wrist_roll A FOND de l'autre cote, puis ENTER...")
        b = read_wrist_roll(bus)
        input("  3/3  Remets wrist_roll a PEU PRES au milieu, puis ENTER...")
        c = read_wrist_roll(bus)

        # Deroule c puis b en partant de a (chaque demi-course fait < 180 deg,
        # donc wrap() leve l'ambiguite de la couture sans risque)
        c_u = a + wrap(c - a)
        b_u = c_u + wrap(b - c)
        lo, hi = min(a, b_u), max(a, b_u)
        span_deg = (hi - lo) * 360.0 / (ENCODER_FULL - 1)
        center_raw = round((lo + hi) / 2) % ENCODER_FULL

        print()
        print(f"  Course mesuree : {span_deg:.0f} deg")
        if span_deg > 350:
            print("  PROBLEME : wrist_roll semble tourner sur ~360 deg ou plus,")
            print("  impossible de le centrer hors de la couture. Montre-moi ce message.")
            return
        if span_deg < 90:
            print("  ATTENTION : course tres courte (<90 deg). As-tu bien tourne")
            print("  jusqu'aux deux butees ? Relance le script si besoin.")
        print(f"  Centre vise : position encodeur {center_raw}")
        print()

        # --- Phase 2 : guidage vers le centre ---
        print("  Tourne wrist_roll pour ramener l'ecart vers 0 deg.")
        print("  (si l'ecart augmente, tourne dans l'autre sens)")
        print()
        stable = 0
        while stable < STABLE_ITERS:
            pos = read_wrist_roll(bus)
            delta = wrap(pos - center_raw)
            if abs(delta) <= TOL_COUNTS:
                stable += 1
                label = "CENTRE " + "." * (stable // 5)
            else:
                stable = 0
                delta_deg = delta * 360.0 / (ENCODER_FULL - 1)
                label = f"ecart {delta_deg:+6.1f} deg"
            print(f"\r    wrist_roll : {label:<34}", end="", flush=True)
            time.sleep(0.05)

        print()
        print()
        print("=" * 60)
        print("  wrist_roll est CENTRE. NE LE BOUGE PLUS.")
        print("=" * 60)
        print("  Etape suivante :")
        print("    python scripts/calibrate.py --follower")
        print("  -> temps 1 (milieu) : centre les AUTRES joints, laisse wrist_roll")
        print("     tel quel (il est deja centre).")
        print("  -> temps 2 : balaye tous les joints a fond, wrist_roll inclus.")
        print()
    except KeyboardInterrupt:
        print("\n  Annule.")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
