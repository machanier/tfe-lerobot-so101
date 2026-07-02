#!/usr/bin/env python3
"""
check_motor_calibration.py - Verifie la calibration moteur du follower.

A lancer apres `python scripts/calibrate.py --follower` et avant de recapturer
les donnees hand-eye. Detecte les defauts susceptibles d'invalider la
calibration :
  1. articulation sous-balayee : la plage calibree est nettement plus etroite
     que la course prevue par l'URDF (le joint n'a pas ete amene en butee) ;
  2. plage superieure a 345 deg : le balayage a probablement traverse la
     couture 0/4095 de l'encodeur 12 bits ;
  3. plage trop proche de la couture 0/4095 (risque de wraparound a l'usage).

Compare configs/calibration_follower.json aux limites de
configs/so101_new_calib.urdf.

Entree  : configs/calibration_follower.json, configs/so101_new_calib.urdf.
Sortie  : rapport par articulation sur la sortie standard ; code de retour 0
          si la calibration est coherente, 1 sinon.

Usage :
    python scripts/check_motor_calibration.py
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CALIB = REPO / "configs" / "calibration_follower.json"
URDF = REPO / "configs" / "so101_new_calib.urdf"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ENCODER_MAX = 4095            # STS3215 : encodeur 12 bits, 0..4095 = 360 deg

# Seuils de verification
MIN_SPAN_RATIO = 0.80         # plage calibree >= 80 % de la course URDF
MAX_SPAN_DEG = 345.0          # au-dela : le balayage a traverse la couture
SEAM_MARGIN = 120             # range_min/max doivent rester a >= 120 counts des bords


def urdf_joint_spans(urdf_path):
    """Retourne {joint: span_deg} d'apres les balises <limit> de l'URDF."""
    root = ET.parse(urdf_path).getroot()
    spans = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is not None and name in ARM_JOINTS:
            lo = float(limit.get("lower"))
            hi = float(limit.get("upper"))
            spans[name] = math.degrees(hi - lo)
    return spans


def main():
    if not CALIB.exists():
        print(f"ERREUR : {CALIB} introuvable.")
        print("  Executer d'abord : python scripts/calibrate.py --follower")
        sys.exit(1)
    if not URDF.exists():
        print(f"ERREUR : {URDF} introuvable.")
        sys.exit(1)

    calib = json.load(open(CALIB))
    urdf_spans = urdf_joint_spans(URDF)

    print(f"Verification : configs/{CALIB.name}")
    print(f"Reference    : configs/{URDF.name} (course mecanique attendue)")
    print()
    print(f"  {'articulation':<14} {'plage brute':<15} {'span':>9} {'attendu':>9} {'milieu':>8}  verdict")
    print(f"  {'-' * 14} {'-' * 15} {'-' * 9} {'-' * 9} {'-' * 8}  {'-' * 8}")

    all_ok = True
    for j in ARM_JOINTS:
        c = calib[j]
        lo, hi = c["range_min"], c["range_max"]
        span_deg = (hi - lo) * 360.0 / ENCODER_MAX
        mid = (lo + hi) / 2
        urdf_deg = urdf_spans.get(j)

        problems = []
        if urdf_deg and span_deg < MIN_SPAN_RATIO * urdf_deg:
            problems.append(
                f"sous-balayee : {span_deg:.0f} deg captures vs ~{urdf_deg:.0f} deg attendus"
            )
        if span_deg > MAX_SPAN_DEG:
            problems.append(
                f"plage de {span_deg:.0f} deg : le balayage a traverse la couture 0/4095"
            )
        if lo < SEAM_MARGIN or hi > ENCODER_MAX - SEAM_MARGIN:
            problems.append(
                f"plage proche de la couture 0/4095 (range=[{lo},{hi}], risque de wraparound)"
            )

        verdict = "OK" if not problems else "A REVOIR"
        if problems:
            all_ok = False
        attendu = f"{urdf_deg:.0f} deg" if urdf_deg else "?"
        print(f"  {j:<14} [{lo:>5}, {hi:>5}]  {span_deg:>6.0f} deg {attendu:>9} {mid:>8.0f}  {verdict}")
        for p in problems:
            print(f"  {'':<14} -> {p}")

    print()
    if all_ok:
        print("[OK] Calibration moteur coherente.")
        print("  Etape suivante : recapturer les parametres extrinseques des 3 cameras.")
        sys.exit(0)
    else:
        print("[A REVOIR] Au moins une articulation pose probleme.")
        print("  Relancer : python scripts/calibrate.py --follower")
        print("    etape 1 : placer chaque joint au milieu de sa course (en particulier wrist_roll)")
        print("    etape 2 : balayer chaque joint jusqu'en butee dans les deux sens")
        sys.exit(1)


if __name__ == "__main__":
    main()
