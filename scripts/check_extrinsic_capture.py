#!/usr/bin/env python3
"""Vérifie un fichier de capture extrinsèque hand-eye.

À exécuter après `python scripts/calibrate_extrinsic.py --index <N>`, avant de
passer à la caméra suivante. Contrôle que la capture est exploitable par le
solveur hand-eye selon trois critères :
  1. nombre de poses suffisant (>= 15) ;
  2. positions moteur toutes dans la plage calibrée (détecte le repliement
     d'encodeur) ;
  3. orientations du damier suffisamment variées (conditionnement hand-eye).

Entrée : configs/extrinsic_capture_cam_<index>.json et
configs/calibration_follower.json.
Sortie : rapport texte et code de retour 0 (capture exploitable) ou 1 (à revoir).

Usage :
    python scripts/check_extrinsic_capture.py --index 0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

MIN_POSES = 15
RANGE_TOL = 50                 # tolérance en counts hors plage calibrée (~4 deg)
MIN_MAX_PAIR_ROT_DEG = 60.0    # au moins une paire de poses à plus de 60 deg d'écart
MIN_MEAN_PAIR_ROT_DEG = 20.0   # écart moyen minimal entre poses


def rotation_angle_between(rvec_a, rvec_b):
    """Angle (en degrés) de la rotation relative entre deux poses (rvec OpenCV)."""
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64))
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64))
    R_rel = Ra.T @ Rb
    cos = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser(description="Vérifie une capture extrinsèque hand-eye")
    parser.add_argument("--index", type=int, required=True, help="Index de la caméra à vérifier (0, 1 ou 2).")
    args = parser.parse_args()

    cap_path = REPO / "configs" / f"extrinsic_capture_cam_{args.index}.json"
    calib_path = REPO / "configs" / "calibration_follower.json"
    if not cap_path.exists():
        print(f"ERREUR : {cap_path} introuvable.")
        print(f"  Exécutez d'abord : python scripts/calibrate_extrinsic.py --index {args.index}")
        sys.exit(1)
    if not calib_path.exists():
        print(f"ERREUR : {calib_path} introuvable.")
        sys.exit(1)

    data = json.load(open(cap_path))
    calib = json.load(open(calib_path))
    captures = data["captures"]

    print(f"Vérification : configs/{cap_path.name}  ({data.get('camera_key', '?')})")
    print()

    problems = []

    # 1. Nombre de poses
    n = len(captures)
    poses_ok = n >= MIN_POSES
    print(f"  Poses                 : {n}  ({'OK' if poses_ok else f'insuffisant (visez >= {MIN_POSES})'})")
    if not poses_ok:
        problems.append(f"seulement {n} poses (minimum {MIN_POSES})")

    # 2. Positions moteur dans la plage calibrée (détecte le repliement d'encodeur)
    out_of_range = {}
    for j in ARM_JOINTS:
        lo = calib[j]["range_min"] - RANGE_TOL
        hi = calib[j]["range_max"] + RANGE_TOL
        bad = [c["id"] for c in captures if not (lo <= c["motor_positions_raw"][j] <= hi)]
        if bad:
            out_of_range[j] = bad
    if not out_of_range:
        print(f"  Positions moteur      : toutes dans la plage calibrée  (OK)")
    else:
        print(f"  Positions moteur      : hors plage")
        for j, bad in out_of_range.items():
            preview = ", ".join(str(i) for i in bad[:8]) + ("..." if len(bad) > 8 else "")
            print(f"    - {j:<14}: {len(bad)}/{n} captures hors plage (ids : {preview})")
        total = sum(len(b) for b in out_of_range.values())
        problems.append(
            f"{total} positions moteur hors plage : la calibration moteur est encore "
            f"incorrecte (repliement d'encodeur) ; recalibrez le follower avant de recapturer"
        )

    # 3. Diversité des orientations du damier
    rvecs = [c["rvec_target_cam"] for c in captures]
    pair_angles = [
        rotation_angle_between(rvecs[i], rvecs[k])
        for i in range(len(rvecs))
        for k in range(i + 1, len(rvecs))
    ]
    max_rot = max(pair_angles) if pair_angles else 0.0
    mean_rot = float(np.mean(pair_angles)) if pair_angles else 0.0
    div_ok = max_rot >= MIN_MAX_PAIR_ROT_DEG and mean_rot >= MIN_MEAN_PAIR_ROT_DEG
    print(f"  Diversité orientation : écart max {max_rot:.0f} deg, moyen {mean_rot:.0f} deg  "
          f"({'OK' if div_ok else 'trop uniforme'})")
    if not div_ok:
        problems.append(
            "orientations du damier trop similaires ; variez davantage les angles "
            "(inclinaisons, rotations sur au moins deux axes) entre les poses"
        )

    print()
    if not problems:
        print(f"[OK] Capture cam_{args.index} exploitable pour le solveur hand-eye.")
        sys.exit(0)
    else:
        print(f"[À revoir] cam_{args.index} :")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
