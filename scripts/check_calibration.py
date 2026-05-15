#!/usr/bin/env python3
"""
check_calibration.py - Validation complete de la chaine de calibration.

Verifie d'un coup :
  - Calibration moteur (range, plage encodeur, conformite URDF).
  - Calibration intrinseque des 3 cameras (erreurs de reprojection).
  - Calibration extrinseque (hand-eye) des 3 cameras (positions, residus).
  - Coherence inter-cameras (baseline stereo cam_0 <-> cam_1).
  - Self-tests des modules cinematiques (transforms + FK).

Usage :
    python scripts/check_calibration.py
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def ok(label, value=""):
    print(f"  [OK] {label}{' : ' + value if value else ''}")


def warn(label, value=""):
    print(f"  [!]  {label}{' : ' + value if value else ''}")


def check_intrinsics():
    section("CALIBRATIONS INTRINSEQUES (cameras)")
    all_ok = True
    for i in [0, 1, 2]:
        p = REPO / f"configs/calibration_cam_{i}.json"
        if not p.exists():
            warn(f"cam_{i}", f"fichier absent ({p})")
            all_ok = False
            continue
        c = json.load(open(p))
        err = c.get("reprojection_error", float("inf"))
        n = c.get("num_captures", 0)
        K = c["camera_matrix"]
        fx, fy = K[0][0], K[1][1]
        verdict = "OK" if err < 0.5 else ("acceptable" if err < 1.0 else "DEGRADE")
        print(f"  cam_{i} : fx={fx:7.1f}  fy={fy:7.1f}  erreur reproj = {err:.3f} px  "
              f"({n} captures) -> {verdict}")
        if err >= 1.0:
            all_ok = False
    return all_ok


def check_motor():
    section("CALIBRATION MOTEUR (follower)")
    try:
        result = subprocess.run(
            [sys.executable,str(REPO / "scripts/check_motor_calibration.py")],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        warn("check_motor_calibration.py introuvable")
        return False
    # On reaffiche la sortie utile (saute l'en-tete deja imprime)
    for line in result.stdout.splitlines():
        if line.strip():
            print("  " + line)
    return result.returncode == 0


def check_extrinsics():
    section("CALIBRATIONS HAND-EYE (extrinseques)")
    all_ok = True
    cams = {}
    for i in [0, 1, 2]:
        p = REPO / f"configs/handeye_cam_{i}.json"
        if not p.exists():
            warn(f"cam_{i}", f"fichier absent ({p})")
            all_ok = False
            continue
        d = json.load(open(p))
        cams[i] = d
        T = np.array(d["transform"])
        pos = T[:3, 3] * 1000.0
        res = d["residuals"]
        n_used = d.get("n_poses_used", d.get("n_poses", 0))
        n_total = d.get("n_poses_total", n_used)
        role = "eye-to-hand" if d["configuration"] == "eye_to_hand" else "eye-in-hand"
        name = d["transform_name"]
        print(f"  cam_{i} ({role}, {name})")
        print(f"      position (mm)         : ({pos[0]:+7.1f}, {pos[1]:+7.1f}, {pos[2]:+7.1f})")
        print(f"      poses retenues        : {n_used}/{n_total}")
        print(f"      residu translation    : moyen {res['translation_mean_dev_mm']:5.2f} mm | "
              f"max {res['translation_max_dev_mm']:5.2f} mm")
        print(f"      residu rotation       : moyen {res['rotation_mean_dev_deg']:5.2f} deg | "
              f"max {res['rotation_max_dev_deg']:5.2f} deg")
        # Verdict
        t_max = res["translation_max_dev_mm"]
        r_max = res["rotation_max_dev_deg"]
        if t_max < 10 and r_max < 1.5:
            print(f"      -> BON (utilisable pour le pipeline)")
        elif t_max < 20 and r_max < 4:
            print(f"      -> ACCEPTABLE (au plancher de bruit du SO-101)")
        else:
            print(f"      -> A REVOIR (residus trop eleves)")
            all_ok = False
    return all_ok, cams


def check_stereo_baseline(cams):
    section("COHERENCE STEREO (cam_0 <-> cam_1)")
    if 0 not in cams or 1 not in cams:
        warn("cam_0 ou cam_1 manquante, baseline non calculee")
        return
    T0 = np.array(cams[0]["transform"])
    T1 = np.array(cams[1]["transform"])
    p0 = T0[:3, 3] * 1000.0
    p1 = T1[:3, 3] * 1000.0
    baseline_mm = float(np.linalg.norm(p1 - p0))
    delta = p1 - p0
    # Angle entre les axes optiques (Z des cameras)
    z0 = T0[:3, 2]  # axe Z = direction optique
    z1 = T1[:3, 2]
    cos = float(np.clip(np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1)), -1, 1))
    angle_optiques = math.degrees(math.acos(cos))
    print(f"  cam_0 -> cam_1 : delta = ({delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}) mm")
    print(f"  Baseline stereo : {baseline_mm:.1f} mm")
    print(f"  Angle entre axes optiques : {angle_optiques:.1f} deg")
    if 60 < baseline_mm < 250:
        ok("baseline plausible pour configuration stereo")
    else:
        warn(f"baseline = {baseline_mm:.1f} mm (hors plage typique 60-250 mm)")
    if angle_optiques < 30:
        ok("axes optiques quasi-paralleles (convergence faible)")
    else:
        warn(f"angle entre axes optiques = {angle_optiques:.1f} deg (assez convergent)")


def check_chain():
    section("CHAINE CINEMATIQUE (URDF + FK + motor_to_angle)")
    for module in ["src.utils.transforms",
                   "src.calibration.motor_to_angle",
                   "src.calibration.forward_kinematics"]:
        result = subprocess.run(
            [sys.executable,"-m", module],
            capture_output=True, text=True, check=False, cwd=str(REPO),
        )
        last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if "Tous les tests passent" in last or "tous les tests passent" in last:
            ok(module, "self-test passe")
        else:
            warn(module, f"self-test echoue : {last}")
            print(result.stdout)
            print(result.stderr)


def main():
    print()
    print("VALIDATION DE LA CALIBRATION COMPLETE - TFE LeRobot SO-101")
    print()

    intr_ok = check_intrinsics()
    motor_ok = check_motor()
    extr_ok, cams = check_extrinsics()
    check_stereo_baseline(cams)
    check_chain()

    section("BILAN")
    if intr_ok and motor_ok and extr_ok:
        print("  [OK] Toute la chaine de calibration est validee.")
        print("       Tu peux passer a la suite : perception + planification + controle.")
    else:
        print("  [!] Au moins une etape demande revision (voir details ci-dessus).")
    print()


if __name__ == "__main__":
    main()
