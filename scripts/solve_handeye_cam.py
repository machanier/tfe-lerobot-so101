#!/usr/bin/env python3
"""
solve_handeye_cam.py - Resout la calibration hand-eye d'une camera.

Charge la capture extrinseque (configs/extrinsic_capture_cam_<i>.json),
calcule pour chaque pose T_base_gripper via la cinematique directe et
T_target_cam via le rvec/tvec stocke, puis resout :
  - eye-to-hand (cam_0, cam_1) : sortie = T_base_cam
  - eye-in-hand (cam_2)         : sortie = T_gripper_cam

Sauvegarde le resultat dans configs/handeye_cam_<i>.json avec les residus.

Usage:
    python scripts/solve_handeye_cam.py --index 0
    python scripts/solve_handeye_cam.py --index 0 --method PARK
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import CAMERAS  # noqa: E402
from src.calibration import handeye  # noqa: E402
from src.calibration.forward_kinematics import ARM_JOINTS, KinematicChain  # noqa: E402
from src.calibration.motor_to_angle import (  # noqa: E402
    load_encoder_unwrap,
    load_motor_calibration,
    raw_to_radians,
)
from src.utils.transforms import rvec_tvec_to_matrix  # noqa: E402

EYE_TO_HAND_ROLES = {"stereo_left", "stereo_right"}
EYE_IN_HAND_ROLES = {"eye_in_hand"}


def role_for_camera(cam_key):
    """Recupere le role 'stereo_*' ou 'eye_in_hand' depuis config.CAMERAS."""
    if cam_key in CAMERAS:
        return CAMERAS[cam_key]["role"]
    # fallback : best guess sur l'index
    return None


def build_pose_lists(captures, calib_motors, unwrap_centers, chain):
    """Pour chaque capture, construit T_gripper2base (FK) et T_target2cam (PnP).

    Returns:
        T_g2b_list, T_t2c_list : listes de matrices 4x4 en METRES.
    """
    T_g2b_list = []
    T_t2c_list = []
    for cap in captures:
        # Angles articulaires depuis les positions moteur brutes
        raw = cap["motor_positions_raw"]
        q = {
            j: raw_to_radians(raw[j], calib_motors[j], unwrap_centers.get(j))
            for j in ARM_JOINTS
        }
        T_g2b_list.append(chain.fk(q))

        # rvec/tvec stockes en mm (cf square_size_mm du damier) -> convertir en m
        rvec = np.asarray(cap["rvec_target_cam"], dtype=np.float64).reshape(3)
        tvec_mm = np.asarray(cap["tvec_target_cam"], dtype=np.float64).reshape(3)
        T_t2c = rvec_tvec_to_matrix(rvec, tvec_mm / 1000.0)
        T_t2c_list.append(T_t2c)

    return T_g2b_list, T_t2c_list


def format_4x4(T):
    """Joliment formate une matrice 4x4 pour affichage."""
    return "\n".join(
        "    " + "  ".join(f"{v:+10.5f}" for v in row) for row in T
    )


def verdict(stats):
    """Donne un verdict qualitatif d'apres les residus."""
    t_max, r_max = stats["translation_max_dev_mm"], stats["rotation_max_dev_deg"]
    if t_max < 5 and r_max < 0.5:
        return "EXCELLENT (calibration tres precise)"
    if t_max < 15 and r_max < 1.5:
        return "BON (utilisable pour le pipeline)"
    if t_max < 40 and r_max < 4:
        return "ACCEPTABLE (a verifier en pratique)"
    return "DEGRADE (residus eleves : a investiguer)"


def main():
    parser = argparse.ArgumentParser(description="Resout la calibration hand-eye d'une camera")
    parser.add_argument("--index", type=int, required=True, help="Index camera (0, 1 ou 2)")
    parser.add_argument(
        "--method", choices=list(handeye.METHODS), default="HORAUD",
        help="Methode cv2.calibrateHandEye (defaut: HORAUD, robuste)"
    )
    parser.add_argument(
        "--naive", action="store_true",
        help="Desactive le mode robuste (alignement damier symetrique + rejet "
             "d'outliers). Utile pour diagnostiquer."
    )
    args = parser.parse_args()

    cap_path = REPO_ROOT / "configs" / f"extrinsic_capture_cam_{args.index}.json"
    calib_motors_path = REPO_ROOT / "configs" / "calibration_follower.json"
    unwrap_path = REPO_ROOT / "configs" / "encoder_unwrap.json"
    out_path = REPO_ROOT / "configs" / f"handeye_cam_{args.index}.json"

    if not cap_path.exists():
        print(f"ERREUR : {cap_path} introuvable.")
        sys.exit(1)

    data = json.load(open(cap_path))
    captures = data["captures"]
    cam_key = data.get("camera_key", f"cam_{args.index}")
    role = role_for_camera(cam_key)

    if role in EYE_TO_HAND_ROLES:
        config_type = "eye_to_hand"
    elif role in EYE_IN_HAND_ROLES:
        config_type = "eye_in_hand"
    else:
        print(f"ERREUR : role inconnu pour {cam_key} (role={role}).")
        sys.exit(1)

    print(f"Capture : configs/{cap_path.name}  ({cam_key}, role={role})")
    print(f"Configuration : {config_type}")
    print(f"Methode : {args.method}{' (naive)' if args.naive else ' (robuste)'}")
    print(f"Poses : {len(captures)}")
    print()

    # Donnees auxiliaires
    calib_motors = load_motor_calibration(calib_motors_path)
    unwrap_centers = load_encoder_unwrap(unwrap_path, calib_motors)
    chain = KinematicChain()  # utilise configs/so101_new_calib.urdf

    # Listes T_g2b, T_t2c
    T_g2b_list, T_t2c_list = build_pose_lists(captures, calib_motors, unwrap_centers, chain)
    R_g2b = [T[:3, :3] for T in T_g2b_list]
    t_g2b = [T[:3, 3] for T in T_g2b_list]
    R_t2c = [T[:3, :3] for T in T_t2c_list]
    t_t2c = [T[:3, 3] for T in T_t2c_list]

    used_indices = list(range(len(captures)))

    if config_type == "eye_to_hand" and not args.naive:
        # Mode robuste : gere l'ambiguite de detection des damiers symetriques
        # (rotation + decalage d'origine, jusqu'a 4 orientations pour 7x7) +
        # rejet iteratif d'outliers.
        cb = data["checkerboard"]
        corrections = handeye.symmetric_board_corrections(
            cb["square_size_mm"] / 1000.0, cb["rows"], cb["cols"]
        )
        print(f"Mode robuste : {len(corrections)} orientations possibles "
              f"pour le damier {cb['rows']}x{cb['cols']} ({cb['square_size_mm']} mm)")
        T_solved, used_indices, stats = handeye.solve_eye_to_hand_robust(
            R_g2b, t_g2b, R_t2c, t_t2c,
            corrections=corrections,
            method=handeye.METHODS[args.method],
        )
        transform_name = "T_base_cam"
        print(f"  {len(used_indices)} / {len(captures)} poses retenues apres "
              f"alignement + rejet d'outliers")
        print()
    else:
        if config_type == "eye_to_hand":
            T_solved = handeye.solve_eye_to_hand(
                R_g2b, t_g2b, R_t2c, t_t2c, method=handeye.METHODS[args.method]
            )
            stats = handeye.residuals_eye_to_hand(T_g2b_list, T_t2c_list, T_solved)
        else:
            T_solved = handeye.solve_eye_in_hand(
                R_g2b, t_g2b, R_t2c, t_t2c, method=handeye.METHODS[args.method]
            )
            stats = handeye.residuals_eye_in_hand(T_g2b_list, T_t2c_list, T_solved)
        transform_name = "T_base_cam" if config_type == "eye_to_hand" else "T_gripper_cam"

    print(f"--- Resultat ({args.method}) ---")
    print(f"  {transform_name} = ")
    print(format_4x4(T_solved))
    print(f"  position (mm) : "
          f"({T_solved[0, 3] * 1000:+.1f}, {T_solved[1, 3] * 1000:+.1f}, "
          f"{T_solved[2, 3] * 1000:+.1f})")
    print(f"  Residus {stats['label']} sur {stats['n_poses']} poses :")
    print(f"    translation : moyenne {stats['translation_mean_dev_mm']:.2f} mm  |  "
          f"max {stats['translation_max_dev_mm']:.2f} mm  |  "
          f"mediane {stats['translation_median_dev_mm']:.2f} mm")
    print(f"    rotation    : moyenne {stats['rotation_mean_dev_deg']:.3f} deg  |  "
          f"max {stats['rotation_max_dev_deg']:.3f} deg  |  "
          f"mediane {stats['rotation_median_dev_deg']:.3f} deg")
    print(f"  -> {verdict(stats)}")
    print()

    out = {
        "camera_key": cam_key,
        "camera_index": args.index,
        "configuration": config_type,
        "method": args.method,
        "robust": (config_type == "eye_to_hand" and not args.naive),
        "transform_name": transform_name,
        "transform": T_solved.tolist(),
        "n_poses_total": len(captures),
        "n_poses_used": len(used_indices),
        "used_capture_ids": [int(captures[i]["id"]) for i in used_indices],
        "residuals": stats,
        "capture_file": str(cap_path.relative_to(REPO_ROOT)),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Resultat sauvegarde : configs/{out_path.name}")


if __name__ == "__main__":
    main()
