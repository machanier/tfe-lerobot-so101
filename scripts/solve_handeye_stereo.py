#!/usr/bin/env python3
"""Resolution hand-eye stereo conjointe des deux cameras.

Prend en entree le fichier JSON stereo produit par
calibrate_extrinsic_stereo.py (qui contient les paires d'images synchronisees
du damier) et enchaine les etapes suivantes :

  1. Calcule T_cam0_cam1 via cv2.stereoCalibrate() avec l'option
     CALIB_FIX_INTRINSIC (les matrices K et coefficients D calibres au prealable
     sont figes). Cette optimisation conjointe sur les memes poses du damier
     fournit une precision de l'ordre du demi-millimetre sur la transformation
     entre les deux cameras.

  2. Resout le hand-eye eye-to-hand pour cam_0 independamment -> T_base_cam0.

  3. Deduit T_base_cam1 = T_base_cam0 @ T_cam0_cam1.
     Avec cette methode, les deux calibrations sont coherentes par
     construction : si T_base_cam0 comporte un biais geometrique, T_base_cam1
     porte le meme biais, qui s'annule lors de la triangulation stereo. La
     difference entre les deux cameras reste correcte par construction.

  4. Calcule les residus :
       - cam_0 : residus hand-eye classiques
       - cam_1 : meme metrique, avec T_base_cam1 deduit
       - stereo : RMS de cv2.stereoCalibrate (en pixels)

  5. Sauvegarde handeye_cam_0.json et handeye_cam_1.json au format attendu par
     le reste de la pipeline, ainsi que handeye_stereo_info.json (T_cam0_cam1).

References : Hartley & Zisserman 2018 ch.10 (calibration stereo), Tsai & Lenz
1989 (hand-eye), Zhang 2000 (calibration de camera).

Usage :
    python scripts/solve_handeye_stereo.py
    python scripts/solve_handeye_stereo.py --capture-file configs/extrinsic_capture_stereo.json

Entrees : le JSON de captures stereo et les fichiers intrinseques references.
Sorties : configs/handeye_cam_<idx>.json (x2) et configs/handeye_stereo_info.json.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import CAMERAS  # noqa: E402
from src.calibration import handeye  # noqa: E402
from src.calibration.forward_kinematics import ARM_JOINTS, KinematicChain  # noqa: E402
from src.calibration.motor_to_angle import (  # noqa: E402
    load_encoder_unwrap, load_motor_calibration, raw_to_radians,
)
from src.utils.transforms import rvec_tvec_to_matrix  # noqa: E402


def fmt_mat(T, indent="    "):
    return "\n".join(indent + "  ".join(f"{v:+10.5f}" for v in row) for row in T)


def verdict(mean_mm, max_mm):
    if max_mm < 5 and mean_mm < 2:
        return "EXCELLENT"
    if max_mm < 12 and mean_mm < 5:
        return "OK (au plancher SO-101)"
    if max_mm < 20 and mean_mm < 8:
        return "ACCEPTABLE"
    return "INSUFFISANT"


def build_g2b_t2c(captures, cam_suffix, calib_motors, unwrap_centers, chain):
    """Construit les listes T_g2b et T_t2c utilisees par le solveur hand-eye
    eye-to-hand, pour cam_0 (suffix='cam0') ou cam_1 (suffix='cam1')."""
    T_g2b_list, T_t2c_list = [], []
    for cap in captures:
        raw = cap["motor_positions_raw"]
        q = {
            j: raw_to_radians(raw[j], calib_motors[j], unwrap_centers.get(j))
            for j in ARM_JOINTS
        }
        T_g2b_list.append(chain.fk(q))
        rvec = np.asarray(cap[f"rvec_target_{cam_suffix}"], dtype=np.float64).reshape(3)
        tvec_mm = np.asarray(cap[f"tvec_target_{cam_suffix}"], dtype=np.float64).reshape(3)
        T_t2c_list.append(rvec_tvec_to_matrix(rvec, tvec_mm / 1000.0))
    return T_g2b_list, T_t2c_list


def write_handeye_json(out_path, cam_key, cam_index, T_base_cam, captures,
                       used_indices, stats, method_label, source_info):
    out = {
        "camera_key": cam_key,
        "camera_index": cam_index,
        "configuration": "eye_to_hand",
        "method": method_label,
        "robust": True,
        "transform_name": "T_base_cam",
        "transform": T_base_cam.tolist(),
        "n_poses_total": len(captures),
        "n_poses_used": len(used_indices),
        "used_capture_ids": [int(captures[i]["id"]) for i in used_indices],
        "residuals": stats,
        "capture_file": source_info,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Resolution hand-eye stereo conjointe de cam_0 et cam_1."
    )
    parser.add_argument("--capture-file", default="configs/extrinsic_capture_stereo.json",
                        help="Fichier JSON de captures stereo produit par "
                             "calibrate_extrinsic_stereo.py "
                             "(defaut : configs/extrinsic_capture_stereo.json).")
    parser.add_argument("--method", choices=list(handeye.METHODS), default="HORAUD",
                        help="Methode passee a cv2.calibrateHandEye "
                             "(defaut : HORAUD).")
    parser.add_argument("--naive", action="store_true",
                        help="Desactive le mode robuste du solveur hand-eye "
                             "(defaut : mode robuste actif).")
    args = parser.parse_args()

    cap_path = REPO / args.capture_file
    if not cap_path.exists():
        print(f"ERREUR : {cap_path} introuvable. Lance d'abord calibrate_extrinsic_stereo.py")
        sys.exit(1)

    data = json.load(open(cap_path))
    captures = data["captures"]
    idx_l, idx_r = data["cam_indices"]
    cam_l_key, cam_r_key = data["cam_keys"]
    intr_l_path = REPO / data["intrinsic_files"][0]
    intr_r_path = REPO / data["intrinsic_files"][1]
    img_size_l = tuple(data["image_size_left"])
    img_size_r = tuple(data["image_size_right"])
    cb = data["checkerboard"]

    print("=" * 70)
    print(f" SOLVE HAND-EYE STEREO  {cam_l_key} (idx {idx_l}) + {cam_r_key} (idx {idx_r})")
    print("=" * 70)
    print(f"  Captures        : {len(captures)}")
    print(f"  Damier          : {cb['rows']}x{cb['cols']} @ {cb['square_size_mm']}mm")
    print(f"  Image size left : {img_size_l}")
    print(f"  Image size right: {img_size_r}")
    print()

    if len(captures) < 8:
        print(f"[WARN] Seulement {len(captures)} captures. stereoCalibrate "
              f"recommande 10+. Resultat fragile.")

    # --- Charge intrinseques ---
    int_l = json.load(open(intr_l_path))
    int_r = json.load(open(intr_r_path))
    K_l = np.array(int_l["camera_matrix"])
    D_l = np.array(int_l["dist_coeffs"]).reshape(-1)
    K_r = np.array(int_r["camera_matrix"])
    D_r = np.array(int_r["dist_coeffs"]).reshape(-1)

    # =====================================================================
    # PHASE 1 : cv2.stereoCalibrate -> T_cam0_cam1 (transformation entre cams)
    # =====================================================================
    print("PHASE 1 : Calibration stereo conjointe (cv2.stereoCalibrate)")
    obj_pts_list = [np.array(c["obj_points"], dtype=np.float32) for c in captures]
    img_pts_l = [np.array(c["img_points_cam0"], dtype=np.float32).reshape(-1, 1, 2)
                 for c in captures]
    img_pts_r = [np.array(c["img_points_cam1"], dtype=np.float32).reshape(-1, 1, 2)
                 for c in captures]

    # Les obj_points sont en mm, donc T sera en mm ; conversion en m ensuite.
    flags = cv2.CALIB_FIX_INTRINSIC  # K et D deja calibres : on les fige
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    rms_stereo, _, _, _, _, R_l2r, t_l2r, E, F = cv2.stereoCalibrate(
        obj_pts_list, img_pts_l, img_pts_r,
        K_l, D_l, K_r, D_r,
        img_size_l,
        flags=flags, criteria=criteria,
    )

    # cv2.stereoCalibrate retourne (R_l2r, t_l2r) tels que
    # P_right = R @ P_left + T. Autrement dit, (R, T) forment T_right_left
    # (pose de cam_left dans le repere cam_right). Pour obtenir T_cam0_cam1
    # (pose de cam_1 dans le repere cam_0), il faut inverser cette
    # transformation : T_cam0_cam1 = inverse(T_right_left) = [R.T  -R.T@T; 0 1].
    # Prendre directement [R T; 0 1] reviendrait a stocker T_cam1_cam0 et
    # placerait cam_1 du mauvais cote (delta Y inverse) lors de la composition.
    R_cam0_to_cam1 = R_l2r.T
    t_cam0_to_cam1_in_cam0_mm = (-R_l2r.T @ t_l2r).flatten()
    T_cam0_cam1 = np.eye(4)
    T_cam0_cam1[:3, :3] = R_cam0_to_cam1
    T_cam0_cam1[:3, 3] = t_cam0_to_cam1_in_cam0_mm / 1000.0  # mm -> m

    # Baseline : norme de la translation (independante du sens).
    baseline_mm = float(np.linalg.norm(t_l2r))
    # Angle entre les axes optiques z (apres rotation).
    axis_l = np.array([0, 0, 1.0])
    axis_r_in_l = R_l2r.T @ axis_l  # axe z de cam_1 exprime dans cam_0
    angle_axes_deg = np.degrees(np.arccos(np.clip(axis_l @ axis_r_in_l, -1, 1)))

    print(f"  RMS reprojection stereo : {rms_stereo:.3f} px  "
          f"({'bon (<0.5)' if rms_stereo < 0.5 else 'eleve, captures a verifier' if rms_stereo > 1 else 'OK'})")
    print(f"  Baseline cam_0 -> cam_1 : {baseline_mm:.1f} mm")
    print(f"  Angle entre axes optiques z : {angle_axes_deg:.1f} deg")
    print(f"  T_cam0_cam1 (translation cam_0 -> cam_1 dans cam_0, en mm) : "
          f"({t_cam0_to_cam1_in_cam0_mm[0]:+.1f}, "
          f"{t_cam0_to_cam1_in_cam0_mm[1]:+.1f}, "
          f"{t_cam0_to_cam1_in_cam0_mm[2]:+.1f})")
    print()

    # =====================================================================
    # PHASE 2 : Hand-eye cam_0 (eye-to-hand classique)
    # =====================================================================
    print("PHASE 2 : Hand-eye eye-to-hand sur cam_0 (independant)")
    calib_motors = load_motor_calibration(REPO / "configs/calibration_follower.json")
    unwrap_centers = load_encoder_unwrap(REPO / "configs/encoder_unwrap.json", calib_motors)
    chain = KinematicChain()

    T_g2b_list, T_t2c_l_list = build_g2b_t2c(captures, "cam0", calib_motors, unwrap_centers, chain)
    R_g2b = [T[:3, :3] for T in T_g2b_list]
    t_g2b = [T[:3, 3] for T in T_g2b_list]
    R_t2c_l = [T[:3, :3] for T in T_t2c_l_list]
    t_t2c_l = [T[:3, 3] for T in T_t2c_l_list]

    if not args.naive:
        corrections = handeye.symmetric_board_corrections(
            cb["square_size_mm"] / 1000.0, cb["rows"], cb["cols"]
        )
        T_base_cam0, used_idx_0, stats_0 = handeye.solve_eye_to_hand_robust(
            R_g2b, t_g2b, R_t2c_l, t_t2c_l,
            corrections=corrections,
            method=handeye.METHODS[args.method],
        )
    else:
        T_base_cam0 = handeye.solve_eye_to_hand(
            R_g2b, t_g2b, R_t2c_l, t_t2c_l, method=handeye.METHODS[args.method]
        )
        stats_0 = handeye.residuals_eye_to_hand(T_g2b_list, T_t2c_l_list, T_base_cam0)
        used_idx_0 = list(range(len(captures)))

    print(f"  T_base_cam0 position (mm) : "
          f"({T_base_cam0[0,3]*1000:+.1f}, {T_base_cam0[1,3]*1000:+.1f}, "
          f"{T_base_cam0[2,3]*1000:+.1f})")
    print(f"  Residus cam_0 : mean={stats_0['translation_mean_dev_mm']:.2f}mm  "
          f"max={stats_0['translation_max_dev_mm']:.2f}mm  "
          f"({len(used_idx_0)}/{len(captures)} poses retenues)")
    print(f"  Verdict cam_0 : {verdict(stats_0['translation_mean_dev_mm'], stats_0['translation_max_dev_mm'])}")
    print()

    # =====================================================================
    # PHASE 3 : Deduction T_base_cam1 = T_base_cam0 @ T_cam0_cam1
    # =====================================================================
    print("PHASE 3 : Deduction T_base_cam1 (= T_base_cam0 @ T_cam0_cam1)")
    T_base_cam1 = T_base_cam0 @ T_cam0_cam1
    print(f"  T_base_cam1 position (mm) : "
          f"({T_base_cam1[0,3]*1000:+.1f}, {T_base_cam1[1,3]*1000:+.1f}, "
          f"{T_base_cam1[2,3]*1000:+.1f})")

    # Residus pour cam_1 : avec T_base_cam1 deduit, on evalue l'erreur de
    # prediction de la cible dans cam_1, sur les memes poses.
    _, T_t2c_r_list = build_g2b_t2c(captures, "cam1", calib_motors, unwrap_centers, chain)
    stats_1 = handeye.residuals_eye_to_hand(T_g2b_list, T_t2c_r_list, T_base_cam1)
    print(f"  Residus cam_1 : mean={stats_1['translation_mean_dev_mm']:.2f}mm  "
          f"max={stats_1['translation_max_dev_mm']:.2f}mm  "
          f"(memes {len(captures)} poses)")
    print(f"  Verdict cam_1 : {verdict(stats_1['translation_mean_dev_mm'], stats_1['translation_max_dev_mm'])}")
    print()

    # =====================================================================
    # PHASE 4 : Sauvegarde
    # =====================================================================
    print("PHASE 4 : Sauvegarde")
    out_l = REPO / f"configs/handeye_cam_{idx_l}.json"
    out_r = REPO / f"configs/handeye_cam_{idx_r}.json"
    source_info = str(cap_path.relative_to(REPO))

    write_handeye_json(out_l, cam_l_key, idx_l, T_base_cam0, captures,
                       used_idx_0, stats_0,
                       method_label=f"{args.method}_STEREO_INDEPENDENT",
                       source_info=source_info)
    print(f"  -> {out_l.name}")

    # Pour cam_1, "used_capture_ids" liste toutes les captures : la pose est
    # deduite de cam_0, non resolue independamment.
    write_handeye_json(out_r, cam_r_key, idx_r, T_base_cam1, captures,
                       list(range(len(captures))), stats_1,
                       method_label=f"{args.method}_STEREO_DEDUCED_FROM_CAM0",
                       source_info=source_info)
    print(f"  -> {out_r.name}")

    # Sauvegarde de T_cam0_cam1 et du RMS stereo, pour tracabilite.
    info_path = REPO / "configs/handeye_stereo_info.json"
    info = {
        "schema_version": "stereo_v1.1",
        "cam_indices": [idx_l, idx_r],
        "cam_keys": [cam_l_key, cam_r_key],
        "T_cam0_cam1": T_cam0_cam1.tolist(),
        "T_cam0_cam1_translation_mm": t_cam0_to_cam1_in_cam0_mm.tolist(),
        "stereo_R_raw_opencv": R_l2r.tolist(),
        "stereo_T_raw_opencv_mm": t_l2r.flatten().tolist(),
        "_doc_opencv": (
            "OpenCV stereoCalibrate retourne (R, T) tels que P_right = R @ P_left + T. "
            "Ce sont donc T_right_left. Pour la composition T_base_cam1 = T_base_cam0 @ "
            "T_cam0_cam1, on prend T_cam0_cam1 = [R.T, -R.T@T; 0 1] (inverse)."
        ),
        "stereo_rms_reprojection_px": float(rms_stereo),
        "baseline_mm": float(baseline_mm),
        "angle_optical_axes_deg": float(angle_axes_deg),
        "source_capture_file": source_info,
        "n_poses_used_for_stereo": len(captures),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  -> {info_path.name} (T_cam0_cam1 + RMS stereo)")
    print()

    # =====================================================================
    # Recapitulatif
    # =====================================================================
    print("=" * 70)
    print(" RECAPITULATIF")
    print("=" * 70)
    print(f"  Stereo RMS reproj : {rms_stereo:.3f} px  (baseline {baseline_mm:.1f}mm)")
    print(f"  cam_0 hand-eye    : mean={stats_0['translation_mean_dev_mm']:.2f}mm  "
          f"max={stats_0['translation_max_dev_mm']:.2f}mm  "
          f"-> {verdict(stats_0['translation_mean_dev_mm'], stats_0['translation_max_dev_mm'])}")
    print(f"  cam_1 (deduit)    : mean={stats_1['translation_mean_dev_mm']:.2f}mm  "
          f"max={stats_1['translation_max_dev_mm']:.2f}mm  "
          f"-> {verdict(stats_1['translation_mean_dev_mm'], stats_1['translation_max_dev_mm'])}")
    print()
    print(f"  Prochaine etape : verifier en pratique avec")
    print(f"    python scripts/check_calibration.py")
    print(f"    python scripts/pick_and_place.py --target orange_cube --detector hf --display")
    print(f"  Une correction Y de l'ordre de +30 a +40 mm indique un residu encore eleve.")
    print(f"  Une correction Y inferieure a 10 mm indique une calibration satisfaisante.")


if __name__ == "__main__":
    main()
