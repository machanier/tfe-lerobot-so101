#!/usr/bin/env python3
"""
calibrate_extrinsic_stereo.py - Capture SIMULTANEE cam_0 + cam_1 pour
calibration hand-eye stereo conjointe (B3b).

POURQUOI :
La calibration separee de cam_0 et cam_1 (l'approche actuelle) produit des
residus independants (~6mm chacun) qui peuvent s'additionner geometriquement
lors de la triangulation -> biais Y +30-40mm constaté empiriquement.
Calibrer les 2 cameras conjointement (sur les MEMES poses du damier au
MEME instant) permet d'ajouter une contrainte forte : cv2.stereoCalibrate()
calcule T_cam0_cam1 avec une precision sub-millimetrique. On en deduit
ensuite T_base_cam1 = T_base_cam0 @ T_cam0_cam1 : les deux calibrations
sont coherentes par construction, le biais s'annule a la triangulation.

Reference : Hartley & Zisserman, Multiple View Geometry, 2018, ch.10.

USAGE :
    python scripts/calibrate_extrinsic_stereo.py
    python scripts/calibrate_extrinsic_stereo.py --rows 6 --cols 9 --square-size 22

PROCEDURE :
  1. Damier (9x6 asymetrique par defaut) COLLE sur la pince FERMEE du robot.
  2. Les 2 cameras sont fixes sur la barriere avant.
  3. Bouge le BRAS dans 30-60 poses variees (>65 deg de diversite angulaire,
     distances 30-80 cm). A chaque pose stable, appuie 'c' pour capturer.
  4. Pour CAPTURER, le damier doit etre detecte DANS LES DEUX cameras
     simultanement. Si une seule le voit, la capture est skip avec un message.
  5. 'q' pour terminer et sauvegarder.

SORTIE :
  configs/extrinsic_capture_stereo.json : paires synchronisees rvec/tvec/img_points
                                            + motor_positions.

Etape suivante : python scripts/solve_handeye_stereo.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from config import CAMERAS, FOLLOWER_PORT  # noqa: E402


def load_intrinsic(path: str):
    data = json.load(open(path))
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


def open_camera(index: int, width: int = 1920, height: int = 1080):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} introuvable")
    # MJPG pour reduire la bande passante USB (cf D10)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Warmup : autoexposure
    for _ in range(5):
        cap.read()
    return cap


def connect_robot(port: str):
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as e:
        raise ImportError("LeRobot indisponible.") from e
    motors = {
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
    bus.connect()
    motor_names = list(motors.keys())
    return bus, motor_names


def estimate_board_pose(frame, K, D, rows, cols, square_size_mm):
    """Detecte le damier et calcule rvec/tvec en mm. Renvoie aussi les corners 2D."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None, None, None
    # Sub-pixel refinement (precision)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Object points en mm (damier plat z=0)
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None, None, None
    return rvec, tvec, corners, obj


def draw_overlay(frame, K, D, rvec, tvec, corners, square_size_mm,
                 cam_label, status_text, rows, cols):
    display = frame.copy()
    if rvec is not None:
        # BUG fix 2026-05-19 21h : drawChessboardCorners attend (cols, rows)
        # du DAMIER, pas la taille de l'image en pixels. Le passage de
        # (1920, 1080) faisait dessiner 2M de coins -> traits partout + FPS=0.
        cv2.drawChessboardCorners(display, (cols, rows), corners, True)
        axis_len = square_size_mm * 3
        axis_pts = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]])
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
        origin = tuple(corners[0].ravel().astype(int))
        cv2.line(display, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
        cv2.line(display, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
        cv2.line(display, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)
    # Bandeau noir + label cam
    cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(display, cam_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    color = (0, 255, 0) if rvec is not None else (0, 0, 255)
    cv2.putText(display, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1)
    return display


def main():
    parser = argparse.ArgumentParser(
        description="Capture stereo SIMULTANEE cam_0 + cam_1 pour hand-eye stereo (B3b)."
    )
    parser.add_argument("--cam-indices", nargs=2, type=int, default=[0, 1],
                        metavar=("LEFT", "RIGHT"),
                        help="Indices OpenCV des 2 cameras stereo (defaut: 0 1).")
    parser.add_argument("--port", default=FOLLOWER_PORT, help="Port USB follower")
    parser.add_argument("--rows", type=int, default=6, help="Coins internes lignes")
    parser.add_argument("--cols", type=int, default=9, help="Coins internes colonnes")
    parser.add_argument("--square-size", type=float, default=22.0,
                        help="Taille case damier en mm")
    parser.add_argument("--output", default="configs/extrinsic_capture_stereo.json")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--display-scale", type=float, default=0.45,
                        help="Echelle d'affichage de la mosaic side-by-side (defaut 0.45)")
    args = parser.parse_args()

    idx_l, idx_r = args.cam_indices
    cam_l_key = next((k for k, v in CAMERAS.items() if v["index"] == idx_l), f"cam_{idx_l}")
    cam_r_key = next((k for k, v in CAMERAS.items() if v["index"] == idx_r), f"cam_{idx_r}")
    print()
    print("=" * 70)
    print(f" CALIBRATION EXTRINSEQUE STEREO  {cam_l_key} (idx {idx_l}) + {cam_r_key} (idx {idx_r})")
    print("=" * 70)
    print()
    print("  PROCEDURE EYE-TO-HAND STEREO :")
    print("    1. Le damier 9x6 22mm est COLLE sur la pince FERMEE du robot.")
    print("    2. Les 2 cameras sont FIXES sur la barriere avant.")
    print("    3. Bouge le BRAS pour amener le damier dans le champ des DEUX cameras.")
    print("    4. Quand les 2 voient le damier (status vert) : 'c' pour capturer.")
    print("    5. Diversite angulaire >65deg, 30-60 poses recommandees.")
    print("    6. 'q' pour terminer, ESC pour annuler.")
    print()

    # Charge intrinseques
    intr_l_path = f"configs/calibration_cam_{idx_l}.json"
    intr_r_path = f"configs/calibration_cam_{idx_r}.json"
    if not os.path.exists(intr_l_path) or not os.path.exists(intr_r_path):
        print(f"ERREUR : calibrations intrinseques manquantes : {intr_l_path}, {intr_r_path}")
        sys.exit(1)
    K_l, D_l = load_intrinsic(intr_l_path)
    K_r, D_r = load_intrinsic(intr_r_path)
    print(f"Intrinseques {cam_l_key} : fx={K_l[0,0]:.1f}, residu inclus dans le fichier.")
    print(f"Intrinseques {cam_r_key} : fx={K_r[0,0]:.1f}.")
    print()

    # Connect robot
    print(f"Connexion au follower sur {args.port}...")
    bus, motor_names = connect_robot(args.port)
    print(f"  6 moteurs detectes, torque desactive (bras manipulable a la main).")
    print()

    # Ouvre les 2 cameras
    w_l = CAMERAS.get(cam_l_key, {}).get("width", 1920)
    h_l = CAMERAS.get(cam_l_key, {}).get("height", 1080)
    w_r = CAMERAS.get(cam_r_key, {}).get("width", 1920)
    h_r = CAMERAS.get(cam_r_key, {}).get("height", 1080)
    print(f"Ouverture {cam_l_key} ({w_l}x{h_l})...")
    cap_l = open_camera(idx_l, w_l, h_l)
    print(f"Ouverture {cam_r_key} ({w_r}x{h_r})...")
    cap_r = open_camera(idx_r, w_r, h_r)

    # Dossiers images
    img_dir_l = REPO / f"outputs/calibration_images/extrinsic_stereo_{cam_l_key}"
    img_dir_r = REPO / f"outputs/calibration_images/extrinsic_stereo_{cam_r_key}"
    if not args.no_save_images:
        img_dir_l.mkdir(parents=True, exist_ok=True)
        img_dir_r.mkdir(parents=True, exist_ok=True)
        print(f"Images sauvegardees dans : {img_dir_l}/ et {img_dir_r}/")

    # Output path + helper
    output_path = REPO / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    captures = []
    img_size_l = (w_l, h_l)
    img_size_r = (w_r, h_r)

    def save_now():
        """Sauvegarde incrementale (cf fix calibrate_extrinsic 2026-05-19)."""
        result = {
            "schema_version": "stereo_v1",
            "cam_indices": [idx_l, idx_r],
            "cam_keys": [cam_l_key, cam_r_key],
            "intrinsic_files": [intr_l_path, intr_r_path],
            "motor_calibration_file": "configs/calibration_follower.json",
            "checkerboard": {
                "rows": args.rows,
                "cols": args.cols,
                "square_size_mm": args.square_size,
            },
            "motor_names": motor_names,
            "motor_position_units": "raw_encoder_counts",
            "image_size_left": list(img_size_l),
            "image_size_right": list(img_size_r),
            "num_captures": len(captures),
            "captures": captures,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print()
    print("Controles : 'c'=capturer (les 2 detectent), 'q'=terminer, ESC=annuler")
    print()

    window_name = f"Stereo extrinsec - {cam_l_key} | {cam_r_key}"

    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print("[WARN] echec lecture frame d'une camera, retry...")
                continue

            rvec_l, tvec_l, corners_l, obj_l = estimate_board_pose(
                frame_l, K_l, D_l, args.rows, args.cols, args.square_size)
            rvec_r, tvec_r, corners_r, obj_r = estimate_board_pose(
                frame_r, K_r, D_r, args.rows, args.cols, args.square_size)

            both_detected = (rvec_l is not None) and (rvec_r is not None)
            dist_l_mm = float(np.linalg.norm(tvec_l)) if rvec_l is not None else 0.0
            dist_r_mm = float(np.linalg.norm(tvec_r)) if rvec_r is not None else 0.0

            status_l = (f"OK dist={dist_l_mm:.0f}mm" if rvec_l is not None
                        else "Damier NON detecte")
            status_r = (f"OK dist={dist_r_mm:.0f}mm" if rvec_r is not None
                        else "Damier NON detecte")

            disp_l = draw_overlay(frame_l, K_l, D_l, rvec_l, tvec_l, corners_l,
                                   args.square_size, cam_l_key, status_l,
                                   args.rows, args.cols)
            disp_r = draw_overlay(frame_r, K_r, D_r, rvec_r, tvec_r, corners_r,
                                   args.square_size, cam_r_key, status_r,
                                   args.rows, args.cols)

            mosaic = np.hstack([disp_l, disp_r])
            # Bandeau central
            cv2.rectangle(mosaic, (0, mosaic.shape[0] - 50),
                          (mosaic.shape[1], mosaic.shape[0]), (0, 0, 0), -1)
            global_status = (f"CAPTURES: {len(captures)}  |  "
                             + ("Les 2 OK -> 'c' pour capturer"
                                if both_detected else "Repositionne pour que LES 2 voient le damier"))
            color = (0, 255, 0) if both_detected else (0, 165, 255)
            cv2.putText(mosaic, global_status,
                        (10, mosaic.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if args.display_scale != 1.0:
                ds = args.display_scale
                small = cv2.resize(mosaic, (int(mosaic.shape[1] * ds),
                                             int(mosaic.shape[0] * ds)))
                cv2.imshow(window_name, small)
            else:
                cv2.imshow(window_name, mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and both_detected:
                try:
                    motor_pos = bus.sync_read("Present_Position", normalize=False)
                except Exception as e:
                    print(f"  [WARN] lecture moteur echouee : {e}")
                    continue
                n = len(captures) + 1
                capture_data = {
                    "id": n,
                    "rvec_target_cam0": rvec_l.flatten().tolist(),
                    "tvec_target_cam0": tvec_l.flatten().tolist(),
                    "rvec_target_cam1": rvec_r.flatten().tolist(),
                    "tvec_target_cam1": tvec_r.flatten().tolist(),
                    "img_points_cam0": corners_l.reshape(-1, 2).tolist(),
                    "img_points_cam1": corners_r.reshape(-1, 2).tolist(),
                    "obj_points": obj_l.tolist(),  # meme damier pour les 2
                    "distance_mm_cam0": dist_l_mm,
                    "distance_mm_cam1": dist_r_mm,
                    "motor_positions_raw": {k: float(v) for k, v in motor_pos.items()},
                }
                captures.append(capture_data)
                if not args.no_save_images:
                    cv2.imwrite(str(img_dir_l / f"capture_{n:02d}_raw.png"), frame_l)
                    cv2.imwrite(str(img_dir_l / f"capture_{n:02d}_axes.png"), disp_l)
                    cv2.imwrite(str(img_dir_r / f"capture_{n:02d}_raw.png"), frame_r)
                    cv2.imwrite(str(img_dir_r / f"capture_{n:02d}_axes.png"), disp_r)
                # Sauvegarde incrementale du JSON apres chaque capture
                save_now()
                print(f"  Capture {n} : "
                      f"{cam_l_key} dist={dist_l_mm:.0f}mm, "
                      f"{cam_r_key} dist={dist_r_mm:.0f}mm")
            elif key == ord("c") and not both_detected:
                print(f"  [SKIP] le damier doit etre detecte dans LES DEUX cameras "
                      f"(actuellement L={rvec_l is not None}, R={rvec_r is not None})")
            elif key == ord("q"):
                break
            elif key == 27:
                print("Annule par utilisateur (ESC).")
                if captures:
                    save_now()
                    print(f"  {len(captures)} captures sauvees malgre l'annulation : {output_path}")
                break

    finally:
        try:
            cap_l.release()
        except Exception:
            pass
        try:
            cap_r.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # Sauvegarde finale AVANT bus.disconnect (cf fix 2026-05-19)
    if len(captures) >= 5:
        save_now()
        print()
        print(f"{len(captures)} captures sauvegardees : {output_path}")
        print(f"Etape suivante : python scripts/solve_handeye_stereo.py")
    elif captures:
        save_now()
        print(f"\nSeulement {len(captures)} captures (5 min recommande).")
        print(f"  Le JSON est sauve : {output_path}")
    else:
        print("\nAucune capture, rien a sauver.")

    # Disconnect dans try/except
    try:
        bus.disconnect()
    except Exception as e:
        print(f"  [WARN] bus.disconnect() : {e}")
        if captures:
            print(f"         OK, les captures sont sauvees dans {output_path}.")


if __name__ == "__main__":
    main()
