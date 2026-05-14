#!/usr/bin/env python3
"""
calibrate_extrinsic.py - Capture des donnees pour calibration hand-eye eye-to-hand.

Cible les cameras fixes (cam_0, cam_1) avec damier colle sur la pince fermee.
Pour cam_2 (eye-in-hand) : utiliser calibrate_extrinsic_eye_in_hand.py (a venir).

Usage :
    python scripts/calibrate_extrinsic.py --index 0
    python scripts/calibrate_extrinsic.py --index 1 --rows 7 --cols 7 --square-size 22

Procedure :
    1. Coller le damier rigide sur la pince fermee du robot
    2. Verifier que la camera est a sa position finale (structure assemblee)
    3. Lancer le script
    4. Bouger le bras dans 15-25 poses variees (rotations autour de >=2 axes)
    5. A chaque pose : attendre l'immobilisation puis 'c' pour capturer
    6. 'q' pour terminer et sauvegarder les donnees

Sortie :
    configs/extrinsic_capture_cam_<index>.json : poses damier-camera + angles moteurs
    outputs/calibration_images/extrinsic_cam_<index>/ : images annotees

La resolution hand-eye (cv2.calibrateHandEye) se fait ensuite via
scripts/solve_handeye_eye_to_hand.py (utilise la cinematique directe).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from config import CAMERAS, FOLLOWER_ID, FOLLOWER_PORT


def load_intrinsic(path):
    """Charge les parametres intrinseques depuis un fichier JSON."""
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    return camera_matrix, dist_coeffs


def connect_robot(port):
    """Connecte le bus moteur Feetech et retourne (bus, motor_names)."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print("ERREUR : LeRobot non installe ou non trouve.")
        print("  Active le venv : source venv/bin/activate")
        sys.exit(1)

    if not os.path.exists(port):
        print(f"ERREUR : port {port} introuvable.")
        import glob
        available = sorted(glob.glob("/dev/tty.usbmodem*"))
        if available:
            print("Ports usbmodem disponibles :")
            for p in available:
                print(f"  {p}")
            print("Utilise --port <chemin> ou mets a jour FOLLOWER_PORT dans scripts/config.py")
        else:
            print("Aucun port /dev/tty.usbmodem* detecte. Branche le robot.")
        sys.exit(1)

    motors = {
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)

    print(f"Connexion au follower sur {port}...")
    try:
        bus.connect()
    except RuntimeError as e:
        msg = str(e)
        if "Missing motor IDs" in msg or "Found: {}" in msg or "Full found motor list (id: model_number):\n{}" in msg:
            print("\nERREUR : aucun moteur ne repond sur ce port.")
            print("Causes les plus frequentes :")
            print("  1. Le bras follower n'est PAS alimente (verifie l'interrupteur / l'alimentation 5V).")
            print("  2. Le cable USB est branche mais le robot n'a pas de courant.")
            print("  3. Le port usbmodem correspond au LEADER, pas au FOLLOWER.")
            print(f"     Verifie avec : ls /dev/tty.usbmodem*")
            print("  4. Un autre processus retient le port (Arduino IDE, ancienne session, etc.)")
            sys.exit(1)
        raise

    bus.disable_torque()
    print("  6 moteurs detectes, torque desactive (bras manipulable a la main).")
    return bus, list(motors.keys())


def estimate_board_pose(frame, camera_matrix, dist_coeffs, rows, cols, square_size_mm):
    """Estime la pose du damier dans le repere camera (rvec, tvec)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not found:
        return None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    _, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
    return rvec, tvec, corners_refined


def main():
    parser = argparse.ArgumentParser(description="Capture des donnees pour calibration hand-eye")
    parser.add_argument("--index", type=int, required=True, help="Index OpenCV de la camera (0 ou 1)")
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT,
                        help=f"Port USB du follower (defaut config.py : {FOLLOWER_PORT})")
    parser.add_argument("--intrinsic", type=str, default=None,
                        help="Fichier intrinseque (defaut: configs/calibration_cam_<index>.json)")
    parser.add_argument("--rows", type=int, default=7, help="Coins internes du damier (lignes)")
    parser.add_argument("--cols", type=int, default=7, help="Coins internes du damier (colonnes)")
    parser.add_argument("--square-size", type=float, default=22.0, help="Taille des carres en mm")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cam_key = next((k for k, v in CAMERAS.items() if v["index"] == args.index), None)
    if cam_key is None:
        print(f"AVERTISSEMENT : index {args.index} non trouve dans config.CAMERAS")
        cam_key = f"cam_{args.index}"
    else:
        print(f"Camera ciblee : {cam_key} ({CAMERAS[cam_key]['role']})")

    intrinsic_path = args.intrinsic or f"configs/calibration_cam_{args.index}.json"
    if not os.path.exists(intrinsic_path):
        print(f"ERREUR : calibration intrinseque introuvable : {intrinsic_path}")
        sys.exit(1)

    print(f"Chargement intrinseque : {intrinsic_path}")
    camera_matrix, dist_coeffs = load_intrinsic(intrinsic_path)
    print(f"  fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")

    bus, motor_names = connect_robot(args.port)

    cam_w = CAMERAS.get(cam_key, {}).get("width", 1920)
    cam_h = CAMERAS.get(cam_key, {}).get("height", 1080)

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"ERREUR : camera {args.index} introuvable")
        bus.disconnect()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera ouverte ({actual_w}x{actual_h})")
    print(f"Damier attendu : {args.cols}x{args.rows} coins, carres {args.square_size} mm")

    images_dir = f"outputs/calibration_images/extrinsic_cam_{args.index}"
    if not args.no_save_images:
        os.makedirs(images_dir, exist_ok=True)
        print(f"Images sauvegardees dans : {images_dir}/")

    print()
    print("Controles : 'c'=capturer (damier vert), 'q'=terminer, ESC=annuler")
    print()
    print("Conseils :")
    print("  - 15 a 25 poses variees")
    print("  - rotations autour de >=2 axes differents (>30 degres entre poses)")
    print("  - damier toujours entierement visible")
    print("  - immobiliser le bras 1-2 sec avant chaque capture")
    print()

    captures = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rvec, tvec, corners = estimate_board_pose(
            frame, camera_matrix, dist_coeffs, args.rows, args.cols, args.square_size
        )

        if rvec is not None:
            cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, True)
            axis_len = args.square_size * 3
            axis_pts = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]])
            imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
            origin = tuple(corners[0].ravel().astype(int))
            cv2.line(display, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
            cv2.line(display, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
            cv2.line(display, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)
            dist_mm = float(np.linalg.norm(tvec))
            status = f"Damier a {dist_mm:.0f}mm | Captures: {len(captures)} | 'c'=capturer"
            color = (0, 255, 0)
        else:
            status = f"Damier non detecte | Captures: {len(captures)}"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(f"Calibration extrinseque - {cam_key}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and rvec is not None:
            try:
                motor_pos = bus.sync_read("Present_Position", normalize=False)
            except Exception as e:
                print(f"  Echec lecture moteurs : {e}")
                continue

            n = len(captures) + 1
            capture_data = {
                "id": n,
                "rvec_target_cam": rvec.flatten().tolist(),
                "tvec_target_cam": tvec.flatten().tolist(),
                "distance_mm": float(np.linalg.norm(tvec)),
                "motor_positions_raw": {k: float(v) for k, v in motor_pos.items()},
            }
            captures.append(capture_data)

            if not args.no_save_images:
                raw_path = os.path.join(images_dir, f"capture_{n:02d}_raw.png")
                annotated_path = os.path.join(images_dir, f"capture_{n:02d}_axes.png")
                cv2.imwrite(raw_path, frame)
                cv2.imwrite(annotated_path, display)

            print(f"  Capture {n} : distance={capture_data['distance_mm']:.0f}mm")

        elif key == ord("q"):
            break
        elif key == 27:
            print("Annule.")
            cap.release()
            cv2.destroyAllWindows()
            bus.disconnect()
            return

    cap.release()
    cv2.destroyAllWindows()
    bus.disconnect()

    if len(captures) < 5:
        print(f"\nSeulement {len(captures)} captures. Il en faut au moins 5 (15-25 recommande).")
        return

    result = {
        "camera_index": args.index,
        "camera_key": cam_key,
        "intrinsic_file": intrinsic_path,
        "motor_calibration_file": "configs/calibration_follower.json",
        "checkerboard": {
            "rows": args.rows,
            "cols": args.cols,
            "square_size_mm": args.square_size,
        },
        "motor_names": motor_names,
        "motor_position_units": "raw_encoder_counts",
        "num_captures": len(captures),
        "captures": captures,
    }

    output_path = args.output or f"configs/extrinsic_capture_cam_{args.index}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{len(captures)} captures sauvegardees : {output_path}")
    print("Etape suivante : lancer la resolution hand-eye sur ce fichier.")


if __name__ == "__main__":
    main()
