#!/usr/bin/env python3
"""
calibrate_extrinsic.py – Calibration extrinseque (camera -> robot)

Usage:
    python scripts/calibrate_extrinsic.py
    python scripts/calibrate_extrinsic.py --camera-index 0 --intrinsic configs/calibration_cam_0.json

La calibration extrinseque determine la transformation (rotation + translation)
entre le repere de la camera et l'espace de travail du robot.

Prerequis :
    1. Avoir fait la calibration intrinseque (calibrate_intrinsic.py)
    2. Avoir un damier visible par la camera

Procedure :
    1. Place le damier dans l'espace de travail
    2. Le script detecte le damier et estime sa pose 3D
    3. Capture plusieurs poses en deplacant le damier -> 'c'
    4. Appuie sur 'q' pour terminer et sauvegarder
"""

import argparse
import json
import os

import cv2
import numpy as np

from config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def load_intrinsic(path):
    """Charge les parametres intrinseques depuis un fichier JSON."""
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    return camera_matrix, dist_coeffs


def estimate_board_pose(frame, camera_matrix, dist_coeffs, rows, cols, square_size_mm):
    """Estime la pose du damier dans le repere camera."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if not found:
        return None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Points 3D du damier
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    _, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

    return rvec, tvec, corners_refined


def main():
    parser = argparse.ArgumentParser(description="Calibration extrinseque camera-robot")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="Index de la camera")
    parser.add_argument(
        "--intrinsic",
        type=str,
        default=None,
        help="Fichier de calibration intrinseque (defaut: configs/calibration_cam_<index>.json)",
    )
    parser.add_argument("--rows", type=int, default=7, help="Coins internes du damier (lignes)")
    parser.add_argument("--cols", type=int, default=7, help="Coins internes du damier (colonnes)")
    parser.add_argument(
        "--square-size", type=float, default=22.19, help="Taille des carres du damier en mm"
    )
    parser.add_argument("--output", type=str, default=None, help="Fichier de sortie")
    args = parser.parse_args()

    intrinsic_path = args.intrinsic or f"configs/calibration_cam_{args.camera_index}.json"
    if not os.path.exists(intrinsic_path):
        print(f"Fichier de calibration intrinseque non trouve: {intrinsic_path}")
        print("Lance d'abord: python scripts/calibrate_intrinsic.py")
        return

    print("Chargement de la calibration intrinseque...")
    camera_matrix, dist_coeffs = load_intrinsic(intrinsic_path)
    print(f"  fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
    print()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la camera {args.camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Dossier pour sauvegarder les images
    images_dir = f"outputs/calibration_images/extrinsic_cam_{args.camera_index}"
    os.makedirs(images_dir, exist_ok=True)
    print(f"Images sauvegardees dans: {images_dir}/")

    print("Place le damier dans l'espace de travail du robot.")
    print("Deplace-le a differentes positions pour capturer plusieurs poses.")
    print()
    print("Controles: 'c'=capturer, 'q'=terminer, ESC=annuler")
    print()

    camera_rvecs = []
    camera_tvecs = []
    captures = 0

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

            # Dessiner les axes du repere (rouge=X, vert=Y, bleu=Z)
            axis_len = args.square_size * 3
            axis_points = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]])
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            origin = tuple(corners[0].ravel().astype(int))
            cv2.line(display, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
            cv2.line(display, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
            cv2.line(display, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)

            # Afficher la distance
            dist_mm = np.linalg.norm(tvec)
            status = f"Damier a {dist_mm:.0f}mm | Captures: {captures} | 'c'=capturer"
            color = (0, 255, 0)
        else:
            status = f"Damier non detecte | Captures: {captures}"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Calibration extrinseque", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and rvec is not None:
            camera_rvecs.append(rvec)
            camera_tvecs.append(tvec)
            captures += 1
            dist_mm = np.linalg.norm(tvec)
            # Sauvegarder les images
            raw_path = os.path.join(images_dir, f"capture_{captures:02d}_raw.png")
            annotated_path = os.path.join(images_dir, f"capture_{captures:02d}_axes.png")
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(annotated_path, display)
            print(f"  Capture {captures}: distance={dist_mm:.0f}mm -> {raw_path}")

        elif key == ord("q"):
            break

        elif key == 27:
            print("Annule.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captures < 1:
        print("Aucune pose capturee.")
        return

    result = {
        "camera_index": args.camera_index,
        "intrinsic_file": intrinsic_path,
        "num_poses": captures,
        "checkerboard": {
            "rows": args.rows,
            "cols": args.cols,
            "square_size_mm": args.square_size,
        },
        "poses": [
            {
                "rvec": camera_rvecs[i].tolist(),
                "tvec": camera_tvecs[i].tolist(),
                "distance_mm": float(np.linalg.norm(camera_tvecs[i])),
            }
            for i in range(captures)
        ],
    }

    output_path = args.output or f"configs/extrinsic_cam_{args.camera_index}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{captures} poses sauvegardees: {output_path}")


if __name__ == "__main__":
    main()
