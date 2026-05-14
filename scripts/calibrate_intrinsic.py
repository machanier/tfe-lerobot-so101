#!/usr/bin/env python3
"""
calibrate_intrinsic.py – Calibration intrinseque d'une camera

Usage:
    python scripts/calibrate_intrinsic.py --index 0 --rows 6 --cols 9 --square-size 25

La calibration intrinseque determine les parametres internes de la camera :
- Matrice de la camera (focale fx, fy et point principal cx, cy)
- Coefficients de distorsion (k1, k2, p1, p2, k3)

Prerequis :
    Imprimer un damier (checkerboard) de calibration.
    Par defaut : 9x6 coins internes, carres de 25mm.
    Tu peux en generer un avec : python scripts/calibrate_intrinsic.py --generate

Procedure :
    1. Lance le script
    2. Presente le damier devant la camera sous differents angles
    3. Appuie sur 'c' pour capturer une pose (15-20 poses recommandees)
    4. Appuie sur 'q' pour lancer la calibration
    5. Le resultat est sauvegarde dans configs/calibration_cam_<index>.json
"""

import argparse
import json
import os

import cv2
import numpy as np

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def generate_checkerboard(rows, cols, square_size_px=80, output_path="configs/checkerboard.png"):
    """Genere une image de damier pour l'impression."""
    h = (rows + 1) * square_size_px
    w = (cols + 1) * square_size_px
    img = np.ones((h, w), dtype=np.uint8) * 255

    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                y0 = i * square_size_px
                x0 = j * square_size_px
                img[y0 : y0 + square_size_px, x0 : x0 + square_size_px] = 0

    cv2.imwrite(output_path, img)
    print(f"Damier ({cols + 1}x{rows + 1} cases, {cols}x{rows} coins internes) sauvegarde: {output_path}")
    print(f"Imprime-le a l'echelle reelle sur une feuille A4.")
    print(f"Mesure la taille reelle d'un carre en mm et utilise --square-size lors de la calibration.")


def calibrate(camera_index, rows, cols, square_size_mm, width, height, save_images=True):
    """Capture des poses d'un damier et calibre la camera."""
    # Dossier pour sauvegarder les images de calibration
    images_dir = f"outputs/calibration_images/intrinsic_cam_{camera_index}"
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
        print(f"Images sauvegardees dans: {images_dir}/")

    # Preparer les points 3D du damier (Z=0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points = []  # Points 3D dans le monde reel
    img_points = []  # Points 2D dans l'image

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la camera {camera_index}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {camera_index} ouverte ({actual_w}x{actual_h})")
    print(f"Damier attendu: {cols}x{rows} coins internes, carres de {square_size_mm}mm")
    print()
    print("Controles:")
    print("  'c' = Capturer la pose actuelle (quand le damier est detecte en vert)")
    print("  'q' = Terminer et lancer la calibration")
    print("  ESC = Annuler")
    print()
    print("Conseils pour une bonne calibration:")
    print("  - Prendre 15 a 20 poses minimum")
    print("  - Varier les angles (inclinaisons, rotations)")
    print("  - Couvrir toute la surface de l'image (coins, centre)")
    print("  - Inclure des poses proches et eloignees")
    print()

    captures = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Chercher le damier
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (cols, rows), corners_refined, found)
            status = f"Damier detecte | Captures: {captures} | 'c'=capturer, 'q'=calibrer"
            color = (0, 255, 0)
        else:
            status = f"Damier non detecte | Captures: {captures} | 'q'=calibrer"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Calibration intrinseque", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and found:
            obj_points.append(objp)
            img_points.append(corners_refined)
            captures += 1
            if save_images:
                # Sauvegarder l'image brute + l'image avec les coins dessines
                raw_path = os.path.join(images_dir, f"capture_{captures:02d}_raw.png")
                annotated_path = os.path.join(images_dir, f"capture_{captures:02d}_corners.png")
                cv2.imwrite(raw_path, frame)
                cv2.imwrite(annotated_path, display)
                print(f"  Capture {captures} enregistree -> {raw_path}")
            else:
                print(f"  Capture {captures} enregistree")

        elif key == ord("q"):
            break

        elif key == 27:  # ESC
            print("Annule.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    if captures < 5:
        print(f"Seulement {captures} captures. Il en faut au moins 5 pour une calibration fiable.")
        return None

    print(f"\nCalibration avec {captures} poses...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # Calculer l'erreur de reprojection
    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(obj_points)

    print(f"\nCalibration terminee.")
    print(f"  Erreur de reprojection moyenne: {mean_error:.4f} pixels")
    if mean_error < 0.5:
        print("  -> Excellente calibration.")
    elif mean_error < 1.0:
        print("  -> Bonne calibration.")
    else:
        print("  -> Calibration acceptable mais pourrait etre amelioree.")

    print(f"\n  Matrice camera:")
    print(f"    fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
    print(f"    cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}")
    print(f"  Distorsion: {dist_coeffs.ravel()}")

    result = {
        "camera_index": camera_index,
        "image_width": actual_w,
        "image_height": actual_h,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error": mean_error,
        "num_captures": captures,
        "checkerboard": {"rows": rows, "cols": cols, "square_size_mm": square_size_mm},
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Calibration intrinseque de camera")
    parser.add_argument("--index", type=int, default=CAMERA_INDEX, help="Index de la camera")
    parser.add_argument("--rows", type=int, default=7, help="Nombre de coins internes (lignes)")
    parser.add_argument("--cols", type=int, default=7, help="Nombre de coins internes (colonnes)")
    parser.add_argument(
        "--square-size", type=float, default=22.19, help="Taille d'un carre du damier en mm"
    )
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Largeur de l'image")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Hauteur de l'image")
    parser.add_argument(
        "--generate", action="store_true", help="Generer une image de damier pour impression"
    )
    parser.add_argument(
        "--no-save-images", action="store_true", help="Ne pas sauvegarder les images de capture"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier de sortie (defaut: configs/calibration_cam_<index>.json)",
    )
    args = parser.parse_args()

    if args.generate:
        generate_checkerboard(args.rows, args.cols)
        return

    result = calibrate(
        args.index, args.rows, args.cols, args.square_size, args.width, args.height,
        save_images=not args.no_save_images,
    )

    if result is not None:
        output_path = args.output or f"configs/calibration_cam_{args.index}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nCalibration sauvegardee: {output_path}")


if __name__ == "__main__":
    main()
