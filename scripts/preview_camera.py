#!/usr/bin/env python3
"""
preview_camera.py – Apercu temps reel d'une camera pour regler le cadrage

Usage:
    python scripts/preview_camera.py
    python scripts/preview_camera.py --camera 1
    python scripts/preview_camera.py --width 1280 --height 720

Touches (dans la fenetre video) :
    q : quitter
    s : sauvegarder l'image courante dans outputs/captured_images/
"""

import argparse
import os
from datetime import datetime

import cv2

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def main():
    parser = argparse.ArgumentParser(description="Apercu temps reel d'une camera.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Index de la camera.")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Largeur demandee.")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Hauteur demandee.")
    parser.add_argument("--fps", type=int, default=CAMERA_FPS, help="FPS demandes.")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/captured_images",
        help="Dossier ou sauvegarder une capture avec la touche 's'.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la camera {args.camera}.")
        print("  Verifie l'autorisation camera dans macOS, ou essaie --camera 1 / --camera 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    window_name = f"Preview Camera {args.camera}"
    print(f"Apercu camera {args.camera} demarre.")
    print(f"  Resolution: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print("  Touches: 'q' pour quitter, 's' pour sauvegarder une image")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lecture de frame impossible, arret.")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = os.path.join(args.output_dir, f"preview_cam_{args.camera}_{timestamp}.png")
                cv2.imwrite(path, frame)
                print(f"  Image sauvegardee: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Apercu camera arrete.")


if __name__ == "__main__":
    main()
