#!/usr/bin/env python3
"""
Aperçu temps réel d'une caméra OpenCV pour régler la focale.

Usage:
    python scripts/preview_camera.py
    python scripts/preview_camera.py --camera 1
    python scripts/preview_camera.py --camera 0 --width 1280 --height 720 --fps 30

Touches:
    q : quitter
    s : sauvegarder l'image courante dans outputs/captured_images
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aperçu temps réel d'une caméra OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Index de la caméra OpenCV.")
    parser.add_argument("--width", type=int, default=1920, help="Largeur demandée.")
    parser.add_argument("--height", type=int, default=1080, help="Hauteur demandée.")
    parser.add_argument("--fps", type=int, default=30, help="FPS demandés.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/captured_images"),
        help="Dossier où sauvegarder une capture avec la touche 's'.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {args.camera}.")
        print("   Vérifie l'autorisation caméra dans macOS, ou essaie --camera 1 / --camera 2.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    window_name = f"Preview Camera {args.camera}"
    print(f"📷 Aperçu caméra {args.camera} démarré.")
    print(f"   Résolution active: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
    print("   Touches: q pour quitter, s pour sauvegarder une image")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Lecture de frame impossible, arrêt de l'aperçu.")
                return 1

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = args.output_dir / f"preview_camera_{args.camera}_{timestamp}.png"
                cv2.imwrite(str(path), frame)
                print(f"💾 Image sauvegardée: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("👋 Aperçu caméra arrêté.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
