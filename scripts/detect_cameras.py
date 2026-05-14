#!/usr/bin/env python3
"""
detect_cameras.py – Detecte toutes les cameras connectees

Usage:
    python scripts/detect_cameras.py

Affiche l'index, la resolution et le backend de chaque camera detectee.
Utile pour mettre a jour les index dans config.py apres branchement du hub USB.
"""

import cv2


def main():
    print("Recherche des cameras connectees...")
    print()
    found = 0
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Demander 1920x1080 pour voir si la camera le supporte
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()

            # Essayer de lire une frame pour confirmer que la camera fonctionne
            ret, frame = cap.read()
            status = "OK" if ret else "PAS DE FRAME"

            print(f"  Camera {i}: {w}x{h} @ {fps:.0f}fps  [{backend}]  ({status})")
            cap.release()
            found += 1

    print()
    if found == 0:
        print("Aucune camera detectee.")
        print("  - Verifie les branchements USB")
        print("  - Verifie l'autorisation camera dans Reglages > Confidentialite > Camera")
    elif found < 3:
        print(f"Seulement {found} camera(s) detectee(s) sur 3 attendues.")
        print("  - Verifie que le hub USB est bien branche")
        print("  - Essaie de debrancher/rebrancher les cameras")
    else:
        print(f"{found} camera(s) detectee(s).")
        print()
        print("Mets a jour les index dans scripts/config.py si necessaire.")
        print("Pour identifier chaque camera visuellement :")
        print("  python scripts/preview_camera.py --camera 0")
        print("  python scripts/preview_camera.py --camera 1")
        print("  python scripts/preview_camera.py --camera 2")


if __name__ == "__main__":
    main()
