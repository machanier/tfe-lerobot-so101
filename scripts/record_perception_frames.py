#!/usr/bin/env python3
"""Capture synchronisee de 3 cameras et de l'etat moteur du robot.

Produit un dossier `data/perception_<timestamp>/` au format attendu par
`src.perception.camera_io.ReplayCamera` :

    manifest.json     liste ordonnee de snapshots, chacun avec :
                          - id, timestamp
                          - robot_state : raw_positions ou joint_angles_rad
                          - frames      : { "cam_0": "snap_01/cam_0.png", ... }
    snap_01/cam_0.png ...

Objectifs :
    - constituer un jeu de validation reproductible pour le memoire ;
    - iterer sur le detecteur sans robot branche, via le mode replay ;
    - servir de base a un futur jeu de donnees d'annotation.

Usage :
    1. Disposer les objets dans differentes configurations (occlusion, distance).
    2. Lancer le script (le robot follower doit etre branche : ses moteurs sont lus).
    3. Presser 'c' pour capturer un snapshot de la scene courante.
    4. Presser 'q' pour terminer et clore le manifest.

Sans robot branche, utiliser --no-robot : les snapshots sont enregistres avec
un robot_state en configuration zero. Le pipeline peut les rejouer pour cam_0 et
cam_1, mais cam_2 reste approximative faute d'etat moteur reel.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import CAMERAS, FOLLOWER_PORT  # noqa: E402


def open_cameras(cam_keys):
    caps = {}
    for k in cam_keys:
        cfg = CAMERAS[k]
        cap = cv2.VideoCapture(cfg["index"])
        if not cap.isOpened():
            for c in caps.values():
                c.release()
            raise RuntimeError(f"Impossible d'ouvrir {k} (index {cfg['index']}).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg["height"]))
        cap.set(cv2.CAP_PROP_FPS, float(cfg["fps"]))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        caps[k] = cap
    return caps


def grab_synchronized(caps):
    """Effectue grab puis retrieve sur chaque camera avec un timestamp commun."""
    grab_ok = {k: c.grab() for k, c in caps.items()}
    ts = time.time()
    frames = {}
    for k, c in caps.items():
        if not grab_ok[k]:
            frames[k] = None
            continue
        ok, img = c.retrieve()
        frames[k] = img if ok else None
    return frames, ts


def connect_robot(port):
    """Connecte le bus Feetech en lecture seule (torque desactive)."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

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
    bus.disable_torque()
    return bus


def main():
    parser = argparse.ArgumentParser(description="Enregistre des snapshots synchronises pour le replay.")
    parser.add_argument("--output", type=str, default=None,
                        help="Dossier de sortie. Par defaut : data/perception_<timestamp>/.")
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT,
                        help="Port USB du follower pour la lecture moteur. Par defaut : FOLLOWER_PORT.")
    parser.add_argument("--no-robot", action="store_true",
                        help="Enregistre sans robot branche, avec une configuration zero. Par defaut : desactive.")
    parser.add_argument("--cams", type=str, default="cam_0,cam_1,cam_2",
                        help="Cameras a enregistrer, en liste separee par des virgules. Par defaut : cam_0,cam_1,cam_2.")
    args = parser.parse_args()

    cam_keys = [k.strip() for k in args.cams.split(",") if k.strip()]
    out_root = Path(args.output) if args.output else (
        REPO / "data" / f"perception_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Sortie : {out_root}")
    print(f"Cameras : {cam_keys}")
    caps = open_cameras(cam_keys)

    bus = None
    if not args.no_robot:
        try:
            bus = connect_robot(args.port)
            print("Robot connecte, torque desactive.")
        except Exception as e:
            print(f"Echec de la connexion au robot ({e}). Bascule en mode --no-robot.")
            bus = None

    snapshots = []
    win = "Record perception"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames, ts = grab_synchronized(caps)

            # Affichage : juxtaposition horizontale des cameras (redimensionnees a la moitie pour tenir a l'ecran)
            tiles = []
            for k in cam_keys:
                img = frames[k]
                if img is None:
                    tiles.append(np.zeros((540, 960, 3), dtype=np.uint8))
                else:
                    tiles.append(cv2.resize(img, (960, 540)))
            display = np.hstack(tiles)
            cv2.putText(
                display,
                f"snapshots : {len(snapshots)} | 'c' = capturer  'q' = quitter",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                if any(frames[k] is None for k in cam_keys):
                    print("  Frame manquante sur au moins une camera : capture ignoree.")
                    continue
                # Lecture des moteurs si le bus est disponible
                rs_payload = None
                if bus is not None:
                    try:
                        raw = bus.sync_read("Present_Position", normalize=False)
                        raw = {k: float(v) for k, v in raw.items()}
                        rs_payload = {"raw_positions": raw}
                    except Exception as e:
                        print(f"  Echec de lecture moteur : {e} (snapshot sans robot_state)")
                else:
                    # Configuration zero servant de valeur par defaut explicite
                    rs_payload = {
                        "joint_angles_rad": {
                            "shoulder_pan": 0.0, "shoulder_lift": 0.0,
                            "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0,
                        },
                        "_note": "placeholder (--no-robot)"
                    }
                # Enregistrement des images
                idx = len(snapshots) + 1
                snap_dir = out_root / f"snap_{idx:03d}"
                snap_dir.mkdir(parents=True, exist_ok=True)
                rel_frames = {}
                for k in cam_keys:
                    rel = f"snap_{idx:03d}/{k}.png"
                    cv2.imwrite(str(out_root / rel), frames[k])
                    rel_frames[k] = rel
                snap = {
                    "id": idx,
                    "timestamp": float(ts),
                    "robot_state": rs_payload,
                    "frames": rel_frames,
                }
                snapshots.append(snap)
                print(f"  Snapshot #{idx} enregistre ({snap_dir.name})")
    finally:
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass

    manifest = {
        "_doc": "Genere par scripts/record_perception_frames.py",
        "created_at": datetime.now().isoformat(),
        "cam_keys": cam_keys,
        "snapshots": snapshots,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(snapshots)} snapshot(s) enregistre(s) : {out_root}")


if __name__ == "__main__":
    main()
