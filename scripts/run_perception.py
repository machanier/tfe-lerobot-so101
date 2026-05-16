#!/usr/bin/env python3
"""
run_perception.py - Pipeline complet de perception multi-cameras du SO-101.

Trois modes :
    --mode live    (defaut) : ouvre les 3 cameras + le bus moteur, boucle de
                              detection en temps reel, affiche les detections
                              annotees et les positions 3D estimees.
    --mode replay  : rejoue un dataset enregistre par
                              record_perception_frames.py. Pas besoin de hardware.
    --mode oneshot : capture UNE trame, fait la perception, imprime la Scene
                              et sauve un snapshot dans outputs/perception/.

Usage :
    python scripts/run_perception.py
    python scripts/run_perception.py --mode replay --replay data/perception_20260515_180000/
    python scripts/run_perception.py --mode oneshot --no-robot

Le detecteur par defaut est HSVDetector. Il charge les plages depuis
configs/perception/hsv_specs.json si present, sinon utilise
detector.default_hsv_specs() (plages indicatives a recalibrer).

Sortie en mode live : affichage cv2 + log console des positions estimees.
Sortie en mode oneshot : outputs/perception/scene_<timestamp>.json + 3 images annotees.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402

from src.perception.camera_io import MultiCamera, ReplayCamera  # noqa: E402
from src.perception.detector import (  # noqa: E402
    HFDetector,
    HSVDetector,
    default_hf_labels,
    default_hsv_specs,
    load_hf_specs,
    load_hsv_specs,
)
from src.perception.pose_estimator import PoseEstimator, PoseEstimatorConfig  # noqa: E402
from src.perception.robot_state import RobotStateProvider  # noqa: E402
from src.perception.scene import Frame, Scene  # noqa: E402


# ============================================================
# Helpers d'affichage
# ============================================================


def annotate_frame(frame: Frame, detections, scene: Scene) -> np.ndarray:
    """Dessine les bbox + centres 2D + reprojection 3D->2D des objets de la scene."""
    img = frame.image.copy()

    # 1. Detections 2D (vert)
    for d in detections:
        cx, cy = d.center_px
        if d.bbox is not None:
            x0, y0, x1, y1 = (int(v) for v in d.bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.putText(img, d.label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 2. Reprojection des positions 3D estimees (rouge)
    from src.perception.pose_estimator import _projection_matrix
    P = _projection_matrix(frame.K, frame.T_base_cam)
    for obj in scene.objects:
        Xh = np.hstack([obj.position_base_m, 1.0])
        uvw = P @ Xh
        if uvw[2] <= 0:
            continue
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), 2)
        pos_mm = obj.position_base_m * 1000.0
        label = f"{obj.label} ({pos_mm[0]:+.0f},{pos_mm[1]:+.0f},{pos_mm[2]:+.0f})"
        cv2.putText(img, label, (int(u) + 8, int(v) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def horizontal_tile(images, target_w=960):
    """Pile horizontale, downscale a target_w par image."""
    tiles = []
    for img in images:
        if img is None:
            tiles.append(np.zeros((540, target_w, 3), dtype=np.uint8))
            continue
        h, w = img.shape[:2]
        scale = target_w / w
        tiles.append(cv2.resize(img, (target_w, int(h * scale))))
    return np.hstack(tiles)


# ============================================================
# Pipelines
# ============================================================


def make_detector(detector_kind: str, specs_path: str, hf_specs_path: str):
    """Construit le detecteur (HSV ou HF) + le mapping label->meta.

    Args:
        detector_kind : "hsv" (V1 deterministe) ou "hf" (V2 OWL-ViTv2).
        specs_path    : chemin hsv_specs.json (utilise si detector_kind=hsv).
        hf_specs_path : chemin hf_specs.json (utilise si detector_kind=hf).

    Returns:
        (detector, specs_by_label_dict)
    """
    if detector_kind == "hsv":
        if specs_path and Path(specs_path).exists():
            specs = load_hsv_specs(Path(specs_path))
            print(f"Specs HSV chargees : {specs_path}  ({len(specs)} labels)")
        else:
            specs = default_hsv_specs()
            print("Specs HSV : valeurs par defaut "
                  "(genere configs/perception/hsv_specs.json via scripts/calibrate_hsv.py)")
        return HSVDetector(specs), {s.label: s.meta for s in specs}

    if detector_kind == "hf":
        if hf_specs_path and Path(hf_specs_path).exists():
            cfg = load_hf_specs(Path(hf_specs_path))
            labels = cfg["labels"]
            model_name = cfg.get("model_name", "google/owlv2-base-patch16-ensemble")
            threshold = float(cfg.get("score_threshold", 0.15))
            mapping = cfg.get("_label_mapping") or {}
            print(f"Specs HF chargees : {hf_specs_path}  ({len(labels)} labels)")
        else:
            labels = default_hf_labels()
            model_name = "google/owlv2-base-patch16-ensemble"
            threshold = 0.15
            mapping = {}
            print("Specs HF : valeurs par defaut.")

        det = HFDetector(prompt_labels=labels, model_name=model_name,
                         score_threshold=threshold)
        # Si un mapping est fourni, on wrap pour renommer les labels
        if mapping:
            from src.perception.detector import HFDetector as _HF  # type: ignore
            orig_detect = det.detect

            def detect_with_mapping(frame):
                dets = orig_detect(frame)
                for d in dets:
                    if d.label in mapping:
                        d.label = mapping[d.label]
                return dets
            det.detect = detect_with_mapping  # type: ignore[assignment]

        # specs_by_label vide (pas de dimensions metriques pour HF)
        # Le PnP mono ne marchera pas sans (mais c'est OK car HF couvre
        # plus de cas que HSV via la stereo).
        return det, {}

    raise ValueError(f"detector_kind inconnu: {detector_kind!r}")


def run_live(args):
    detector, specs_meta = make_detector(args.detector, args.specs, args.hf_specs)
    estimator = PoseEstimator(specs_by_label=specs_meta)

    provider = RobotStateProvider()
    if not args.no_robot:
        try:
            provider.connect_live(args.port)
            print(f"Robot connecte sur {args.port}")
        except Exception as e:
            print(f"Connexion robot KO ({e}). On bascule en mode --no-robot.")
            args.no_robot = True

    # Garantit la liberation du bus moteur meme si Ctrl+C ou crash dans la
    # boucle (sinon le port reste en etat occupe et le prochain lancement
    # echoue avec "Incorrect status packet").
    try:
        with MultiCamera() as mc:
            print(mc.info())
            win = "Perception (live)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            try:
                while True:
                    # 1. Etat robot pour cam_2 eye-in-hand
                    if args.no_robot:
                        rs = provider.from_angles({j: 0.0 for j in
                                                   ["shoulder_pan", "shoulder_lift",
                                                    "elbow_flex", "wrist_flex", "wrist_roll"]})
                    else:
                        rs = provider.read_live()
                    # 2. Capture
                    frames = mc.grab(robot_state=rs)
                    # 3. Detection
                    dets_by_cam = detector.detect_multi(frames)
                    # 4. Reconstruction 3D
                    scene = estimator.build_scene(dets_by_cam, frames)
                    # 5. Affichage
                    tiles = []
                    for k in mc.cam_keys:
                        if frames[k] is None:
                            tiles.append(None)
                        else:
                            tiles.append(annotate_frame(frames[k], dets_by_cam[k], scene))
                    cv2.imshow(win, horizontal_tile(tiles))
                    if args.print_each_frame and scene.objects:
                        print(scene.pretty())
                    # waitKey 100ms : permet de capter 'q' meme entre des frames
                    # lentes (HFDetector prend 3-5s par frame sur M4). Sinon
                    # waitKey(1) n'attend que 1ms pendant lesquelles l'OS doit
                    # detecter la touche -> souvent rate.
                    # En boucle classique HSV (10 FPS = 100ms/frame), 100ms
                    # ne change rien au framerate (on attend deja la frame).
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' ou ESC
                        break
            except KeyboardInterrupt:
                print("\nInterrompu par utilisateur. Liberation propre des ressources...")
            finally:
                cv2.destroyAllWindows()
    finally:
        # CRITIQUE : libere le bus moteur en TOUTES circonstances
        # (sinon le prochain lancement echouera avec port deja ouvert).
        provider.disconnect_live()


def run_replay(args):
    detector, specs_meta = make_detector(args.detector, args.specs, args.hf_specs)
    estimator = PoseEstimator(specs_by_label=specs_meta)
    rc = ReplayCamera(Path(args.replay))
    print(f"Replay : {len(rc)} snapshots")
    win = "Perception (replay)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        for i, (frames, _rs) in enumerate(rc):
            dets_by_cam = detector.detect_multi(frames)
            scene = estimator.build_scene(dets_by_cam, frames)
            print(f"=== snap {i + 1}/{len(rc)} ===")
            print(scene.pretty())
            tiles = []
            for k in rc.cam_keys:
                if frames[k] is None:
                    tiles.append(None)
                else:
                    tiles.append(annotate_frame(frames[k], dets_by_cam[k], scene))
            cv2.imshow(win, horizontal_tile(tiles))
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def run_oneshot(args):
    detector, specs_meta = make_detector(args.detector, args.specs, args.hf_specs)
    estimator = PoseEstimator(specs_by_label=specs_meta)

    provider = RobotStateProvider()
    if not args.no_robot:
        try:
            provider.connect_live(args.port)
        except Exception as e:
            print(f"Connexion robot KO ({e}). Mode --no-robot.")
            args.no_robot = True

    with MultiCamera() as mc:
        rs = (provider.read_live() if not args.no_robot
              else provider.from_angles({j: 0.0 for j in
                                         ["shoulder_pan", "shoulder_lift",
                                          "elbow_flex", "wrist_flex", "wrist_roll"]}))
        # Laisse 0.3 s de warm-up (autoexposure)
        for _ in range(3):
            mc.grab(robot_state=rs)
            time.sleep(0.1)
        frames = mc.grab(robot_state=rs)
        dets = detector.detect_multi(frames)
        scene = estimator.build_scene(dets, frames)

    print(scene.pretty())

    # Sauve
    out_dir = REPO / "outputs" / "perception"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": scene.timestamp,
        "objects": [
            {
                "label": o.label,
                "position_base_m": o.position_base_m.tolist(),
                "score": o.score,
                "method": o.meta.get("method", ""),
                "reproj_error_px": o.meta.get("reproj_error_px"),
            }
            for o in scene.objects
        ],
    }
    with open(out_dir / f"scene_{stamp}.json", "w") as f:
        json.dump(payload, f, indent=2)
    for k, frm in frames.items():
        if frm is None:
            continue
        ann = annotate_frame(frm, dets[k], scene)
        cv2.imwrite(str(out_dir / f"scene_{stamp}_{k}.png"), ann)
    print(f"Sauvegarde : {out_dir}/scene_{stamp}.json + 3 images")

    provider.disconnect_live()


# ============================================================
# Entree
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Pipeline de perception multi-cameras.")
    parser.add_argument("--mode", choices=["live", "replay", "oneshot"], default="live")
    parser.add_argument("--detector", choices=["hsv", "hf"], default="hsv",
                        help="Detecteur a utiliser. 'hsv' = V1 (seuillage couleur, "
                             "rapide, deterministe). 'hf' = V2 (OWL-ViTv2, robuste, "
                             "necessite transformers+torch installes).")
    parser.add_argument("--specs", type=str,
                        default=str(REPO / "configs" / "perception" / "hsv_specs.json"),
                        help="Fichier specs HSV (utilise si --detector hsv).")
    parser.add_argument("--hf-specs", type=str,
                        default=str(REPO / "configs" / "perception" / "hf_specs.json"),
                        help="Fichier specs HF (utilise si --detector hf).")
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT)
    parser.add_argument("--no-robot", action="store_true",
                        help="Pas de bus moteur ; cam_2 utilisera la config zero (degrade).")
    parser.add_argument("--replay", type=str, default=None,
                        help="Dossier de dataset pour --mode replay.")
    parser.add_argument("--print-each-frame", action="store_true")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(args)
    elif args.mode == "replay":
        if not args.replay:
            print("--mode replay requiert --replay <dossier>"); sys.exit(2)
        run_replay(args)
    elif args.mode == "oneshot":
        run_oneshot(args)


if __name__ == "__main__":
    main()
