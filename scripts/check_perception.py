#!/usr/bin/env python3
"""
check_perception.py - Validation chiffree du pipeline de perception.

Mesure experimentalement l'erreur de localisation 3D :

    1. L'utilisateur place les primitives colorees a des positions MESUREES
       au pied a coulisse depuis la base du robot, dans le repere robot
       (X_pied_a_coulisse en mm).
    2. Le script enregistre la position attendue (ground truth) et lance
       la perception.
    3. Compare position perception <-> position pied a coulisse, calcule
       erreur euclidienne par objet + agregat.

Le ground truth peut etre fourni de DEUX facons :

  --gt FILE   : fichier JSON pre-rempli (recommande, reproductible) :
                  {
                    "objects": [
                      {"label": "red_cube", "position_base_mm": [120, -30, 25]},
                      ...
                    ]
                  }

  --interactive : on saisit chaque position au clavier au lancement.

Sortie :
    outputs/perception/validation_<timestamp>.json
        {
          "ground_truth": [...], "estimated": [...],
          "errors_mm": [{"label": ..., "error_xyz_mm": [...], "error_norm_mm": ...}],
          "summary": {"mean_mm": ..., "max_mm": ..., "median_mm": ...}
        }

Critere de succes (Sprint 2 dans PROJECT_STATUS.md) : erreur moyenne <= 10 mm
(au plancher de bruit de la calibration hand-eye actuelle, 5-7 mm).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402

from src.perception.camera_io import MultiCamera  # noqa: E402
from src.perception.detector import (  # noqa: E402
    HFDetector,
    HSVDetector,
    default_hf_labels,
    default_hsv_specs,
    flatten_specs,
    load_hf_specs,
    load_hsv_specs,
)
from src.perception.pose_estimator import PoseEstimator  # noqa: E402
from src.perception.robot_state import RobotStateProvider  # noqa: E402


def load_ground_truth(path: Path) -> list[dict]:
    data = json.load(open(path))
    out = []
    for o in data["objects"]:
        out.append({
            "label": o["label"],
            "position_base_m": np.array(o["position_base_mm"], dtype=float) / 1000.0,
        })
    return out


def prompt_ground_truth(labels: list[str]) -> list[dict]:
    """Demande a l'utilisateur les positions des objets en mm."""
    print(f"Objets attendus (labels detectables) : {labels}")
    out = []
    print("Pour chaque objet, tape : LABEL X Y Z (mm)")
    print("Termine par une ligne vide.")
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 4:
            print("  Format : LABEL X Y Z (en mm). Reessaie.")
            continue
        label, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        out.append({"label": label, "position_base_m": np.array([x, y, z]) / 1000.0})
    return out


def estimate_scene(no_robot: bool, port: str, specs_path: str,
                   detector_kind: str = "hsv",
                   hf_specs_path: Optional[str] = None,
                   warmup: int = 5):
    """Acquiert une scene 3D avec la chaine complete (HSV ou HF detector)."""
    # `known_labels` : liste des labels que le detecteur PEUT detecter (utile
    # pour le mode --interactive et le retour de la fonction). En HF, ce sont
    # les valeurs du mapping (ou les prompts si pas de mapping).
    if detector_kind == "hsv":
        if Path(specs_path).exists():
            specs = load_hsv_specs(Path(specs_path))
        else:
            specs = default_hsv_specs()
            print("  (specs HSV par defaut - resultats indicatifs)")
        detector = HSVDetector(specs)
        flat = flatten_specs(specs)  # union si specs par-camera
        specs_meta = {s.label: s.meta for s in flat}
        known_labels = [s.label for s in flat]
    elif detector_kind == "hf":
        if hf_specs_path and Path(hf_specs_path).exists():
            cfg = load_hf_specs(Path(hf_specs_path))
            labels = cfg["labels"]
            model_name = cfg.get("model_name", "google/owlv2-base-patch16-ensemble")
            threshold = float(cfg.get("score_threshold", 0.15))
            mapping = cfg.get("_label_mapping") or {}
        else:
            labels = default_hf_labels()
            model_name = "google/owlv2-base-patch16-ensemble"
            threshold = 0.15
            mapping = {}
        detector = HFDetector(prompt_labels=labels, model_name=model_name,
                              score_threshold=threshold)
        # Wrap detect pour appliquer mapping si fourni
        if mapping:
            orig_detect = detector.detect
            def detect_with_mapping(frame):
                dets = orig_detect(frame)
                for d in dets:
                    if d.label in mapping:
                        d.label = mapping[d.label]
                return dets
            detector.detect = detect_with_mapping
        specs_meta = {}
        # known_labels = labels internes (apres mapping) ou prompts si pas de mapping
        known_labels = (list(mapping.values()) if mapping else list(labels))
    else:
        raise ValueError(f"detector_kind inconnu: {detector_kind}")
    estimator = PoseEstimator(specs_by_label=specs_meta)

    provider = RobotStateProvider()
    if not no_robot:
        try:
            provider.connect_live(port)
        except Exception as e:
            print(f"Robot KO ({e}). Bascule en --no-robot.")
            no_robot = True

    # try/finally CRITIQUE pour liberer le bus en cas d'erreur (sinon le port
    # USB reste occupe et le prochain lancement echoue avec "Incorrect status packet").
    try:
        with MultiCamera() as mc:
            rs = (provider.read_live() if not no_robot
                  else provider.from_angles({j: 0.0 for j in
                                             ["shoulder_pan", "shoulder_lift",
                                              "elbow_flex", "wrist_flex", "wrist_roll"]}))
            # Warm-up autoexposure
            for _ in range(warmup):
                mc.grab(robot_state=rs)
                time.sleep(0.1)
            frames = mc.grab(robot_state=rs)
            dets = detector.detect_multi(frames)
            scene = estimator.build_scene(dets, frames)
    finally:
        provider.disconnect_live()
    return scene, known_labels


def evaluate(scene, ground_truth):
    """Compare scene <-> ground truth, retourne stats + entries detail."""
    est_by_label = {o.label: o for o in scene.objects}
    errors = []
    for gt in ground_truth:
        label = gt["label"]
        gt_pos = gt["position_base_m"]
        if label not in est_by_label:
            errors.append({
                "label": label,
                "detected": False,
                "gt_mm": (gt_pos * 1000).tolist(),
            })
            continue
        est_pos = est_by_label[label].position_base_m
        diff = (est_pos - gt_pos) * 1000.0
        errors.append({
            "label": label,
            "detected": True,
            "gt_mm": (gt_pos * 1000).tolist(),
            "est_mm": (est_pos * 1000).tolist(),
            "error_xyz_mm": diff.tolist(),
            "error_norm_mm": float(np.linalg.norm(diff)),
            "method": est_by_label[label].meta.get("method", ""),
            "reproj_error_px": est_by_label[label].meta.get("reproj_error_px"),
        })

    detected = [e for e in errors if e["detected"]]
    if detected:
        norms = [e["error_norm_mm"] for e in detected]
        summary = {
            "n_detected": len(detected),
            "n_missing": len(errors) - len(detected),
            "mean_mm": float(np.mean(norms)),
            "median_mm": float(np.median(norms)),
            "max_mm": float(np.max(norms)),
        }
    else:
        summary = {"n_detected": 0, "n_missing": len(errors),
                   "mean_mm": float("nan"), "median_mm": float("nan"),
                   "max_mm": float("nan")}
    return errors, summary


def main():
    parser = argparse.ArgumentParser(description="Validation chiffree de la perception 3D.")
    parser.add_argument("--gt", type=str, default=None,
                        help="Fichier JSON ground truth (positions mm)")
    parser.add_argument("--interactive", action="store_true",
                        help="Saisir le ground truth au clavier")
    parser.add_argument("--detector", choices=["hsv", "hf"], default="hsv",
                        help="hsv = V1 (rapide deterministe), hf = V2 (OWL-ViTv2 robuste)")
    parser.add_argument("--specs", type=str,
                        default=str(REPO / "configs" / "perception" / "hsv_specs.json"))
    parser.add_argument("--hf-specs", type=str,
                        default=str(REPO / "configs" / "perception" / "hf_specs.json"))
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT)
    parser.add_argument("--no-robot", action="store_true")
    args = parser.parse_args()

    print("============================================================")
    print(" VALIDATION PERCEPTION 3D - TFE LeRobot SO-101")
    print("============================================================")
    print()

    # 1. Ground truth
    if args.gt:
        gt = load_ground_truth(Path(args.gt))
        print(f"Ground truth charge : {len(gt)} objet(s)")
    elif args.interactive:
        # Detecteur charge juste pour exposer les labels possibles
        if Path(args.specs).exists():
            specs = load_hsv_specs(Path(args.specs))
        else:
            specs = default_hsv_specs()
        gt = prompt_ground_truth([s.label for s in flatten_specs(specs)])
    else:
        print("Specifie --gt FICHIER ou --interactive")
        sys.exit(2)
    if not gt:
        print("Ground truth vide. Annule.")
        sys.exit(2)

    # 2. Acquisition
    print()
    print("Acquisition de la scene en cours...")
    scene, _ = estimate_scene(args.no_robot, args.port, args.specs,
                              detector_kind=args.detector,
                              hf_specs_path=args.hf_specs)
    print(scene.pretty())
    print()

    # 3. Evaluation
    errors, summary = evaluate(scene, gt)

    print("== Erreur par objet ==")
    for e in errors:
        if not e["detected"]:
            print(f"  {e['label']:<18} NON DETECTE  (gt={e['gt_mm']})")
            continue
        gt_mm = e["gt_mm"]; est_mm = e["est_mm"]; n = e["error_norm_mm"]
        method = e["method"]
        print(f"  {e['label']:<18} gt=({gt_mm[0]:+6.1f},{gt_mm[1]:+6.1f},{gt_mm[2]:+6.1f}) "
              f"est=({est_mm[0]:+6.1f},{est_mm[1]:+6.1f},{est_mm[2]:+6.1f}) mm  "
              f"err={n:5.1f} mm  ({method})")

    print()
    print("== Synthese ==")
    print(f"  detectes : {summary['n_detected']} / {summary['n_detected'] + summary['n_missing']}")
    print(f"  erreur moyenne   : {summary['mean_mm']:.2f} mm")
    print(f"  erreur mediane   : {summary['median_mm']:.2f} mm")
    print(f"  erreur max       : {summary['max_mm']:.2f} mm")

    # Verdict cahier des charges
    print()
    if summary["n_detected"] == 0:
        print("  [!] Aucun objet detecte : verifie l'eclairage / les plages HSV "
              "(scripts/calibrate_hsv.py)")
    elif summary["mean_mm"] < 10:
        print("  [OK] Erreur moyenne < 10 mm : conforme au plancher de bruit "
              "calibration. Passe au Sprint 3 (grasp planning).")
    elif summary["mean_mm"] < 20:
        print("  [ACCEPTABLE] Erreur moyenne 10-20 mm : utilisable, mais "
              "la replanification (Sprint 4) compensera.")
    else:
        print("  [A REVOIR] Erreur moyenne > 20 mm : verifie hand-eye + HSV.")

    # Sauvegarde
    out_dir = REPO / "outputs" / "perception"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": time.time(),
        "ground_truth": [{"label": g["label"],
                          "position_base_mm": (g["position_base_m"] * 1000).tolist()}
                         for g in gt],
        "errors": errors,
        "summary": summary,
    }
    out_path = out_dir / f"validation_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRapport sauve : {out_path}")


if __name__ == "__main__":
    main()
