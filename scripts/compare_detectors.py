#!/usr/bin/env python3
"""Benchmark des detecteurs HSV et HF sur les memes trames.

Compare les deux detecteurs (objectif 6 du cahier des charges, evaluation
experimentale) selon deux axes :

  1. Precision 3D : erreur euclidienne par rapport a la verite-terrain (--gt),
     via la meme chaine de triangulation stereo. Les trames sont capturees une
     seule fois puis partagees entre HSV et HF, ce qui garantit une comparaison
     equitable (meme entree, meme calibration, meme instant).
  2. Latence de detection : temps passe dans detector.detect_multi(), isole du
     reste de la boucle, en mediane sur --repeat passes. Le premier passage,
     qui inclut le warm-up MPS/JIT, est ignore.

Usage typique (un seul objet pose, bras du robot hors champ ou en position home
pour eviter la confusion bras-orange / cube-orange en HSV) :

    python scripts/compare_detectors.py --gt configs/perception/gt_test.json --label P1

Multi-positions : deplacer le cube, mettre a jour le --gt et relancer avec un
nouveau --label. Chaque execution ajoute deux lignes (hsv, hf) au fichier CSV
cumulatif outputs/perception/detector_comparison.csv. Recapitulatif par
position :

    python scripts/compare_detectors.py --summary

HF charge OWL-ViTv2 (environ 600 Mo, transformers et torch requis) et prend
plusieurs secondes par trame. L'option --no-hf execute la comparaison en mode
HSV seul, plus rapide pour enchainer les positions sans attendre HF.

Entrees : trames des cameras, fichier de verite-terrain JSON, specifications
HSV et HF. Sortie : tableau recapitulatif a l'ecran et lignes ajoutees au CSV
cumulatif.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402

from src.perception.camera_io import MultiCamera  # noqa: E402
from src.perception.pose_estimator import PoseEstimator  # noqa: E402
from src.perception.robot_state import RobotStateProvider  # noqa: E402

# Reutilise la construction des detecteurs et l'evaluation existantes afin de
# ne pas dupliquer la logique.
from run_perception import make_detector  # noqa: E402
from check_perception import evaluate, load_ground_truth  # noqa: E402

CSV_PATH = REPO / "outputs" / "perception" / "detector_comparison.csv"
CSV_FIELDS = ["timestamp", "label", "detector", "n_detected", "n_total",
              "mean_mm", "median_mm", "max_mm", "det_ms"]


def time_detection(detector, frames, repeat: int):
    """Mesure la latence de detection (mediane sur `repeat` passes).

    Une passe de warm-up non chronometree (chargement CUDA/MPS, caches), puis
    `repeat` passes chronometrees. Renvoie (mediane_ms, dernieres_detections).
    """
    detector.detect_multi(frames)  # warm-up non chronometre
    times_ms = []
    dets = None
    for _ in range(max(1, repeat)):
        t0 = time.time()
        dets = detector.detect_multi(frames)
        times_ms.append((time.time() - t0) * 1000.0)
    return float(np.median(times_ms)), dets


def acquire_frames(no_robot: bool, port: str, warmup: int):
    """Capture une trame des trois cameras apres warm-up de l'autoexposition.

    Renvoie les trames capturees. Le bus moteur est libere avant de rendre la
    main : chaque trame embarque deja l'image, la matrice K, les coefficients de
    distorsion et la transformation T_base_cam figee a l'instant de la capture.
    """
    provider = RobotStateProvider()
    if not no_robot:
        try:
            provider.connect_live(port)
        except Exception as e:
            print(f"Robot indisponible ({e}). Bascule en mode --no-robot.")
            no_robot = True
    try:
        with MultiCamera() as mc:
            rs = (provider.read_live() if not no_robot
                  else provider.from_angles({j: 0.0 for j in
                                             ["shoulder_pan", "shoulder_lift",
                                              "elbow_flex", "wrist_flex", "wrist_roll"]}))
            for _ in range(warmup):
                mc.grab(robot_state=rs)
                time.sleep(0.1)
            frames = mc.grab(robot_state=rs)
            # Detache les images du buffer camera pour qu'elles restent valides
            # apres la fermeture de MultiCamera.
            for k, f in frames.items():
                if f is not None:
                    f.image = f.image.copy()
    finally:
        provider.disconnect_live()
    return frames


def build_detectors(args):
    """Construit (kind, detector, specs_meta) pour HSV puis (si possible) HF."""
    out = [("hsv", *make_detector("hsv", args.specs, args.hf_specs))]
    if not args.no_hf:
        try:
            det, meta = make_detector("hf", args.specs, args.hf_specs)
            out.append(("hf", det, meta))
        except ImportError as e:
            print(f"HF indisponible (transformers ou torch manquant) : {e}")
            print("    Comparaison en mode HSV seul. Installer la stack ou utiliser --no-hf.")
    return out


def run_comparison(args):
    gt = load_ground_truth(Path(args.gt))
    print(f"Verite-terrain : {len(gt)} objet(s) charge(s) depuis {args.gt}")

    detectors = build_detectors(args)
    print(f"Capture d'une trame partagee (warmup={args.warmup})...")
    frames = acquire_frames(args.no_robot, args.port, args.warmup)

    rows = []
    results = []
    for kind, det, meta in detectors:
        med_ms, dets = time_detection(det, frames, args.repeat)
        scene = PoseEstimator(specs_by_label=meta).build_scene(dets, frames)
        _errors, summary = evaluate(scene, gt)
        results.append((det.name, summary, med_ms))
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "detector": kind,
            "n_detected": summary["n_detected"],
            "n_total": summary["n_detected"] + summary["n_missing"],
            "mean_mm": round(summary["mean_mm"], 2),
            "median_mm": round(summary["median_mm"], 2),
            "max_mm": round(summary["max_mm"], 2),
            "det_ms": round(med_ms, 1),
        })

    # Tableau
    print()
    print(f"== Comparaison HSV vs HF (memes frames, position '{args.label}') ==")
    print(f"{'detecteur':<28}{'detectes':>10}{'err_moy':>10}"
          f"{'err_med':>10}{'err_max':>10}{'latence':>12}")
    for name, s, med_ms in results:
        n = f"{s['n_detected']}/{s['n_detected'] + s['n_missing']}"
        print(f"{name:<28}{n:>10}{s['mean_mm']:>8.1f}mm{s['median_mm']:>8.1f}mm"
              f"{s['max_mm']:>8.1f}mm{med_ms:>9.0f}ms")

    _append_csv(rows)
    print(f"\n{len(rows)} ligne(s) ajoutee(s) dans {CSV_PATH}")
    print("Recapitulatif multi-positions : python scripts/compare_detectors.py --summary")


def _append_csv(rows):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary():
    if not CSV_PATH.exists():
        print(f"Aucun fichier CSV : {CSV_PATH}. Lancer d'abord des comparaisons.")
        return
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Fichier CSV vide.")
        return
    # Agrege par (label, detector) : moyenne des runs.
    agg = {}
    for r in rows:
        key = (r["label"], r["detector"])
        agg.setdefault(key, []).append(r)
    print("== Recapitulatif par position (moyenne des runs) ==")
    print(f"{'position':<12}{'detecteur':<8}{'runs':>6}{'err_moy':>10}{'latence':>12}")
    for (label, det) in sorted(agg):
        runs = agg[(label, det)]
        mean_err = np.mean([float(x["mean_mm"]) for x in runs])
        mean_ms = np.mean([float(x["det_ms"]) for x in runs])
        print(f"{label:<12}{det:<8}{len(runs):>6}{mean_err:>8.1f}mm{mean_ms:>9.0f}ms")


def main():
    p = argparse.ArgumentParser(description="Benchmark des detecteurs HSV et HF sur les memes trames.")
    p.add_argument("--gt", type=str, default=str(REPO / "configs" / "perception" / "gt_test.json"),
                   help="Fichier de verite-terrain (positions en mm). Defaut : gt_test.json.")
    p.add_argument("--label", type=str, default="P1",
                   help="Etiquette de la position du cube (P1, P2, ...) inscrite dans le CSV. Defaut : P1.")
    p.add_argument("--repeat", type=int, default=3,
                   help="Nombre de passes chronometrees par detecteur, dont on prend la mediane. Defaut : 3.")
    p.add_argument("--warmup", type=int, default=5,
                   help="Nombre de trames de warm-up de l'autoexposition avant la capture. Defaut : 5.")
    p.add_argument("--no-hf", action="store_true",
                   help="Execute la comparaison en mode HSV seul, sans charger le detecteur HF.")
    p.add_argument("--specs", type=str,
                   default=str(REPO / "configs" / "perception" / "hsv_specs.json"),
                   help="Fichier de specifications HSV. Defaut : hsv_specs.json.")
    p.add_argument("--hf-specs", type=str,
                   default=str(REPO / "configs" / "perception" / "hf_specs.json"),
                   help="Fichier de specifications HF. Defaut : hf_specs.json.")
    p.add_argument("--port", type=str, default=FOLLOWER_PORT,
                   help="Port serie du bras suiveur. Defaut : valeur de config.FOLLOWER_PORT.")
    p.add_argument("--no-robot", action="store_true",
                   help="N'interroge pas le robot et utilise une pose articulaire nulle.")
    p.add_argument("--summary", action="store_true",
                   help="N'acquiert aucune trame : affiche le recapitulatif du CSV cumulatif.")
    args = p.parse_args()

    if args.summary:
        print_summary()
        return
    run_comparison(args)


if __name__ == "__main__":
    main()
