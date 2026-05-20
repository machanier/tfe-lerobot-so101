#!/usr/bin/env python3
"""
pick_and_place.py - Script CLI : le robot saisit un objet et le pose dans la boite.

Usage :
    python scripts/pick_and_place.py --target orange_cube
    python scripts/pick_and_place.py --target orange_cube --detector hf
    python scripts/pick_and_place.py --target orange_cube --dry-run    # test sans envoyer aux moteurs

Sequence executee :
    1. Capture les 3 cameras (multi-camera synchronisee).
    2. Detecte l'objet cible (HSV ou OWL-ViTv2).
    3. Triangule sa position 3D dans le repere base du robot.
    4. Planifie une saisie top-down (approche / saisie / retrait).
    5. Resout l'IK pour les 3 poses + drop.
    6. Genere une trajectoire articulaire lisse (quintique).
    7. L'execute sur le bras follower via le bus Feetech.

PRECAUTIONS :
    - Verifie que la BOITE DE DEPOSE est a sa position declaree dans
      configs/scene.json (center_base_m).
    - Verifie que la TABLE est degagee autour de la cible (V1 ne gere
      pas l'evitement d'obstacles, viendra au Sprint 4).
    - Premier essai : utilise --dry-run pour valider la chaine logique
      sans bouger le robot.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402
from src.pipeline import PickAndPlacePipeline, PipelineConfig  # noqa: E402


def apply_calib_profile(profile: str):
    """Copie les handeye_cam_*.json d'un profil de calibration vers configs/.

    Permet de tester plusieurs calibrations sans re-scotcher le damier :
    on a archive chaque session dans configs/calibration_profiles/<profile>/.
    Le pipeline lit toujours configs/handeye_cam_*.json, donc on copie le
    profil voulu juste avant de lancer.
    """
    prof_dir = REPO / "configs" / "calibration_profiles" / profile
    if not prof_dir.exists():
        avail = [p.name for p in (REPO / "configs" / "calibration_profiles").glob("*")
                 if p.is_dir()]
        print(f"!! Profil de calibration '{profile}' introuvable dans {prof_dir.parent}")
        print(f"   Profils disponibles : {avail}")
        sys.exit(1)

    cfg = REPO / "configs"
    for fname in ("handeye_cam_0.json", "handeye_cam_1.json"):
        src = prof_dir / fname
        if src.exists():
            shutil.copy(src, cfg / fname)

    # Affiche les metriques du profil charge
    meta_path = REPO / "configs" / "calibration_profiles" / "profiles_metadata.json"
    info = ""
    if meta_path.exists():
        meta = json.load(open(meta_path)).get(profile, {})
        if meta:
            info = (f" (cam0={meta.get('cam0_mean_mm')}mm, cam1={meta.get('cam1_mean_mm')}mm, "
                    f"coherence={meta.get('stereo_coherence_mean_mm')}mm, "
                    f"{meta.get('n_captures')} captures)")
    print(f">> Calibration profil '{profile}' charge dans configs/{info}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pick-and-place : le robot saisit un objet et le pose dans la boite.",
    )
    parser.add_argument("--target", type=str, default="orange_cube",
                        help="Label de l'objet a saisir (doit etre dans hsv_specs.json ou hf_specs.json)")
    parser.add_argument("--detector", choices=["hsv", "hf"], default="hsv",
                        help="Detecteur. hsv = rapide deterministe. hf = OWL-ViTv2 robuste.")
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT,
                        help="Port USB du follower")
    parser.add_argument("--max-velocity", type=float, default=0.5,
                        help="Vitesse articulaire max (rad/s). 0.5 = prudent.")
    parser.add_argument("--grip-close", type=float, default=5.0,
                        help="Fermeture pince pour grasper (0-100, 5 = presque ferme)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pas d'envoi moteur, juste log les angles calcules")
    parser.add_argument("--no-closed-loop", action="store_true",
                        help="Desactive le raffinement Sprint 4 par cam_2 "
                             "(stereo seule, moins precis ~30mm). Defaut : actif.")
    parser.add_argument("--display", action="store_true",
                        help="Affiche les 3 cameras (cv2.imshow) avec detections "
                             "aux moments cles (perception initiale). Snapshot sauve "
                             "dans outputs/perception/.")
    parser.add_argument("--calib-profile", type=str, default=None,
                        help="Charge un profil de calibration archive avant de lancer "
                             "(ex: s1, s2). Copie configs/calibration_profiles/<profil>/ "
                             "vers configs/. Permet de comparer plusieurs calibrations "
                             "sans re-scotcher le damier.")
    args = parser.parse_args()

    # Charge le profil de calibration demande (avant d'instancier le pipeline,
    # qui lit configs/handeye_cam_*.json au demarrage)
    if args.calib_profile:
        apply_calib_profile(args.calib_profile)

    config = PipelineConfig(
        target_label=args.target,
        detector_kind=args.detector,
        motor_port=args.port,
        max_velocity_rad_s=args.max_velocity,
        grip_close_pct=args.grip_close,
        dry_run=args.dry_run,
        closed_loop=(not args.no_closed_loop),
        display=args.display,
    )

    pipeline = PickAndPlacePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
