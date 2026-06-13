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
    """USAGE EXCEPTIONNEL : ressort une calibration de backup/souvenir.

    Le flux NORMAL n'utilise PAS ce flag : la calibration attitree est
    directement dans configs/handeye_cam_*.json (= s1, la stereo conjointe B3b).
    Ce flag sert seulement a re-tester ponctuellement une ancienne calibration
    archivee dans configs/calibration_backups/<profile>/ (s2, legacy_separate).

    ATTENTION : ce flag ECRASE configs/handeye_cam_*.json avec le backup choisi.
    Pour revenir a la calibration attitree, relance avec --calib-profile s1.
    """
    backups = REPO / "configs" / "calibration_backups"
    prof_dir = backups / profile
    if not prof_dir.exists():
        avail = [p.name for p in backups.glob("*") if p.is_dir()]
        print(f"!! Backup de calibration '{profile}' introuvable dans {backups}")
        print(f"   Backups disponibles : {avail}")
        sys.exit(1)

    cfg = REPO / "configs"
    for fname in ("handeye_cam_0.json", "handeye_cam_1.json"):
        src = prof_dir / fname
        if src.exists():
            shutil.copy(src, cfg / fname)

    # Affiche les metriques du backup charge
    meta_path = backups / "profiles_metadata.json"
    info = ""
    if meta_path.exists():
        meta = json.load(open(meta_path)).get("profils", {}).get(profile, {})
        if meta:
            info = (f" (cam0={meta.get('cam0_residual_mm')}mm, "
                    f"cam1={meta.get('cam1_residual_mm')}mm, "
                    f"coherence={meta.get('coherence_stereo_mean_mm')}mm)")
    print(f">> [BACKUP] calibration '{profile}' chargee dans configs/{info}")
    print(f">> (flux normal = pas de flag, configs/ contient deja la calib attitree s1)")
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
    parser.add_argument("--grasp-threshold", type=float, default=None,
                        help="Seuil de detection saisie (marge %% au-dessus de grip-close). "
                             "Defaut PipelineConfig=8. Baisse si faux negatifs, monte si "
                             "faux positifs. Maxence a calibre ~8-9 pour le cube 30mm.")
    parser.add_argument("--grasp-lateral-offset", type=float, default=None,
                        help="Decalage lateral de la saisie en mm (pince asymetrique SO-101). "
                             "Defaut PipelineConfig=8 (calibre cube 30mm). Augmente/diminue "
                             "pour une prise plus 'carree' sur rectangle/cylindre. "
                             "Conseil reglage : --dry-run d'abord, puis live.")
    parser.add_argument("--grasp-load-threshold", type=float, default=None,
                        help="Seuil de COUPLE pince (Present_Load, 0-1023). Quand fourni, "
                             "c'est le couple SEUL qui juge la saisie (fermeture ET verif "
                             "post-levee) ; il sert aussi de seuil de CONTACT pour la "
                             "fermeture asservie. Reference : vide ~200-230, tenu ~350+. "
                             "Conseille : 300.")
    parser.add_argument("--grasp-close-mode", choices=["servo", "static"],
                        default="servo",
                        help="servo (defaut) = fermeture ASSERVIE au couple : la pince "
                             "descend par pas et S'ARRETE AU CONTACT de l'objet "
                             "(+ --grasp-squeeze de maintien). static = consigne aveugle "
                             "a --grip-close (comportement historique).")
    parser.add_argument("--grasp-squeeze", type=float, default=3.0,
                        help="Mode servo : serrage de maintien (%% de course) ajoute "
                             "apres le contact. Defaut 3.")
    parser.add_argument("--gripper-max-opening", type=float, default=None,
                        help="Ouverture MAX REELLE de la pince en mm (mesure terrain, "
                             "150 sur le poste de Maxence). Sert a calculer l'ouverture "
                             "adaptative : pince_%% = (largeur_objet + 2*marge) / ce max. "
                             "Si la pince ouvre trop/pas assez, ajuste ici.")
    parser.add_argument("--gripper-open-margin", type=float, default=None,
                        help="Marge d'ouverture de CHAQUE cote de l'objet (mm). Defaut 10. "
                             "Plus petit = pince plus juste (mais moins de tolerance a "
                             "l'erreur de visee).")
    parser.add_argument("--grasp-yaw-offset", type=float, default=None,
                        help="Correction de convention pince (deg). DEFAUT 90 (cale pour ce "
                             "montage : la pince ferme a 90deg de la convention nominale). "
                             "Passe 0 pour revenir au comportement nominal, ou une autre valeur "
                             "si la pince est remontee differemment.")
    parser.add_argument("--no-lift-check", action="store_true",
                        help="Desactive la VERIF POST-LEVEE (P1'). Par defaut, apres une "
                             "fermeture jugee OK le bras remonte a retract et RE-LIT "
                             "position+couple pour attraper les faux positifs "
                             "(effleurement du sommet, morsure de bord, appui table).")
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
                        help="EXCEPTIONNEL : ressort une calibration de backup depuis "
                             "configs/calibration_backups/<nom>/ (s2, legacy_separate). "
                             "Le flux NORMAL n'a pas besoin de ce flag : la calibration "
                             "attitree (s1) est deja dans configs/. Pour revenir a s1 "
                             "apres un test : --calib-profile s1.")
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
    if args.grasp_threshold is not None:
        config.grasp_success_threshold_pct = args.grasp_threshold
    if args.grasp_lateral_offset is not None:
        config.grasp_lateral_offset_mm = args.grasp_lateral_offset
    if args.grasp_load_threshold is not None:
        config.grasp_load_threshold = args.grasp_load_threshold
    config.grasp_close_servo = (args.grasp_close_mode == "servo")
    config.grasp_squeeze_pct = args.grasp_squeeze
    config.lift_verify = (not args.no_lift_check)
    if args.gripper_max_opening is not None:
        config.grasp_gripper_max_opening_mm = args.gripper_max_opening
    if args.gripper_open_margin is not None:
        config.grasp_gripper_open_margin_mm = args.gripper_open_margin
    if args.grasp_yaw_offset is not None:
        config.grasp_yaw_offset_deg = args.grasp_yaw_offset

    pipeline = PickAndPlacePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
