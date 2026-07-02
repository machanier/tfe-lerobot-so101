#!/usr/bin/env python3
"""
pick_and_place.py — saisie complète : le robot localise un objet, le saisit et le dépose.

Point d'entrée en ligne de commande de la pipeline perception → planification →
contrôle (implémentée dans src/pipeline.py).

Exemples :
    python scripts/pick_and_place.py --target orange_cube
    python scripts/pick_and_place.py --target orange_cube --detector hf
    python scripts/pick_and_place.py --target orange_cube --dry-run   # calcul seul, sans bouger le robot

Étapes exécutées :
    1. Capture synchronisée des trois caméras.
    2. Détection 2D de l'objet cible (HSV ou détecteur open-vocabulary Hugging Face).
    3. Triangulation stéréo de sa position 3D dans le repère base du robot.
    4. Choix d'un angle de prise adapté à la géométrie et à l'accessibilité
       (--top-down pour forcer une prise verticale).
    5. Cinématique inverse des poses (approche, prise, retrait, dépose) et
       génération d'une trajectoire articulaire lisse.
    6. Exécution sur le bras, avec raffinement en boucle fermée par la caméra
       embarquée juste avant la fermeture de la pince.

Avant un essai réel :
    - vérifier que la boîte de dépose est à la position déclarée dans configs/scene.json ;
    - dégager la table autour de la cible (l'évitement d'obstacles n'est pas géré) ;
    - valider d'abord la chaîne avec --dry-run.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402
from src.pipeline import PickAndPlacePipeline, PipelineConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Saisie d'un objet et dépose dans la boîte.",
    )
    parser.add_argument("--target", type=str, default="orange_cube",
                        help="Label de l'objet à saisir (défini dans hsv_specs.json ou hf_specs.json).")
    parser.add_argument("--detector", choices=["hsv", "hf"], default="hsv",
                        help="Détecteur 2D : hsv (rapide, déterministe) ou hf "
                             "(open-vocabulary, plus robuste). Défaut : hsv.")
    parser.add_argument("--port", type=str, default=FOLLOWER_PORT,
                        help="Port USB du bras follower.")
    parser.add_argument("--max-velocity", type=float, default=0.5,
                        help="Vitesse articulaire maximale (rad/s). Défaut : 0.5.")
    parser.add_argument("--grip-close", type=float, default=5.0,
                        help="Consigne de fermeture de la pince pour la saisie (0-100). Défaut : 5.")
    parser.add_argument("--grasp-threshold", type=float, default=None,
                        help="Seuil de détection de saisie (marge en %% au-dessus de "
                             "--grip-close). Défaut : 8. À baisser en cas de faux négatifs, "
                             "à monter en cas de faux positifs.")
    parser.add_argument("--grasp-lateral-offset", type=float, default=None,
                        help="Fixe manuellement l'offset latéral de prise (mm, repère image "
                             "cam_2, le long des mâchoires) et désactive le calcul adaptatif "
                             "(½ largeur + marge). Utile si la stéréo surestime la largeur "
                             "(objet rond) ; une valeur négative inverse le côté.")
    parser.add_argument("--lateral-offset-margin", type=float, default=None,
                        help="Marge (mm) ajoutée à la ½ largeur dans l'offset latéral "
                             "adaptatif. Défaut : 5. Sans effet avec --grasp-lateral-offset.")
    parser.add_argument("--grasp-forward-offset", type=float, default=None,
                        help="Offset de profondeur de prise (mm, repère image cam_2). Compense "
                             "le décalage entre l'axe du poignet et le point de fermeture des "
                             "doigts. Constante géométrique du montage. Défaut : 15.")
    parser.add_argument("--grasp-load-threshold", type=float, default=None,
                        help="Seuil de couple de la pince (Present_Load, 0-1023) pour juger la "
                             "saisie et détecter le contact en fermeture asservie. Repères : "
                             "pince vide ~200-230, objet tenu ~350+. Conseillé : 300.")
    parser.add_argument("--grasp-close-mode", choices=["servo", "static"],
                        default="servo",
                        help="servo (défaut) = fermeture asservie au couple, arrêt au contact "
                             "(+ --grasp-squeeze) ; static = consigne fixe à --grip-close.")
    parser.add_argument("--grasp-squeeze", type=float, default=3.0,
                        help="Mode servo : serrage de maintien (%% de course) ajouté après le "
                             "contact. Défaut : 3.")
    parser.add_argument("--gripper-max-opening", type=float, default=None,
                        help="Ouverture maximale réelle de la pince (mm), pour calculer "
                             "l'ouverture adaptative. Défaut : 150.")
    parser.add_argument("--gripper-open-margin", type=float, default=None,
                        help="Marge d'ouverture de chaque côté de l'objet (mm). Défaut : 10.")
    parser.add_argument("--grasp-yaw-offset", type=float, default=None,
                        help="Correction d'orientation de la pince (deg). Défaut : 90 (calé "
                             "pour ce montage). Mettre 0 pour la convention nominale.")
    parser.add_argument("--grab-offset", type=float, default=None,
                        help="Offset de serrage le long de l'axe d'approche (mm) : écart entre "
                             "le point visé par l'IK et la hauteur où les mâchoires serrent. "
                             "Défaut : 14.")
    parser.add_argument("--closed-loop-max-correction", type=float, default=None,
                        help="Plafond (mm) de la correction apportée par la caméra embarquée ; "
                             "au-delà, la position stéréo est conservée. Défaut : 80.")
    parser.add_argument("--cam2-observe-height", type=float, default=None,
                        help="Hauteur d'observation de la caméra embarquée au-dessus de l'objet "
                             "(mm) pour la détection de raffinement. Défaut : 120.")
    parser.add_argument("--zone-topdown", type=float, default=None,
                        help="Distance (mm) jusqu'à laquelle un objet bas est saisi par le "
                             "haut ; au-delà, prise inclinée. Défaut : 320.")
    parser.add_argument("--top-down", action="store_true",
                        help="Force une prise verticale par le haut. Par défaut, l'angle de "
                             "prise est choisi automatiquement selon la géométrie et "
                             "l'accessibilité.")
    parser.add_argument("--tilt-roll-offset", type=float, default=None,
                        help="Roll (deg) des prises inclinées autour de l'axe d'approche. "
                             "Défaut : valeur de --grasp-yaw-offset.")
    parser.add_argument("--side-grasp-min-height", type=float, default=None,
                        help="Hauteur de prise minimale (m) pour une prise inclinée "
                             "(dégagement de la table). Défaut : 0.020.")
    parser.add_argument("--ik-tol-trans", type=float, default=None,
                        help="Tolérance de position de l'IK (mm) pour juger un angle "
                             "atteignable. Défaut : 8.")
    parser.add_argument("--ik-tol-rot", type=float, default=None,
                        help="Tolérance d'orientation de l'IK (deg) pour juger un angle "
                             "atteignable. Défaut : 15.")
    parser.add_argument("--max-top-down-height", type=float, default=None,
                        help="Hauteur d'objet (m) au-delà de laquelle la prise par le haut est "
                             "refusée au profit d'une prise inclinée. Défaut : 0.12.")
    parser.add_argument("--wrist-flip-max-deg", type=float, default=None,
                        help="Sécurité : saut maximal de wrist_roll (deg) toléré ; au-delà, la "
                             "prise est refusée (évite un retournement du poignet vers la "
                             "table). Défaut : 120.")
    parser.add_argument("--no-lift-check", action="store_true",
                        help="Désactive la vérification après la levée (relecture "
                             "position + couple pour détecter les faux positifs).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calcule et journalise les angles sans les envoyer aux moteurs.")
    parser.add_argument("--no-closed-loop", action="store_true",
                        help="Désactive le raffinement par la caméra embarquée (stéréo seule, "
                             "moins précise).")
    parser.add_argument("--display", action="store_true",
                        help="Ouvre une fenêtre de suivi en temps réel. Les snapshots de "
                             "diagnostic sont sauvegardés indépendamment de cette option.")
    parser.add_argument("--no-snapshots", action="store_true",
                        help="Désactive la sauvegarde des snapshots de diagnostic dans "
                             "outputs/perception/ (sauvegardés par défaut).")
    args = parser.parse_args()

    config = PipelineConfig(
        target_label=args.target,
        detector_kind=args.detector,
        motor_port=args.port,
        max_velocity_rad_s=args.max_velocity,
        grip_close_pct=args.grip_close,
        dry_run=args.dry_run,
        closed_loop=(not args.no_closed_loop),
        display=args.display,
        save_snapshots=(not args.no_snapshots),
        grasp_mode=("top_down" if args.top_down else "adaptive"),
    )
    if args.grasp_threshold is not None:
        config.grasp_success_threshold_pct = args.grasp_threshold
    if args.grab_offset is not None:
        config.grasp_gripper_grab_offset_m = args.grab_offset / 1000.0
    if args.grasp_lateral_offset is not None:
        # Offset latéral fixé manuellement : on désactive le calcul adaptatif.
        config.grasp_lateral_tool_offset_mm = args.grasp_lateral_offset
        config.grasp_lateral_offset_auto = False
    if args.lateral_offset_margin is not None:
        config.grasp_lateral_offset_margin_mm = args.lateral_offset_margin
    if args.grasp_forward_offset is not None:
        config.grasp_forward_offset_mm = args.grasp_forward_offset
    if args.closed_loop_max_correction is not None:
        config.closed_loop_max_correction_mm = args.closed_loop_max_correction
    if args.cam2_observe_height is not None:
        config.cam2_observe_height_m = args.cam2_observe_height / 1000.0
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
    if args.tilt_roll_offset is not None:
        config.grasp_tilt_roll_deg = args.tilt_roll_offset
    if args.wrist_flip_max_deg is not None:
        config.wrist_flip_max_deg = args.wrist_flip_max_deg
    if args.side_grasp_min_height is not None:
        config.grasp_side_min_height_m = args.side_grasp_min_height
    if args.ik_tol_trans is not None:
        config.ik_tol_trans_mm = args.ik_tol_trans
    if args.ik_tol_rot is not None:
        config.ik_tol_rot_deg = args.ik_tol_rot
    if args.max_top_down_height is not None:
        config.grasp_max_top_down_height_m = args.max_top_down_height
    if args.zone_topdown is not None:
        config.grasp_zone_topdown_m = args.zone_topdown / 1000.0

    pipeline = PickAndPlacePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
