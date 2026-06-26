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
                        help="OVERRIDE MANUEL de l'offset lateral de prise (mm). Par "
                             "DEFAUT l'offset est ADAPTATIF (= ½ largeur de l'objet + "
                             "marge, voir --lateral-offset-margin) pour amener le doigt "
                             "FIXE a fleur de l'arete QUELLE QUE SOIT LA TAILLE -- pas de "
                             "valeur codee par objet. Passer ce flag FIGE l'offset a la "
                             "valeur donnee et coupe l'auto. Repere IMAGE cam_2, le long "
                             "des machoires (suit la pince si l'objet tourne ; n'affecte "
                             "pas le wrist) ; +N = gauche image, NEGATIF = inverse le cote, "
                             "0 = centre. Utile si la stereo sur-lit la largeur (cylindre "
                             "rond) ou pour tester un cote.")
    parser.add_argument("--lateral-offset-margin", type=float, default=None,
                        help="Marge constante (mm) ajoutee a la ½ largeur dans l'offset "
                             "lateral ADAPTATIF (le \"au cas ou la largeur est mal lue\"). "
                             "Defaut PipelineConfig=5. Sans effet si --grasp-lateral-offset "
                             "(override manuel) est fourni.")
    parser.add_argument("--grasp-forward-offset", type=float, default=None,
                        help="OFFSET DE PROFONDEUR (mm) -- repere IMAGE cam_2, vers le BAS "
                             "de l'image (= cote base). Corrige le defaut vu sur les "
                             "snapshots PRISE : l'objet finit TROP HAUT dans l'image, les "
                             "machoires ferment dans le vide AU-DESSUS de lui (les doigts, "
                             "rigides a cam_2, ferment plus bas que l'axe optique). "
                             "CALIBRATION GEOMETRIQUE du montage, CONSTANTE pour tout objet. "
                             "Defaut PipelineConfig=15 (estimation). A AFFINER en regardant "
                             "les snapshots PRISE : monte si encore en arriere, baisse si "
                             "passe devant. NEGATIF = recule, 0 = desactive.")
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
    parser.add_argument("--grab-offset", type=float, default=None,
                        help="Offset de serrage pince en mm (Z, le long de l'axe d'approche) : "
                             "distance entre le point vise par l'IK et la hauteur ou les machoires "
                             "serrent. DEFAUT 14mm (mesure : machoires au centre de l'objet). "
                             "Passe 0 pour tester sans compensation (saisie plus basse).")
    parser.add_argument("--closed-loop-max-correction", type=float, default=None,
                        help="Plafond (mm) de la correction cam_2 : au-dela, on garde la "
                             "stereo (cam_2 jugee suspecte). DEFAUT 80. cam_2 etant le "
                             "raffinement fiable, ce plafond ne sert qu'au cas rarissime ou "
                             "cam_2 voit un faux objet a l'autre bout de la plaque. Mettre "
                             "tres grand (ex 200) = faire TOUJOURS confiance a cam_2.")
    parser.add_argument("--cam2-observe-height", type=float, default=None,
                        help="HAUTEUR D'OBSERVATION cam_2 en mm, AU-DESSUS DE L'OBJET = "
                             "reference de detection. La capture cam_2 se fait a cette "
                             "hauteur (120mm=12cm par defaut) pour eloigner l'objet des "
                             "doigts dans l'image (ils occupent le bas du cadre) -> "
                             "centroide non biaise, quel que soit l'angle de prise. "
                             "Defaut PipelineConfig=120. La descente reste monotone "
                             "(observation 12cm -> approche corrigee 8cm -> grasp). "
                             "Trop haut -> blob trop petit (gating -> stereo).")
    parser.add_argument("--zone-topdown", type=float, default=None,
                        help="Distance (mm) jusqu'a laquelle un objet BAS est pris en "
                             "TOP-DOWN ; au-dela -> diagonale 45. Defaut 320 (=32cm). "
                             "Regle la frontiere top-down/45 sans recompiler (utile pour "
                             "mesurer le domaine en campagne).")
    parser.add_argument("--top-down", action="store_true",
                        help="Revient a la saisie VERTICALE PAR LE HAUT uniquement "
                             "(comportement de reference eprouve). Par defaut (sans "
                             "ce flag) le robot est ADAPTATIF : il choisit l'angle "
                             "d'attaque sur le balayage du plan sagittal (top-down / "
                             "diagonale / frontal) en gardant la 1ere prise atteignable. "
                             "Utile pour re-tester l'ancien comportement ou comparer.")
    parser.add_argument("--tilt-roll-offset", type=float, default=None,
                        help="Saisie adaptative : roll (deg) des prises INCLINEES "
                             "autour de l'axe d'approche (convention pince). Defaut = "
                             "--grasp-yaw-offset (90, cale en top-down). Si au 1er essai "
                             "incline les machoires ferment de travers, essaie 0, -90, etc.")
    parser.add_argument("--side-grasp-min-height", type=float, default=None,
                        help="Saisie adaptative : hauteur de prise mini (m) pour une "
                             "prise inclinee (degagement table). Defaut 0.020 (PROVISOIRE, "
                             "a mesurer). Plus petit = autorise des prises inclinees plus basses.")
    parser.add_argument("--ik-tol-trans", type=float, default=None,
                        help="Saisie adaptative : tolerance de position IK (mm) pour "
                             "juger un angle ATTEIGNABLE. Defaut 8.")
    parser.add_argument("--ik-tol-rot", type=float, default=None,
                        help="Saisie adaptative : tolerance d'orientation IK (deg) pour "
                             "juger un angle atteignable. Defaut 15.")
    parser.add_argument("--max-top-down-height", type=float, default=None,
                        help="Hauteur d'objet (m) au-dela de laquelle le TOP-DOWN est "
                             "refuse (l'adaptatif bascule alors en incline). Defaut 0.12. "
                             "Augmente un peu (ex 0.14) si la hauteur mesuree depasse 12cm "
                             "par bruit alors que le top-down passerait.")
    parser.add_argument("--wrist-flip-max-deg", type=float, default=None,
                        help="SECURITE : saut max de wrist_roll (deg) tolere depuis la "
                             "pose courante. Au-dela = demi-tour de poignet (pince a "
                             "l'envers, plonge vers la table) -> la prise est REFUSEE "
                             "(echec propre). Defaut 120. Baisse-le si tu vois encore un "
                             "retournement ; monte-le pour autoriser de plus grands "
                             "re-orientations (a tes risques).")
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
                        help="FENETRE LIVE cv2.imshow : grab continu des 3 cams "
                             "pendant le mouvement -> sature le bus USB, cam_2 "
                             "DECROCHE souvent. N'est PLUS necessaire pour avoir les "
                             "snapshots : ceux-ci sont sauves par defaut (voir "
                             "--no-snapshots). N'active --display que si tu veux la "
                             "fenetre temps reel ET que le hub tient.")
    parser.add_argument("--no-snapshots", action="store_true",
                        help="Desactive la SAUVEGARDE des snapshots .png dans "
                             "outputs/perception/ (snapshot perception, vues cam_2 du "
                             "raffinement, et snapshots PRISE verite-terrain). Par "
                             "defaut ils sont TOUJOURS sauves -- y compris sans "
                             "--display, car la sauvegarde ne fait qu'ecrire une frame "
                             "deja capturee (aucun grab continu, donc cam_2 reste "
                             "stable). C'est l'aide diagnostique principale avec les "
                             "logs.")
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
        save_snapshots=(not args.no_snapshots),
        grasp_mode=("top_down" if args.top_down else "adaptive"),
    )
    if args.grasp_threshold is not None:
        config.grasp_success_threshold_pct = args.grasp_threshold
    if args.grab_offset is not None:
        config.grasp_gripper_grab_offset_m = args.grab_offset / 1000.0
    if args.grasp_lateral_offset is not None:
        # --grasp-lateral-offset = OVERRIDE MANUEL : fige l'offset lateral a cette
        # valeur (mm, repere IMAGE cam_2, le long des machoires) et DESACTIVE l'auto
        # (½ largeur + marge). A utiliser pour les objets dont la stereo sur-lit la
        # largeur (cylindre rond) ou pour inverser le cote (valeur negative).
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
