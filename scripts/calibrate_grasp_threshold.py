#!/usr/bin/env python3
"""Calibration empirique du seuil de detection de saisie reussie ou ratee.

Le seuil grasp_success_threshold_pct de PipelineConfig fixe la marge minimale
(en % au-dessus de la consigne de fermeture) au-dela de laquelle la pince est
consideree comme bloquee par un objet. Une pince ne se referme jamais tout a
fait a la consigne : sur un objet elle se bloque plus tot qu'a vide. Ce script
mesure les deux positions reelles pour placer le seuil au bon endroit, la ou la
discrimination entre saisie reussie et saisie a vide est maximale.

Procedure :
  1. Connexion au robot (torque active).
  2. Lecture des angles articulaires courants ; le bras ne bouge pas.
  3. Ouverture de la pince, puis fermeture a la consigne sur l'objet cible :
     lecture de la position reelle atteinte.
  4. Retrait de l'objet, fermeture a la meme consigne a vide : lecture de la
     position reelle atteinte.
  5. Calcul d'un seuil recommande, au milieu des deux marges mesurees.

Usage :
  python scripts/calibrate_grasp_threshold.py
  python scripts/calibrate_grasp_threshold.py --grip-close 5 --grip-open 100

Sortie :
  Affiche une marge (% au-dessus de la consigne de fermeture) a reporter dans :
    PipelineConfig.grasp_success_threshold_pct = <valeur>
  ou en argument CLI de experiment_campaign.py :
    --grasp-threshold <valeur>

La calibration est a refaire en cas de changement de pince, de taille d'objet
ou de consigne de fermeture.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402
from src.calibration.forward_kinematics import ARM_JOINTS  # noqa: E402
from src.calibration.motor_to_angle import raw_to_radians  # noqa: E402
from src.control.motor_controller import MotorController  # noqa: E402


def current_arm_angles_rad(controller: MotorController) -> dict[str, float]:
    """Lit les angles articulaires actuels (5 joints arm, sans gripper)."""
    raw_all = controller.read_raw_positions()
    angles = {}
    for j in ARM_JOINTS:
        c = controller.calib[j]
        unwrap = controller.unwrap.get(j)
        angles[j] = raw_to_radians(raw_all[j], c, unwrap)
    return angles


def close_and_measure(controller: MotorController,
                       arm_angles_rad: dict[str, float],
                       grip_open_pct: float,
                       grip_close_pct: float,
                       settle_s: float = 0.7) -> float:
    """Ouvre puis ferme la pince et renvoie la position reelle apres fermeture.

    Le bras est maintenu dans sa position courante (arm_angles_rad) : aucun
    mouvement articulaire n'est envoye, seule la pince s'ouvre puis se ferme.
    """
    # Ouverture de la pince
    print(f"  -> Ouverture pince a {grip_open_pct:.0f}%...")
    controller.send_angles(arm_angles_rad, gripper_pct=grip_open_pct)
    time.sleep(settle_s)

    # Attente de l'operateur (placement ou retrait de l'objet)
    try:
        input("  Placez ou retirez l'objet selon l'instruction, puis Entree : ")
    except (EOFError, KeyboardInterrupt):
        raise

    # Fermeture a la consigne
    print(f"  -> Fermeture pince a {grip_close_pct:.0f}%...")
    controller.send_angles(arm_angles_rad, gripper_pct=grip_close_pct)
    time.sleep(settle_s)

    # Lecture de la position reelle atteinte
    pct = controller.read_gripper_pct()
    print(f"     position reelle apres fermeture : {pct:.1f}%")
    return pct


def main():
    p = argparse.ArgumentParser(
        description="Calibration empirique du seuil de detection de saisie.",
    )
    p.add_argument("--port", default=FOLLOWER_PORT,
                   help="Port USB du bras follower (defaut : valeur de config.py).")
    p.add_argument("--grip-open", type=float, default=100.0,
                   help="Ouverture de la pince, de 0 a 100 (defaut : 100, grand ouvert).")
    p.add_argument("--grip-close", type=float, default=5.0,
                   help="Consigne de fermeture de la pince, de 0 a 100 (defaut : 5, quasi ferme).")
    p.add_argument("--settle", type=float, default=0.7,
                   help="Pause de stabilisation apres commande de pince avant lecture, en secondes (defaut : 0.7).")
    args = p.parse_args()

    print("=" * 70)
    print(" CALIBRATION DU SEUIL DE DETECTION DE SAISIE")
    print("=" * 70)
    print(f"  port               : {args.port}")
    print(f"  ouverture pince    : {args.grip_open:.0f}%")
    print(f"  consigne fermeture : {args.grip_close:.0f}%")
    print(f"  pause stabilisation: {args.settle:.1f}s")
    print()
    print(" Placez le bras dans une position de travail confortable avant de")
    print(" continuer. Le bras ne bouge pas pendant la calibration ; seule la")
    print(" pince s'ouvre et se ferme.")
    try:
        input(" Entree pour continuer, Ctrl+C pour annuler : ")
    except (EOFError, KeyboardInterrupt):
        print("\nAnnule.")
        return

    controller = MotorController()
    try:
        controller.connect(args.port)
        controller.enable_torque()
        print(">> Robot connecte, torque active.")
        print()

        # Memorise les angles articulaires courants, c'est-a-dire la position
        # du bras maintenue pendant les mesures.
        arm_angles = current_arm_angles_rad(controller)
        print(">> Angles articulaires memorises (le bras reste dans cette position) :")
        for j, a in arm_angles.items():
            import numpy as np
            print(f"     {j:<14} = {np.degrees(a):+7.1f} deg")
        print()

        # Mesure 1 : fermeture de la pince sur l'objet
        print(">" * 70)
        print(">>> Mesure 1 : fermeture de la pince sur l'objet")
        print(">>> Placez l'objet cible entre les doigts de la pince.")
        print(">" * 70)
        pct_on_object = close_and_measure(
            controller, arm_angles,
            grip_open_pct=args.grip_open,
            grip_close_pct=args.grip_close,
            settle_s=args.settle,
        )
        print()

        # Mesure 2 : fermeture de la pince a vide
        print(">" * 70)
        print(">>> Mesure 2 : fermeture de la pince a vide")
        print(">>> Retirez l'objet et ne laissez rien entre les doigts.")
        print(">" * 70)
        pct_empty = close_and_measure(
            controller, arm_angles,
            grip_open_pct=args.grip_open,
            grip_close_pct=args.grip_close,
            settle_s=args.settle,
        )
        print()

        # Calcul des marges
        margin_empty = pct_empty - args.grip_close
        margin_on_object = pct_on_object - args.grip_close
        # Seuil place au milieu des deux marges, pour une discrimination maximale.
        seuil_recommande = (margin_empty + margin_on_object) / 2.0
        delta = margin_on_object - margin_empty

        print("=" * 70)
        print(" RESULTATS")
        print("=" * 70)
        print(f"  Consigne fermeture pince : {args.grip_close:.1f}%")
        print()
        print(f"  Pince a vide       : {pct_empty:5.1f}%  "
              f"(marge {margin_empty:+5.1f}% au-dessus de la consigne)")
        print(f"  Pince sur objet    : {pct_on_object:5.1f}%  "
              f"(marge {margin_on_object:+5.1f}% au-dessus de la consigne)")
        print(f"  Difference         : {delta:+5.1f}%  "
              f"(plus elle est grande, plus la detection est fiable)")
        print()
        print(f"  Seuil recommande : grasp_success_threshold_pct = {seuil_recommande:.1f}")
        print()
        print("  Application :")
        print(f"    - dans PipelineConfig (defaut)   : grasp_success_threshold_pct = {seuil_recommande:.1f}")
        print(f"    - dans experiment_campaign.py CLI : --grasp-threshold {seuil_recommande:.1f}")
        print()
        if delta < 5.0:
            print("  [WARN] Difference inferieure a 5%. La detection sera fragile.")
            print("         Verifier que :")
            print("         - l'objet est bien centre dans la pince a la mesure 1 ;")
            print("         - la pince est en bon etat (ni grippee ni cassee) ;")
            print("         - les angles articulaires sont corrects (bras non tordu).")
        elif delta < 10.0:
            print("  [INFO] Difference moderee (5 a 10%). Le seuil discrimine,")
            print("         mais des saisies marginales pourront echouer.")
        else:
            print("  [OK] Difference superieure a 10%, detection robuste.")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
    finally:
        try:
            controller.disable_torque()
        except Exception:
            pass
        try:
            controller.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
