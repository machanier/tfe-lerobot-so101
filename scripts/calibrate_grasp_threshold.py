#!/usr/bin/env python3
"""
calibrate_grasp_threshold.py - Calibre empiriquement le seuil de detection
de saisie reussie/ratee (P1, etape B1 de la remise au propre).

PROBLEME RESOLU : le seuil par defaut grasp_success_threshold_pct=15.0 est
arbitraire. Selon ton hardware (pince TPU XLRobot + grip de tennis), la pince
ne peut PAS se fermer en dessous de ~12-15% sur un cube 30mm. Donc une saisie
REUSSIE (pince a 14%) se retrouve classee RATEE car marge 14-5=9% < seuil 15%.

PROCEDURE :
  1. Connecte le robot (torque active).
  2. Lit les angles articulaires courants (le bras ne bougera PAS).
  3. Ouvre la pince a 100%, attend.
  4. Demande a l'user de placer le cube cible entre les doigts.
  5. Ferme la pince a la consigne (grip_close_pct=5) et lit la position reelle.
  6. Demande a l'user de retirer le cube.
  7. Ferme la pince a la meme consigne et lit la position reelle (a vide).
  8. Calcule un seuil recommande = milieu entre les 2 mesures.

USAGE :
  python scripts/calibrate_grasp_threshold.py
  python scripts/calibrate_grasp_threshold.py --grip-close 5 --grip-open 100

OUTPUT :
  Affiche un seuil de marge (% au-dessus de grip_close) a mettre dans :
    PipelineConfig.grasp_success_threshold_pct = <valeur>
  ou en argument CLI de experiment_campaign.py :
    --grasp-threshold <valeur>

A REFAIRE si tu changes :
  - de pince (TPU vs metallique vs grip de tennis)
  - de taille d'objet (cube 30mm vs petit cube 15mm)
  - de consigne grip_close (different de 5%)
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
    """Ouvre puis ferme la pince, et renvoie la position reelle apres fermeture.

    Le bras est MAINTENU dans sa position courante (arm_angles_rad) -- on
    n'envoie pas de mouvement articulaire, juste de l'ouverture/fermeture pince.
    """
    # Ouverture
    print(f"  -> Ouverture pince a {grip_open_pct:.0f}%...")
    controller.send_angles(arm_angles_rad, gripper_pct=grip_open_pct)
    time.sleep(settle_s)

    # Attente user (placer/retirer cube)
    try:
        input("  >> Place/retire selon l'instruction puis ENTREE : ")
    except (EOFError, KeyboardInterrupt):
        raise

    # Fermeture a la consigne
    print(f"  -> Fermeture pince a {grip_close_pct:.0f}%...")
    controller.send_angles(arm_angles_rad, gripper_pct=grip_close_pct)
    time.sleep(settle_s)

    # Lecture position reelle
    pct = controller.read_gripper_pct()
    print(f"     position reelle apres fermeture : {pct:.1f}%")
    return pct


def main():
    p = argparse.ArgumentParser(
        description="Calibration empirique du seuil de detection saisie (P1/B1).",
    )
    p.add_argument("--port", default=FOLLOWER_PORT,
                   help="Port USB follower.")
    p.add_argument("--grip-open", type=float, default=100.0,
                   help="Ouverture pince 0-100 (defaut 100 = grand ouvert).")
    p.add_argument("--grip-close", type=float, default=5.0,
                   help="Consigne fermeture pince 0-100 (defaut 5 = quasi-ferme).")
    p.add_argument("--settle", type=float, default=0.7,
                   help="Pause apres commande pince avant lecture (s, defaut 0.7).")
    args = p.parse_args()

    print("=" * 70)
    print(" CALIBRATION SEUIL DE DETECTION SAISIE (B1)")
    print("=" * 70)
    print(f"  port               : {args.port}")
    print(f"  ouverture pince    : {args.grip_open:.0f}%")
    print(f"  consigne fermeture : {args.grip_close:.0f}%")
    print(f"  pause stabilisation: {args.settle:.1f}s")
    print()
    print(" IMPORTANT : place le bras dans une position confortable POUR TOI")
    print(" avant de continuer (le bras ne bougera pas pendant la calibration,")
    print(" seule la pince ouvrira/fermera).")
    try:
        input(" ENTREE pour continuer (Ctrl+C pour annuler) : ")
    except (EOFError, KeyboardInterrupt):
        print("\nAnnule.")
        return

    controller = MotorController()
    try:
        controller.connect(args.port)
        controller.enable_torque()
        print(">> Robot connecte, torque ACTIVE.")
        print()

        # Memorise les angles articulaires courants (= la position du bras
        # qu'on va maintenir pendant les mesures).
        arm_angles = current_arm_angles_rad(controller)
        print(f">> Angles articulaires memorises (le bras restera la) :")
        for j, a in arm_angles.items():
            import numpy as np
            print(f"     {j:<14} = {np.degrees(a):+7.1f} deg")
        print()

        # Mesure #1 : pince ferme SUR L'OBJET
        print(">" * 70)
        print(">>> MESURE #1 : pince ferme SUR L'OBJET")
        print(">>> Place le cube cible entre les doigts de la pince.")
        print(">" * 70)
        pct_on_object = close_and_measure(
            controller, arm_angles,
            grip_open_pct=args.grip_open,
            grip_close_pct=args.grip_close,
            settle_s=args.settle,
        )
        print()

        # Mesure #2 : pince ferme A VIDE
        print(">" * 70)
        print(">>> MESURE #2 : pince ferme A VIDE")
        print(">>> Retire l'objet, ne laisse RIEN entre les doigts.")
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
        # Seuil ideal = milieu entre les 2 marges (discrimination maximale)
        seuil_recommande = (margin_empty + margin_on_object) / 2.0
        delta = margin_on_object - margin_empty

        print("=" * 70)
        print(" RESULTATS")
        print("=" * 70)
        print(f"  Consigne fermeture pince : {args.grip_close:.1f}%")
        print()
        print(f"  Pince a VIDE       : {pct_empty:5.1f}%  "
              f"(marge {margin_empty:+5.1f}% au-dessus consigne)")
        print(f"  Pince sur OBJET    : {pct_on_object:5.1f}%  "
              f"(marge {margin_on_object:+5.1f}% au-dessus consigne)")
        print(f"  Difference         : {delta:+5.1f}%  "
              f"(plus c'est grand, plus la detection est fiable)")
        print()
        print(f"  --> SEUIL RECOMMANDE : grasp_success_threshold_pct = {seuil_recommande:.1f}")
        print()
        print(f"  Application :")
        print(f"    - Dans PipelineConfig (defaut)   : grasp_success_threshold_pct = {seuil_recommande:.1f}")
        print(f"    - Dans experiment_campaign.py CLI : --grasp-threshold {seuil_recommande:.1f}")
        print()
        if delta < 5.0:
            print("  [WARN] Difference < 5%. La detection sera fragile.")
            print("         Verifie que :")
            print("         - le cube est bien centre dans la pince a la mesure #1")
            print("         - la pince est en bon etat (pas grippee ni cassee)")
            print("         - les angles articulaires sont corrects (bras pas tordu)")
        elif delta < 10.0:
            print("  [INFO] Difference moderee (5-10%). Le seuil discrimine,")
            print("         mais des saisies marginales pourront echouer.")
        else:
            print("  [OK] Difference > 10%, detection robuste.")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n!! Interrompu par utilisateur.")
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
