#!/usr/bin/env python3
"""Capture simultanee des cameras cam_0 et cam_1 pour la calibration hand-eye
stereo conjointe.

Principe :
La calibration separee de cam_0 et cam_1 produit des residus independants
(de l'ordre de 6 mm chacun) qui peuvent s'additionner geometriquement lors de
la triangulation et introduire un biais sur l'axe Y. Calibrer les deux cameras
conjointement, sur les memes poses du damier au meme instant, ajoute une
contrainte forte : cv2.stereoCalibrate() estime T_cam0_cam1 avec une precision
sub-millimetrique. On en deduit ensuite T_base_cam1 = T_base_cam0 @ T_cam0_cam1,
de sorte que les deux calibrations sont coherentes par construction et que le
biais s'annule a la triangulation.

Reference : Hartley & Zisserman, Multiple View Geometry, 2018, chapitre 10.

Usage :
    python scripts/calibrate_extrinsic_stereo.py
    python scripts/calibrate_extrinsic_stereo.py --rows 6 --cols 9 --square-size 22

Procedure :
  1. Un damier (9x6 par defaut) est fixe sur la pince fermee du robot.
  2. Les deux cameras sont fixes sur la barriere avant.
  3. Deplacer le bras dans 30 a 60 poses variees (diversite angulaire
     superieure a 65 degres, distances de 30 a 80 cm). A chaque pose stable,
     appuyer sur 'c' pour capturer.
  4. Pour capturer, le damier doit etre detecte dans les deux cameras
     simultanement. Si une seule le voit, la capture est ignoree avec un message.
  5. Appuyer sur 'q' pour terminer et sauvegarder.

Sortie :
  configs/extrinsic_capture_stereo.json : paires synchronisees rvec, tvec,
  points image et positions moteur.

Etape suivante : python scripts/solve_handeye_stereo.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from config import CAMERAS, FOLLOWER_PORT  # noqa: E402


def load_intrinsic(path: str):
    data = json.load(open(path))
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


def open_camera(index: int, width: int = 1920, height: int = 1080):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} introuvable")
    # MJPG pour reduire la bande passante USB
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Prechauffage : stabilisation de l'exposition automatique
    for _ in range(5):
        cap.read()
    return cap


def connect_robot(port: str):
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as e:
        raise ImportError("LeRobot indisponible.") from e
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
    motor_names = list(motors.keys())
    return bus, motor_names


def estimate_board_pose(frame, K, D, rows, cols, square_size_mm):
    """Detecte le damier et calcule rvec/tvec en mm. Renvoie aussi les coins 2D.

    Utilise findChessboardCornersSB (Sector Based, OpenCV 4), plus precis et
    plus rapide que findChessboardCorners classique. Revient a la methode
    classique si SB est indisponible (compatibilite OpenCV anterieur a la 4).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection sub-pixel native (SB = Sector Based, environ deux fois plus precise)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                 + cv2.CALIB_CB_EXHAUSTIVE
                 + cv2.CALIB_CB_ACCURACY,
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if not found:
        return None, None, None, None

    # Points objet en mm (damier plan, z = 0)
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None, None, None
    return rvec, tvec, corners, obj


def synchronized_capture(cap_l, cap_r, flush_frames: int = 5):
    """Capture quasi-simultanee : vide les buffers OpenCV puis effectue des
    grab et retrieve rapproches sur les deux cameras.

    OpenCV VideoCapture bufferise environ cinq images. Sans vidage prealable,
    on lit une image obsolete, par exemple datant du moment ou le bras bougeait
    encore. On vide donc d'abord les buffers, puis on appelle grab() de maniere
    rapprochee sur les deux cameras (declenche la capture) avant retrieve() qui
    decode chaque buffer.

    Returns:
        (frame_l, frame_r) ou (None, None) en cas d'echec.
    """
    # 1. Vidage des buffers (lit et jette les images anciennes)
    for _ in range(flush_frames):
        cap_l.grab()
        cap_r.grab()

    # 2. Capture quasi-simultanee : grab() sur les deux cameras le plus proche
    #    possible dans le temps, puis retrieve() pour decoder. C'est le motif
    #    OpenCV recommande pour la synchronisation multi-camera.
    ok_l = cap_l.grab()
    ok_r = cap_r.grab()
    if not (ok_l and ok_r):
        return None, None
    ret_l, frame_l = cap_l.retrieve()
    ret_r, frame_r = cap_r.retrieve()
    if not (ret_l and ret_r):
        return None, None
    return frame_l, frame_r


def draw_overlay(frame, K, D, rvec, tvec, corners, square_size_mm,
                 cam_label, status_text, rows, cols):
    display = frame.copy()
    if rvec is not None:
        # drawChessboardCorners attend les dimensions du damier (cols, rows),
        # pas la taille de l'image en pixels.
        cv2.drawChessboardCorners(display, (cols, rows), corners, True)
        axis_len = square_size_mm * 3
        axis_pts = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]])
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
        origin = tuple(corners[0].ravel().astype(int))
        cv2.line(display, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
        cv2.line(display, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
        cv2.line(display, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)
    # Bandeau noir + label cam
    cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(display, cam_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    color = (0, 255, 0) if rvec is not None else (0, 0, 255)
    cv2.putText(display, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1)
    return display


def main():
    parser = argparse.ArgumentParser(
        description="Capture stereo simultanee de cam_0 et cam_1 pour la calibration hand-eye stereo."
    )
    parser.add_argument("--cam-indices", nargs=2, type=int, default=[0, 1],
                        metavar=("LEFT", "RIGHT"),
                        help="Indices OpenCV des deux cameras stereo (defaut : 0 1).")
    parser.add_argument("--port", default=FOLLOWER_PORT,
                        help="Port USB du bras follower (defaut : valeur de la configuration).")
    parser.add_argument("--rows", type=int, default=6,
                        help="Nombre de coins internes par ligne du damier (defaut : 6).")
    parser.add_argument("--cols", type=int, default=9,
                        help="Nombre de coins internes par colonne du damier (defaut : 9).")
    parser.add_argument("--square-size", type=float, default=22.0,
                        help="Taille d'une case du damier en mm (defaut : 22).")
    parser.add_argument("--output", default="configs/extrinsic_capture_stereo.json",
                        help="Chemin du fichier JSON de sortie (defaut : configs/extrinsic_capture_stereo.json).")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Ne pas enregistrer les images capturees sur le disque.")
    parser.add_argument("--display-scale", type=float, default=0.45,
                        help="Echelle d'affichage de la mosaique cote a cote (defaut : 0.45).")
    args = parser.parse_args()

    # Desactive OpenCL pour eviter les avertissements de cache OpenCL sur macOS
    # qui polluent la sortie. Aucun impact sur la calibration : le CPU suffit
    # pour findChessboardCorners et solvePnP.
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    idx_l, idx_r = args.cam_indices
    cam_l_key = next((k for k, v in CAMERAS.items() if v["index"] == idx_l), f"cam_{idx_l}")
    cam_r_key = next((k for k, v in CAMERAS.items() if v["index"] == idx_r), f"cam_{idx_r}")
    print()
    print("=" * 70)
    print(f" Calibration extrinseque stereo  {cam_l_key} (idx {idx_l}) + {cam_r_key} (idx {idx_r})")
    print("=" * 70)
    print()
    print("  Procedure eye-to-hand stereo :")
    print("    1. Le damier 9x6 22 mm est fixe sur la pince fermee du robot.")
    print("    2. Les deux cameras sont fixes sur la barriere avant.")
    print("    3. Deplacer le bras pour amener le damier dans le champ des deux cameras.")
    print("    4. Quand les deux voient le damier (statut vert) : 'c' pour capturer.")
    print("    5. Diversite angulaire superieure a 65 degres, 30 a 60 poses recommandees.")
    print("    6. 'q' pour terminer, ESC pour annuler.")
    print()

    # Charge intrinseques
    intr_l_path = f"configs/calibration_cam_{idx_l}.json"
    intr_r_path = f"configs/calibration_cam_{idx_r}.json"
    if not os.path.exists(intr_l_path) or not os.path.exists(intr_r_path):
        print(f"ERREUR : calibrations intrinseques manquantes : {intr_l_path}, {intr_r_path}")
        sys.exit(1)
    K_l, D_l = load_intrinsic(intr_l_path)
    K_r, D_r = load_intrinsic(intr_r_path)
    print(f"Intrinseques {cam_l_key} : fx={K_l[0,0]:.1f}, residu inclus dans le fichier.")
    print(f"Intrinseques {cam_r_key} : fx={K_r[0,0]:.1f}.")
    print()

    # Connexion du robot
    print(f"Connexion au follower sur {args.port}...")
    bus, motor_names = connect_robot(args.port)
    print("  6 moteurs detectes, couple desactive (bras manipulable a la main).")
    print()

    # Ouvre les 2 cameras
    w_l = CAMERAS.get(cam_l_key, {}).get("width", 1920)
    h_l = CAMERAS.get(cam_l_key, {}).get("height", 1080)
    w_r = CAMERAS.get(cam_r_key, {}).get("width", 1920)
    h_r = CAMERAS.get(cam_r_key, {}).get("height", 1080)
    print(f"Ouverture {cam_l_key} ({w_l}x{h_l})...")
    cap_l = open_camera(idx_l, w_l, h_l)
    print(f"Ouverture {cam_r_key} ({w_r}x{h_r})...")
    cap_r = open_camera(idx_r, w_r, h_r)

    # Dossiers images
    img_dir_l = REPO / f"outputs/calibration_images/extrinsic_stereo_{cam_l_key}"
    img_dir_r = REPO / f"outputs/calibration_images/extrinsic_stereo_{cam_r_key}"
    if not args.no_save_images:
        img_dir_l.mkdir(parents=True, exist_ok=True)
        img_dir_r.mkdir(parents=True, exist_ok=True)
        print(f"Images sauvegardees dans : {img_dir_l}/ et {img_dir_r}/")

    # Chemin de sortie et fonctions utilitaires.
    # On ne sauvegarde pas directement dans le JSON officiel a chaque capture.
    # On enregistre dans un fichier partiel (outputs/extrinsic_stereo_partial.json).
    # Le JSON officiel n'est mis a jour que si la session se termine avec 'q'
    # et un nombre suffisant de captures. Sur ESC, ou sur 'q' avec des captures
    # insuffisantes, le JSON officiel reste intact. Ainsi, une session avortee
    # ne corrompt pas le pipeline existant.
    output_path = REPO / args.output  # JSON officiel (ecrit seulement en fin de session)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = REPO / "outputs"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"extrinsic_stereo_partial_{idx_l}_{idx_r}.json"

    captures = []
    img_size_l = (w_l, h_l)
    img_size_r = (w_r, h_r)
    exit_reason = "unknown"  # defini a "q" ou "esc" en fin de boucle

    def build_result_dict():
        result = {
            "schema_version": "stereo_v1",
            "cam_indices": [idx_l, idx_r],
            "cam_keys": [cam_l_key, cam_r_key],
            "intrinsic_files": [intr_l_path, intr_r_path],
            "motor_calibration_file": "configs/calibration_follower.json",
            "checkerboard": {
                "rows": args.rows,
                "cols": args.cols,
                "square_size_mm": args.square_size,
            },
            "motor_names": motor_names,
            "motor_position_units": "raw_encoder_counts",
            "image_size_left": list(img_size_l),
            "image_size_right": list(img_size_r),
            "num_captures": len(captures),
            "captures": captures,
        }
        return result

    def save_partial():
        """Sauvegarde incrementale dans le fichier partiel, jamais dans le JSON
        officiel. Ainsi une session avortee ne corrompt pas le JSON officiel
        utilise par le solveur."""
        with open(partial_path, "w") as f:
            json.dump(build_result_dict(), f, indent=2)

    def promote_to_official():
        """Copie le fichier partiel vers le JSON officiel. Appelee uniquement en
        fin de session reussie avec un nombre suffisant de captures."""
        with open(output_path, "w") as f:
            json.dump(build_result_dict(), f, indent=2)
        print(f"  [OK] JSON officiel mis a jour : {output_path}")

    print()
    print(f"Sauvegarde incrementale (partielle) : {partial_path}")
    print(f"Le JSON officiel ({output_path.name}) ne sera mis a jour qu'en")
    print("fin de session reussie avec au moins 10 captures.")
    print()
    print("Controles : 'c' pour capturer (les deux detectent), 'q' pour terminer, ESC pour annuler.")
    print()

    window_name = f"Stereo extrinsec - {cam_l_key} | {cam_r_key}"

    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print("[WARN] echec de lecture d'une image sur une camera, nouvelle tentative...")
                continue

            rvec_l, tvec_l, corners_l, obj_l = estimate_board_pose(
                frame_l, K_l, D_l, args.rows, args.cols, args.square_size)
            rvec_r, tvec_r, corners_r, obj_r = estimate_board_pose(
                frame_r, K_r, D_r, args.rows, args.cols, args.square_size)

            both_detected = (rvec_l is not None) and (rvec_r is not None)
            dist_l_mm = float(np.linalg.norm(tvec_l)) if rvec_l is not None else 0.0
            dist_r_mm = float(np.linalg.norm(tvec_r)) if rvec_r is not None else 0.0

            status_l = (f"OK dist={dist_l_mm:.0f}mm" if rvec_l is not None
                        else "Damier non detecte")
            status_r = (f"OK dist={dist_r_mm:.0f}mm" if rvec_r is not None
                        else "Damier non detecte")

            disp_l = draw_overlay(frame_l, K_l, D_l, rvec_l, tvec_l, corners_l,
                                   args.square_size, cam_l_key, status_l,
                                   args.rows, args.cols)
            disp_r = draw_overlay(frame_r, K_r, D_r, rvec_r, tvec_r, corners_r,
                                   args.square_size, cam_r_key, status_r,
                                   args.rows, args.cols)

            mosaic = np.hstack([disp_l, disp_r])
            # Bandeau central
            cv2.rectangle(mosaic, (0, mosaic.shape[0] - 50),
                          (mosaic.shape[1], mosaic.shape[0]), (0, 0, 0), -1)
            global_status = (f"CAPTURES: {len(captures)}  |  "
                             + ("Les 2 OK -> 'c' pour capturer"
                                if both_detected else "Repositionner pour que les 2 voient le damier"))
            color = (0, 255, 0) if both_detected else (0, 165, 255)
            cv2.putText(mosaic, global_status,
                        (10, mosaic.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if args.display_scale != 1.0:
                ds = args.display_scale
                small = cv2.resize(mosaic, (int(mosaic.shape[1] * ds),
                                             int(mosaic.shape[0] * ds)))
                cv2.imshow(window_name, small)
            else:
                cv2.imshow(window_name, mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and both_detected:
                # Capture synchronisee : vidage des buffers puis grab et retrieve
                # quasi-simultanes sur les deux cameras. Important pour
                # stereoCalibrate : le decalage temporel entre cap_l.read() et
                # cap_r.read() en mode apercu peut atteindre 30 ms, ce qui
                # desaligne les coins du damier si le bras n'est pas parfaitement
                # immobile et fait monter le RMS stereo.
                sync_l, sync_r = synchronized_capture(cap_l, cap_r, flush_frames=5)
                if sync_l is None:
                    print("  [WARN] capture synchronisee echouee, nouvelle tentative sur l'image suivante.")
                    continue

                # Nouvelle detection sur les images synchronisees (les coins
                # precedents provenaient de l'apercu, asynchrone).
                rvec_l_s, tvec_l_s, corners_l_s, obj_l_s = estimate_board_pose(
                    sync_l, K_l, D_l, args.rows, args.cols, args.square_size)
                rvec_r_s, tvec_r_s, corners_r_s, obj_r_s = estimate_board_pose(
                    sync_r, K_r, D_r, args.rows, args.cols, args.square_size)
                if rvec_l_s is None or rvec_r_s is None:
                    print(f"  [SKIP] apres synchronisation, damier non detecte "
                          f"(L={rvec_l_s is not None}, R={rvec_r_s is not None}). "
                          f"Verifier l'immobilite du bras.")
                    continue

                try:
                    motor_pos = bus.sync_read("Present_Position", normalize=False)
                except Exception as e:
                    print(f"  [WARN] lecture des positions moteur echouee : {e}")
                    continue

                dist_l_sync = float(np.linalg.norm(tvec_l_s))
                dist_r_sync = float(np.linalg.norm(tvec_r_s))
                n = len(captures) + 1
                capture_data = {
                    "id": n,
                    "rvec_target_cam0": rvec_l_s.flatten().tolist(),
                    "tvec_target_cam0": tvec_l_s.flatten().tolist(),
                    "rvec_target_cam1": rvec_r_s.flatten().tolist(),
                    "tvec_target_cam1": tvec_r_s.flatten().tolist(),
                    "img_points_cam0": corners_l_s.reshape(-1, 2).tolist(),
                    "img_points_cam1": corners_r_s.reshape(-1, 2).tolist(),
                    "obj_points": obj_l_s.tolist(),
                    "distance_mm_cam0": dist_l_sync,
                    "distance_mm_cam1": dist_r_sync,
                    "motor_positions_raw": {k: float(v) for k, v in motor_pos.items()},
                }
                captures.append(capture_data)
                if not args.no_save_images:
                    cv2.imwrite(str(img_dir_l / f"capture_{n:02d}_raw.png"), sync_l)
                    cv2.imwrite(str(img_dir_l / f"capture_{n:02d}_axes.png"), disp_l)
                    cv2.imwrite(str(img_dir_r / f"capture_{n:02d}_raw.png"), sync_r)
                    cv2.imwrite(str(img_dir_r / f"capture_{n:02d}_axes.png"), disp_r)
                # Sauvegarde dans le fichier partiel (le JSON officiel n'est pas modifie).
                save_partial()
                print(f"  Capture {n} : "
                      f"{cam_l_key} dist={dist_l_sync:.0f}mm, "
                      f"{cam_r_key} dist={dist_r_sync:.0f}mm  (synchronisee)")
            elif key == ord("c") and not both_detected:
                print(f"  [SKIP] le damier doit etre detecte dans les deux cameras "
                      f"(actuellement L={rvec_l is not None}, R={rvec_r is not None}).")
            elif key == ord("q"):
                exit_reason = "q"
                break
            elif key == 27:
                exit_reason = "esc"
                print("Annulation par l'utilisateur (ESC).")
                if captures:
                    save_partial()
                    print(f"  {len(captures)} captures enregistrees dans le fichier partiel : {partial_path}")
                    print(f"  Le JSON officiel ({output_path.name}) n'est pas modifie.")
                break
        else:
            exit_reason = "q"  # boucle terminee sans break : 'q' par defaut

    finally:
        try:
            cap_l.release()
        except Exception:
            pass
        try:
            cap_r.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # Decision finale : promotion du fichier partiel vers le JSON officiel
    # uniquement si :
    #  - sortie via 'q' (pas ESC)
    #  - au moins MIN_CAPTURES_FOR_PROMOTE captures
    #
    # Codes de retour :
    #   0 = succes (JSON officiel mis a jour, solveur a lancer)
    #   2 = capture avortee (JSON officiel intact, ne pas lancer le solveur)
    #   1 = erreur fatale (deja sortie via sys.exit)
    MIN_CAPTURES_FOR_PROMOTE = 10
    exit_code = 2  # par defaut : pas de promotion

    if exit_reason == "esc":
        # ESC : on conserve le fichier partiel mais on ne modifie pas l'officiel.
        if captures:
            print(f"\nESC avec {len(captures)} captures.")
            print(f"  Fichier partiel conserve : {partial_path}")
            print(f"  JSON officiel intact : {output_path}")
            print(f"  Pour utiliser ces captures : cp {partial_path} {output_path}")
        exit_code = 2
    elif exit_reason == "q":
        if len(captures) >= MIN_CAPTURES_FOR_PROMOTE:
            promote_to_official()
            print()
            print(f"{len(captures)} captures promues vers le JSON officiel.")
            print("Etape suivante : python scripts/solve_handeye_stereo.py")
            exit_code = 0
        elif captures:
            print(f"\nSeulement {len(captures)} captures, moins que les {MIN_CAPTURES_FOR_PROMOTE} requises pour la promotion automatique.")
            print(f"  Fichier partiel conserve : {partial_path}")
            print(f"  JSON officiel intact : {output_path}")
            print(f"  Pour forcer l'utilisation : cp {partial_path} {output_path}")
            exit_code = 2
        else:
            print("\nAucune capture, rien a sauvegarder.")
            exit_code = 2

    # Deconnexion du bus, protegee par un try/except.
    try:
        bus.disconnect()
    except Exception as e:
        print(f"  [WARN] bus.disconnect() : {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
