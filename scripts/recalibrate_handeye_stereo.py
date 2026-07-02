#!/usr/bin/env python3
"""Recalibration hand-eye stereo conjointe de cam_0 et cam_1 en une commande.

La calibration separee de cam_0 et cam_1 laisse un biais Y residuel d'environ
40 mm a la triangulation stereo : les deux cameras etant calibrees
independamment, leurs erreurs ne s'annulent pas. Ce script capture les deux
vues simultanement et resout conjointement (cv2.stereoCalibrate + deduction),
de sorte que les deux calibrations restent coherentes par construction.

Usage :
    python scripts/recalibrate_handeye_stereo.py

Etapes :
  1. Sauvegarde de handeye_cam_0.json, handeye_cam_1.json et
     extrinsic_capture_stereo.json.
  2. Capture stereo simultanee (calibrate_extrinsic_stereo.py, interactif).
  3. Resolution stereo conjointe (solve_handeye_stereo.py, automatique).
  4. Comparaison avant/apres et verdict.
  5. Si les residus sont conformes, proposition de neutraliser
     bias_correction.json.

Criteres de succes :
  - RMS de reprojection stereo : < 0.5 px.
  - cam_0 hand-eye             : moyenne <= 5 mm, maximum <= 12 mm.
  - cam_1 (deduit)             : moyenne <= 5 mm, maximum <= 12 mm.

Damier : 9x6 asymetrique, cases de 22 mm, colle sur la pince fermee du robot.

Duree estimee : 30 a 45 min (30 a 60 captures simultanees).

Entrees  : configs/handeye_cam_{0,1}.json, configs/extrinsic_capture_stereo.json.
Sorties  : configs/handeye_cam_{0,1}.json et configs/handeye_stereo_info.json
           mis a jour, sauvegardes horodatees des fichiers modifies.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def banner(title, char="=", width=70):
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def confirm(prompt):
    try:
        input(prompt + " [ENTREE pour continuer, Ctrl+C pour annuler] : ")
        return True
    except (EOFError, KeyboardInterrupt):
        return False


def ask_yes_no(prompt, default_no=True):
    default = "[o/N]" if default_no else "[O/n]"
    try:
        ans = input(f"{prompt} {default} : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return not default_no
    return ans in ("o", "oui", "y", "yes")


def backup_file(path: Path, suffix: str):
    if not path.exists():
        return None
    bk = path.with_name(f"{path.stem}.{suffix}.backup{path.suffix}")
    shutil.copy(path, bk)
    print(f"  [backup] {path.name} -> {bk.name}")
    return bk


def read_residuals(path: Path):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        r = d.get("residuals", {})
        return {
            "mean": r.get("translation_mean_dev_mm"),
            "max": r.get("translation_max_dev_mm"),
            "median": r.get("translation_median_dev_mm"),
            "rot_mean": r.get("rotation_mean_dev_deg"),
            "rot_max": r.get("rotation_max_dev_deg"),
            "n_used": d.get("n_poses_used"),
            "n_total": d.get("n_poses_total"),
        }
    except Exception:
        return None


def is_ok(mean, mx):
    return mean is not None and mean <= 5.0 and mx is not None and mx <= 12.0


def disable_bias(stamp):
    path = REPO / "configs/perception/bias_correction.json"
    if not path.exists():
        return
    backup_file(path, f"before_B3b_{stamp}")
    data = json.load(open(path))
    old = {k: data.get(k) for k in ("dx_mm", "dy_mm", "dz_mm")}
    data["dx_mm"] = 0; data["dy_mm"] = 0; data["dz_mm"] = 0
    data["_disabled_by_B3b"] = (
        f"Compensation desactivee le {datetime.now().isoformat(timespec='seconds')} "
        f"apres recalibration stereo conjointe cam_0+cam_1. Valeurs precedentes : {old}."
    )
    json.dump(data, open(path, "w"), indent=2)
    print("  [OK] bias_correction.json mis a dx=dy=dz=0 (sauvegarde conservee).")


def main():
    p = argparse.ArgumentParser(description="Recalibration hand-eye stereo conjointe de cam_0 et cam_1.")
    p.add_argument("--rows", type=int, default=6,
                   help="Nombre de coins internes du damier en hauteur (defaut : 6).")
    p.add_argument("--cols", type=int, default=9,
                   help="Nombre de coins internes du damier en largeur (defaut : 9).")
    p.add_argument("--square-size", type=float, default=22.0,
                   help="Taille d'une case du damier en millimetres (defaut : 22.0).")
    p.add_argument("--cam-indices", nargs=2, type=int, default=[0, 1],
                   help="Indices des deux cameras a recalibrer (defaut : 0 1).")
    p.add_argument("--skip-capture", action="store_true",
                   help="Reutilise la capture stereo existante sans en refaire une "
                        "et lance directement la resolution (defaut : desactive).")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx_l, idx_r = args.cam_indices

    banner("Recalibration hand-eye stereo conjointe")
    print(f"  damier         : {args.cols} x {args.rows} cases @ {args.square_size}mm")
    print(f"  cameras        : cam_{idx_l} + cam_{idx_r} (capture simultanee)")
    print(f"  suffixe backup : before_B3b_{stamp}")
    print()
    print("  Principe de la procedure :")
    print("    La calibration actuelle traite cam_0 et cam_1 separement. Leurs")
    print("    erreurs hand-eye (environ 6 mm chacune) s'additionnent")
    print("    geometriquement a la triangulation stereo, d'ou le biais Y d'environ")
    print("    40 mm constate. En calibrant conjointement (cv2.stereoCalibrate +")
    print("    deduction), les deux calibrations deviennent coherentes par")
    print("    construction : l'ecart entre les deux cameras tombe autour de")
    print("    0.5 mm et le biais s'annule.")
    print()
    print("  Pre-requis materiel :")
    print("    - Damier 9x6 22 mm colle sur la pince fermee (eye-to-hand).")
    print("    - Les deux cameras a leur position definitive sur la barriere.")
    print()
    print("  Duree : 30 a 45 min (30 a 60 captures simultanees).")
    print()
    if not confirm("  Pret a demarrer ?"):
        print("Annule.")
        return

    # ---- Sauvegardes ----
    banner("Phase 1 : sauvegardes", char="-")
    for i in args.cam_indices:
        backup_file(REPO / f"configs/handeye_cam_{i}.json", f"before_B3b_{stamp}")
    backup_file(REPO / "configs/extrinsic_capture_stereo.json", f"before_B3b_{stamp}")

    # Lecture des residus avant recalibration, pour la comparaison finale.
    before = {
        idx_l: read_residuals(REPO / f"configs/handeye_cam_{idx_l}.json"),
        idx_r: read_residuals(REPO / f"configs/handeye_cam_{idx_r}.json"),
    }

    # ---- Capture stereo ----
    if not args.skip_capture:
        banner("Phase 2 : capture stereo simultanee (interactif)", char="-")
        print("  Le script ouvre une fenetre avec les deux vues cote a cote.")
        print("  Le damier doit etre detecte dans les deux cameras pour capturer.")
        print("  Deplacer le bras pour varier les poses (plus de 30 captures,")
        print("  diversite angulaire).")
        print()
        if not confirm("  Pret pour la capture ?"):
            return
        rc = subprocess.run([
            sys.executable, str(REPO / "scripts" / "calibrate_extrinsic_stereo.py"),
            "--cam-indices", str(idx_l), str(idx_r),
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--square-size", str(args.square_size),
        ], cwd=REPO).returncode

        # Codes de retour de calibrate_extrinsic_stereo.py :
        #   0     : succes (JSON officiel mis a jour).
        #   2     : capture interrompue (ESC ou moins de 10 captures, JSON intact).
        #   autre : erreur fatale.
        if rc == 2:
            banner("Capture interrompue, resolution non lancee", char="!")
            print("  La capture a ete annulee (ESC) ou comptait moins de 10 captures.")
            print("  Le JSON officiel n'a pas ete modifie : les calibrations")
            print("  handeye_cam_0.json et handeye_cam_1.json restent inchangees.")
            print()
            print("  Pour reprendre :")
            print("    python scripts/recalibrate_handeye_stereo.py")
            print()
            print("  Pour reutiliser une capture partielle precedente :")
            print("    cp outputs/extrinsic_stereo_partial_0_1.json configs/extrinsic_capture_stereo.json")
            print("    python scripts/recalibrate_handeye_stereo.py --skip-capture")
            return
        if rc != 0:
            print(f"\n!! La capture a echoue avec un code inattendu ({rc}).")
            print("   Le JSON officiel n'a pas ete modifie.")
            return

    # ---- Resolution stereo ----
    banner("Phase 3 : resolution stereo conjointe (automatique)", char="-")
    rc = subprocess.run([
        sys.executable, str(REPO / "scripts" / "solve_handeye_stereo.py"),
    ], cwd=REPO).returncode
    if rc != 0:
        print(f"\n!! La resolution stereo a echoue (code de retour {rc}).")
        print("   Relancer avec : python scripts/solve_handeye_stereo.py")
        return

    # ---- Comparaison avant/apres ----
    banner("Phase 4 : comparaison avant / apres")
    after = {
        idx_l: read_residuals(REPO / f"configs/handeye_cam_{idx_l}.json"),
        idx_r: read_residuals(REPO / f"configs/handeye_cam_{idx_r}.json"),
    }

    print(f"  {'cam':<8} {'mean avant':>12} {'mean apres':>12} {'max avant':>12} {'max apres':>12}  {'verdict apres'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}  {'-'*25}")
    all_ok = True
    for i in args.cam_indices:
        b = before[i] or {}
        a = after[i] or {}
        ok = is_ok(a.get("mean"), a.get("max"))
        verdict_str = "OK" if ok else "insuffisant (cible : moyenne<=5, max<=12)"
        print(f"  cam_{i}    {(b.get('mean') or 0):>10.2f}mm "
              f"{(a.get('mean') or 0):>10.2f}mm "
              f"{(b.get('max') or 0):>10.2f}mm "
              f"{(a.get('max') or 0):>10.2f}mm  {verdict_str}")
        if not ok:
            all_ok = False

    # Informations stereo
    si_path = REPO / "configs/handeye_stereo_info.json"
    if si_path.exists():
        si = json.load(open(si_path))
        print()
        print(f"  RMS de reprojection stereo : {si['stereo_rms_reprojection_px']:.3f} px  "
              f"({'bon' if si['stereo_rms_reprojection_px'] < 0.5 else 'eleve, >0.5'})")
        print(f"  Baseline cam_0->cam_1 : {si['baseline_mm']:.1f} mm")

    # ---- Validation de la pipeline ----
    banner("Phase 5 : validation de la pipeline")
    print("  Lancement de check_calibration.py...")
    subprocess.run([sys.executable, str(REPO / "scripts" / "check_calibration.py")],
                   cwd=REPO)

    gt_path = REPO / "configs/perception/gt_test.json"
    if gt_path.exists():
        print()
        print("  Validation 3D contre la reference (gt_test.json)...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "check_perception.py"),
                        "--gt", str(gt_path)], cwd=REPO)
    else:
        print("  [INFO] gt_test.json absent, validation 3D ignoree.")

    # ---- Correction de biais ----
    banner("Phase 6 : bias_correction.json")
    if all_ok:
        print("  Les deux cameras sont conformes.")
        print("  La correction bias_correction.json peut etre neutralisee (dx=dy=dz=0).")
        print("  Verification rapide recommandee :")
        print("    python scripts/pick_and_place.py --target orange_cube --detector hf --display")
        print("  Si le refinement #1 corrige moins de 10 mm en Y, la recalibration a reussi.")
        print()
        if ask_yes_no("  Neutraliser bias_correction.json maintenant ?", default_no=False):
            disable_bias(stamp)
        else:
            print("  bias_correction.json laisse en l'etat.")
            print("  Avec des residus conformes et le biais actif, une sur-correction")
            print("  est probable. Pour le neutraliser plus tard :")
            print("    python -c \"import json; p='configs/perception/bias_correction.json'; "
                  "d=json.load(open(p)); d['dy_mm']=0; d['dx_mm']=0; d['dz_mm']=0; "
                  "json.dump(d, open(p,'w'), indent=2)\"")
    else:
        print("  Au moins une camera n'est pas conforme.")
        print("  bias_correction.json reste actif comme filet de securite.")
        print()
        print("  Causes possibles :")
        print("    - diversite angulaire insuffisante (viser plus de 65 deg d'ecart moyen) ;")
        print("    - damier deforme ou impression de mauvaise qualite ;")
        print("    - une camera mal alignee (structure deplacee) ;")
        print("    - RMS stereo > 0.5 px : incoherence des coins detectes.")

    banner("Termine")
    print(f"  Sauvegardes conservees : configs/*.before_B3b_{stamp}.backup.json")
    print("  Pour revenir en arriere :")
    print(f"    for i in {idx_l} {idx_r}; do")
    print(f"      cp configs/handeye_cam_$i.before_B3b_{stamp}.backup.json configs/handeye_cam_$i.json")
    print(f"    done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n!! Interrompu par l'utilisateur.")
        sys.exit(130)
