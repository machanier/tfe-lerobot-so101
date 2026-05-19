#!/usr/bin/env python3
"""
recalibrate_handeye_stereo.py - Orchestrateur B3b : recalibration STEREO
conjointe de cam_0 + cam_1 en 1 commande.

POURQUOI :
La calibration separee de cam_0/cam_1 produit ~+40mm de biais Y residuel
qu'on observe systematiquement dans le refinement #1. Le probleme : les
2 cameras sont calibrees independamment, donc leurs erreurs ne s'annulent
pas a la triangulation stereo. Solution : capturer simultanement et resoudre
conjointement (cv2.stereoCalibrate + deduction).

UNE SEULE commande pour l'utilisateur :
    python scripts/recalibrate_handeye_stereo.py

Etapes :
  1. Backup automatique de handeye_cam_0.json, handeye_cam_1.json,
     bias_correction.json.
  2. Capture stereo simultanee (calibrate_extrinsic_stereo.py interactif).
  3. Solve stereo conjoint (solve_handeye_stereo.py auto).
  4. Affichage verdict + comparaison avant/apres.
  5. Si residus OK : propose de mettre dy=0 dans bias_correction.json.

CRITERES DE SUCCES :
  - Stereo RMS reprojection : < 0.5 px (calibration intra-stereo precise)
  - cam_0 hand-eye          : mean <= 5mm, max <= 12mm
  - cam_1 (deduit)          : mean <= 5mm, max <= 12mm (coherent avec cam_0)

DAMIER : 9x6 asymetrique 22mm colle sur la pince FERMEE du robot.

DUREE estimee : 30-45 min (30-60 captures simultanees).
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
        f"apres recalibration STEREO conjointe cam_0+cam_1. Valeurs precedentes : {old}."
    )
    json.dump(data, open(path, "w"), indent=2)
    print(f"  [OK] bias_correction.json mis a dx=dy=dz=0 (backup conserve).")


def main():
    p = argparse.ArgumentParser(description="Recalibration hand-eye STEREO conjointe (B3b).")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--square-size", type=float, default=22.0)
    p.add_argument("--cam-indices", nargs=2, type=int, default=[0, 1])
    p.add_argument("--skip-capture", action="store_true",
                   help="Saute la capture (utilise un JSON deja existant pour debug solve)")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx_l, idx_r = args.cam_indices

    banner("RECALIBRATION HAND-EYE STEREO CONJOINTE (B3b)")
    print(f"  damier         : {args.cols} x {args.rows} cases @ {args.square_size}mm")
    print(f"  cameras        : cam_{idx_l} + cam_{idx_r} (capture SIMULTANEE)")
    print(f"  backup suffix  : before_B3b_{stamp}")
    print()
    print("  POURQUOI cette procedure ?")
    print("    La calibration actuelle traite cam_0 et cam_1 separement. Leurs")
    print("    erreurs hand-eye (~6mm chacune) s'additionnent geometriquement a")
    print("    la triangulation stereo -> biais Y +40mm constate.")
    print("    En calibrant CONJOINTEMENT (cv2.stereoCalibrate + deduction), les")
    print("    2 calibrations deviennent COHERENTES par construction : la difference")
    print("    entre les 2 cameras est ~0.5mm precis -> le biais s'annule.")
    print()
    print("  PRE-REQUIS HARDWARE :")
    print("    - Damier 9x6 22mm COLLE sur la pince FERMEE (eye-to-hand).")
    print("    - Les 2 cameras a leur position definitive sur la barriere.")
    print()
    print("  DUREE : 30-45 min (30-60 captures simultanees).")
    print()
    if not confirm("  Pret a demarrer ?"):
        print("Annule.")
        return

    # ---- Backups ----
    banner("PHASE 1 : Backups", char="-")
    for i in args.cam_indices:
        backup_file(REPO / f"configs/handeye_cam_{i}.json", f"before_B3b_{stamp}")
    backup_file(REPO / "configs/extrinsic_capture_stereo.json", f"before_B3b_{stamp}")

    # Lit residus avant pour comparaison finale
    before = {
        idx_l: read_residuals(REPO / f"configs/handeye_cam_{idx_l}.json"),
        idx_r: read_residuals(REPO / f"configs/handeye_cam_{idx_r}.json"),
    }

    # ---- Capture stereo ----
    if not args.skip_capture:
        banner("PHASE 2 : Capture stereo simultanee (interactif)", char="-")
        print("  Le script va ouvrir une fenetre avec les 2 vues cote a cote.")
        print("  Le damier doit etre detecte dans LES DEUX cameras pour pouvoir capturer.")
        print("  Bouge le bras pour varier les poses (>30 captures, diversite angulaire).")
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
        if rc != 0:
            print(f"\n!! Capture stereo a echoue (return code {rc}).")
            print("   Le JSON est peut-etre sauve quand meme (sauvegarde incrementale).")
            print("   Tu peux relancer juste le solve avec : ")
            print("     python scripts/recalibrate_handeye_stereo.py --skip-capture")
            return

    # ---- Solve stereo ----
    banner("PHASE 3 : Resolution stereo conjointe (auto)", char="-")
    rc = subprocess.run([
        sys.executable, str(REPO / "scripts" / "solve_handeye_stereo.py"),
    ], cwd=REPO).returncode
    if rc != 0:
        print(f"\n!! Solve stereo a echoue (return code {rc}).")
        print("   Tu peux relancer avec : python scripts/solve_handeye_stereo.py")
        return

    # ---- Comparaison avant/apres ----
    banner("PHASE 4 : Comparaison avant / apres")
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
        verdict_str = "OK" if ok else f"INSUFF (cible mean<=5, max<=12)"
        print(f"  cam_{i}    {(b.get('mean') or 0):>10.2f}mm "
              f"{(a.get('mean') or 0):>10.2f}mm "
              f"{(b.get('max') or 0):>10.2f}mm "
              f"{(a.get('max') or 0):>10.2f}mm  {verdict_str}")
        if not ok:
            all_ok = False

    # Stereo info
    si_path = REPO / "configs/handeye_stereo_info.json"
    if si_path.exists():
        si = json.load(open(si_path))
        print()
        print(f"  Stereo RMS reproj : {si['stereo_rms_reprojection_px']:.3f} px  "
              f"({'BON' if si['stereo_rms_reprojection_px'] < 0.5 else 'ELEVE >0.5'})")
        print(f"  Baseline cam_0->cam_1 : {si['baseline_mm']:.1f} mm")

    # ---- Validation pipeline ----
    banner("PHASE 5 : Validation pipeline")
    print("  Lancement de check_calibration.py...")
    subprocess.run([sys.executable, str(REPO / "scripts" / "check_calibration.py")],
                   cwd=REPO)

    gt_path = REPO / "configs/perception/gt_test.json"
    if gt_path.exists():
        print()
        print("  Validation 3D contre ground truth (gt_test.json)...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "check_perception.py"),
                        "--gt", str(gt_path)], cwd=REPO)
    else:
        print(f"  [INFO] gt_test.json absent, validation 3D sautee.")

    # ---- Bias correction ----
    banner("PHASE 6 : bias_correction.json")
    if all_ok:
        print("  ✓ Les 2 cams sont OK.")
        print("  Tu peux desactiver bias_correction.json (mise a dx=dy=dz=0).")
        print("  Test rapide en parallele a recommander :")
        print("    python scripts/pick_and_place.py --target orange_cube --detector hf --display")
        print("  Si le refinement #1 corrige <10mm en Y, B3b a reussi.")
        print()
        if ask_yes_no("  Desactiver bias_correction.json maintenant ?", default_no=False):
            disable_bias(stamp)
        else:
            print("  bias_correction.json laisse en l'etat.")
            print("  ATTENTION : avec residus OK + bias actif, sur-correction probable.")
            print("  Pour le desactiver plus tard :")
            print("    python -c \"import json; p='configs/perception/bias_correction.json'; "
                  "d=json.load(open(p)); d['dy_mm']=0; d['dx_mm']=0; d['dz_mm']=0; "
                  "json.dump(d, open(p,'w'), indent=2)\"")
    else:
        print("  ✗ Au moins une cam n'est pas OK.")
        print("  bias_correction.json reste actif comme filet de securite.")
        print()
        print("  Causes possibles :")
        print("    - pas assez de diversite angulaire (vise >65deg ecart moyen)")
        print("    - damier deforme ou impression de mauvaise qualite")
        print("    - une camera mal alignee (structure 3D qui a bouge)")
        print("    - rms stereo > 0.5 px = probleme de coherence des coins detectes")

    banner("TERMINE")
    print(f"  Backups conserves : configs/*.before_B3b_{stamp}.backup.json")
    print(f"  Pour revert : ")
    print(f"    for i in {idx_l} {idx_r}; do")
    print(f"      cp configs/handeye_cam_$i.before_B3b_{stamp}.backup.json configs/handeye_cam_$i.json")
    print(f"    done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n!! Interrompu par utilisateur.")
        sys.exit(130)
