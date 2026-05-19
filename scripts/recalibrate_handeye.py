#!/usr/bin/env python3
"""
recalibrate_handeye.py - Recalibration hand-eye complete des 3 cameras (B3).

OBJECTIF : ramener les residus hand-eye au plancher de bruit du SO-101
pour eliminer le biais Y residuel ~+40mm vu dans le refinement #1, et
pouvoir desactiver bias_correction.json (workaround actuel).

UNE SEULE commande pour l'utilisateur :
    python scripts/recalibrate_handeye.py

Le script orchestre tout :
  1. Backups automatiques (handeye_cam_*.json, extrinsic_capture_cam_*.json,
     bias_correction.json) avec timestamp.
  2. Pour cam_0, cam_1, cam_2 sequentiellement :
       - Affiche la procedure adaptee (eye-to-hand vs eye-in-hand).
       - Attend ENTREE.
       - Lance calibrate_extrinsic.py --index <i> (interactif, OpenCV window).
       - Lance solve_handeye_cam.py --index <i>.
       - Affiche residus + verdict (OK / ACCEPTABLE / INSUFFISANT).
  3. check_calibration.py + check_perception.py (si gt_test.json existe).
  4. Si tous les residus OK : propose de desactiver bias_correction.json.

CRITERES DE SUCCES (cf Tsai-Lenz 1989, Park-Martin 1994, et plancher SO-101) :
  - cam_0, cam_1 (eye-to-hand) : mean <= 5mm, max <= 12mm
  - cam_2 (eye-in-hand)        : mean <= 3mm, max <= 6mm

DAMIER : 9x6 asymetrique par defaut (cf D2 dans PROJECT_STATUS.md). 22mm/case.

USAGE COURANT :
    # Refait tout (3 cams) :
    python scripts/recalibrate_handeye.py

    # Refait seulement cam_0 (si l'une a echoue) :
    python scripts/recalibrate_handeye.py --cams 0

    # Avec damier non-standard :
    python scripts/recalibrate_handeye.py --cols 7 --rows 7 --square-size 25

ANNULATION : Ctrl+C a tout moment. Les backups conserves dans configs/
permettent de revenir a l'etat precedent en copiant les .before_B3_*.backup
vers les originaux.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Mapping index -> role d'apres scripts/config.py
ROLE_FOR_INDEX = {0: "eye_to_hand", 1: "eye_to_hand", 2: "eye_in_hand"}


def banner(title: str, char: str = "=", width: int = 70):
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def confirm(prompt: str) -> bool:
    """Attend ENTREE (ou ctrl+c pour annuler). Renvoie True si confirme."""
    try:
        input(prompt + " [ENTREE pour continuer, Ctrl+C pour annuler] : ")
        return True
    except (EOFError, KeyboardInterrupt):
        return False


def ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    default = "[o/N]" if default_no else "[O/n]"
    try:
        ans = input(f"{prompt} {default} : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return not default_no
    return ans in ("o", "oui", "y", "yes")


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{suffix}.backup{path.suffix}")
    shutil.copy(path, backup)
    print(f"  [backup] {path.name} -> {backup.name}")
    return backup


def run_step(cmd: list[str], label: str) -> int:
    """Lance une sous-commande, propage stdin/stdout (necessaire pour les
    fenetres OpenCV interactives de calibrate_extrinsic.py)."""
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"  !! {label} a echoue (return code {rc})")
    return rc


def read_residuals(index: int) -> dict | None:
    """Lit les residus du handeye_cam_<index>.json apres solve."""
    path = REPO / "configs" / f"handeye_cam_{index}.json"
    if not path.exists():
        return None
    try:
        data = json.load(open(path))
        r = data.get("residuals", {})
        return {
            "mean_trans_mm": r.get("translation_mean_dev_mm"),
            "max_trans_mm": r.get("translation_max_dev_mm"),
            "median_trans_mm": r.get("translation_median_dev_mm"),
            "mean_rot_deg": r.get("rotation_mean_dev_deg"),
            "max_rot_deg": r.get("rotation_max_dev_deg"),
            "n_poses_used": data.get("n_poses_used"),
            "n_poses_total": data.get("n_poses_total"),
        }
    except Exception as e:
        print(f"  [WARN] impossible de lire {path.name} : {e}")
        return None


def verdict_for_cam(role: str, mean_mm: float, max_mm: float) -> tuple[str, bool]:
    """Verdict + bool 'is_ok' pour decider de desactiver bias_correction."""
    if role == "eye_in_hand":
        if mean_mm <= 3 and max_mm <= 6:
            return ("OK (mean<=3mm, max<=6mm)", True)
        elif mean_mm <= 5 and max_mm <= 10:
            return ("ACCEPTABLE (mean 3-5mm)", False)
        else:
            return ("INSUFFISANT (mean >5mm)", False)
    else:
        if mean_mm <= 5 and max_mm <= 12:
            return ("OK (mean<=5mm, max<=12mm)", True)
        elif mean_mm <= 8 and max_mm <= 16:
            return ("ACCEPTABLE (mean 5-8mm)", False)
        else:
            return ("INSUFFISANT (mean >8mm)", False)


def calibrate_one_cam(index: int, args) -> dict:
    """Capture + solve pour 1 camera. Retourne dict avec residus ou status."""
    role = ROLE_FOR_INDEX.get(index, "unknown")
    banner(f"CAMERA cam_{index} ({role})", char="#")

    if role == "eye_in_hand":
        print("  PROCEDURE EYE-IN-HAND (cam_2 montee sur la pince) :")
        print("    1. POSE le damier FIXE sur la table (bien plat, bien eclaire,")
        print("       au moins 30cm de la base du robot pour eviter les zones d'exclusion).")
        print("    2. Le script lance la capture. Tu vas TELEOPERER le bras")
        print("       (avec le leader si branche, sinon manuellement) pour amener")
        print("       cam_2 a 15-25 positions au-dessus du damier.")
        print("    3. Diversite angulaire >65deg : inclinaisons +/-30deg, rotations,")
        print("       distances variees entre 15 et 40cm de cam_2 au damier.")
        print("    4. 'c' pour capturer chaque pose, 'q' pour terminer.")
    else:
        print("  PROCEDURE EYE-TO-HAND (cam_0 ou cam_1 fixes sur la barriere) :")
        print("    1. COLLE le damier sur la PINCE FERMEE (bien centre, plat).")
        print("    2. La camera est FIXE. Tu vas teleoperer (ou bouger a la main)")
        print("       le BRAS pour amener le damier devant la camera.")
        print("    3. Diversite angulaire >65deg : inclinaisons +/-30deg, rotations,")
        print("       distances variees entre 30cm et 80cm de la camera au damier.")
        print("    4. Vise 50-70 poses (le solveur rejette les outliers, il restera")
        print("       typiquement 25-30 poses utiles).")
        print("    5. 'c' pour capturer chaque pose, 'q' pour terminer.")
    print()
    if not confirm("  Pret pour la capture ?"):
        return {"aborted": True}

    # Capture
    cmd_cap = [
        sys.executable, str(REPO / "scripts" / "calibrate_extrinsic.py"),
        "--index", str(index),
        "--rows", str(args.rows),
        "--cols", str(args.cols),
        "--square-size", str(args.square_size),
    ]
    if run_step(cmd_cap, f"capture cam_{index}") != 0:
        return {"capture_failed": True}

    # Solve
    print()
    print(f"  -> Resolution hand-eye cam_{index}...")
    cmd_solve = [
        sys.executable, str(REPO / "scripts" / "solve_handeye_cam.py"),
        "--index", str(index),
    ]
    if run_step(cmd_solve, f"solve cam_{index}") != 0:
        return {"solve_failed": True}

    # Lit residus
    res = read_residuals(index)
    if res is None or res.get("mean_trans_mm") is None:
        print(f"  [WARN] residus non lisibles pour cam_{index}")
        return {"unreadable": True}

    verdict, _ = verdict_for_cam(role, res["mean_trans_mm"], res["max_trans_mm"])
    print()
    print(f"  --- RESULTAT cam_{index} ---")
    print(f"    mean translation : {res['mean_trans_mm']:.2f} mm")
    print(f"    max translation  : {res['max_trans_mm']:.2f} mm")
    print(f"    mean rotation    : {res['mean_rot_deg']:.2f} deg")
    print(f"    max rotation     : {res['max_rot_deg']:.2f} deg")
    print(f"    poses utiles     : {res['n_poses_used']}/{res['n_poses_total']}")
    print(f"    verdict          : {verdict}")
    res["role"] = role
    res["verdict"] = verdict
    return res


def disable_bias_correction(stamp: str):
    """Met dx=dy=dz=0 dans bias_correction.json apres backup."""
    path = REPO / "configs" / "perception" / "bias_correction.json"
    if not path.exists():
        print("  [INFO] bias_correction.json absent, rien a desactiver.")
        return
    backup_file(path, f"before_B3_{stamp}")
    data = json.load(open(path))
    old = {k: data.get(k) for k in ("dx_mm", "dy_mm", "dz_mm")}
    data["dx_mm"] = 0
    data["dy_mm"] = 0
    data["dz_mm"] = 0
    data["_disabled_by_B3"] = (
        f"Compensation desactivee le {datetime.now().isoformat(timespec='seconds')} "
        f"apres recalibration hand-eye reussie. Valeurs precedentes : {old}. "
        "Le biais Y est maintenant dans la calibration elle-meme (residus <=5mm)."
    )
    json.dump(data, open(path, "w"), indent=2)
    print(f"  [OK] {path.name} mis a dx=dy=dz=0 (backup conserve).")
    print(f"       Tu peux verifier : python scripts/check_perception.py --gt configs/perception/gt_test.json")


def print_summary(all_residuals: dict, stamp: str):
    banner("RESUME GLOBAL")
    print(f"  Backups conserves avec suffix : before_B3_{stamp}")
    print()
    print(f"  {'cam':<6} {'role':<14} {'mean':>9} {'max':>9} {'verdict'}")
    print(f"  {'-'*6} {'-'*14} {'-'*9} {'-'*9} {'-'*30}")
    all_ok = True
    for i, res in all_residuals.items():
        if "mean_trans_mm" not in res:
            print(f"  cam_{i}  {ROLE_FOR_INDEX[i]:<14} (echec: {list(res.keys())[0]})")
            all_ok = False
            continue
        role = res["role"]
        m = res["mean_trans_mm"]
        mx = res["max_trans_mm"]
        verdict, is_ok = verdict_for_cam(role, m, mx)
        print(f"  cam_{i}  {role:<14} {m:>6.2f}mm {mx:>6.2f}mm  {verdict}")
        if not is_ok:
            all_ok = False
    return all_ok


def main():
    p = argparse.ArgumentParser(
        description="Recalibration hand-eye complete (B3 du plan TFE).",
    )
    p.add_argument("--cols", type=int, default=9,
                   help="Colonnes damier (defaut 9, asymetrique avec rows=6).")
    p.add_argument("--rows", type=int, default=6,
                   help="Lignes damier (defaut 6).")
    p.add_argument("--square-size", type=float, default=22.0,
                   help="Taille case en mm (defaut 22.0).")
    p.add_argument("--cams", nargs="+", type=int, default=[0, 1, 2],
                   help="Indices des cams a recalibrer (defaut : 0 1 2).")
    p.add_argument("--skip-validation", action="store_true",
                   help="Saute check_calibration + check_perception + bias_correction.")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    banner("RECALIBRATION HAND-EYE COMPLETE (B3)")
    print(f"  damier         : {args.cols} x {args.rows} cases @ {args.square_size}mm")
    print(f"  cameras        : cam_{', cam_'.join(map(str, args.cams))}")
    print(f"  backup suffix  : before_B3_{stamp}")
    print()
    print("  POURQUOI cette procedure ?")
    print("    Le pipeline actuel montre un biais Y residuel ~+40mm au refinement #1,")
    print("    signe d'une calibration hand-eye degradee. bias_correction.json")
    print("    compense la moyenne (-30mm) mais le residu varie selon la position.")
    print("    Cette recalibration vise a ramener les residus au plancher SO-101 :")
    print("    - cam_0, cam_1 (eye-to-hand) : mean <= 5mm, max <= 12mm")
    print("    - cam_2        (eye-in-hand) : mean <= 3mm, max <= 6mm")
    print()
    print("  DUREE estimee : 1h30-2h hardware (capture + solve par cam).")
    print()
    if not confirm("  Pret a demarrer ?"):
        print("Annule.")
        return

    # Backups initiaux
    banner("PHASE 1 : Backups des calibrations actuelles", char="-")
    for i in args.cams:
        backup_file(REPO / "configs" / f"handeye_cam_{i}.json",
                    f"before_B3_{stamp}")
        backup_file(REPO / "configs" / f"extrinsic_capture_cam_{i}.json",
                    f"before_B3_{stamp}")

    # Capture + solve par cam
    all_residuals: dict[int, dict] = {}
    for i in args.cams:
        res = calibrate_one_cam(i, args)
        all_residuals[i] = res
        if "aborted" in res:
            print(f"\n!! Procedure abandonnee a cam_{i}. Les autres cams non faites.")
            print("   Les backups sont conserves, tu peux relancer plus tard avec :")
            remaining = [c for c in args.cams if c >= i]
            print(f"   python scripts/recalibrate_handeye.py --cams {' '.join(map(str, remaining))}")
            return

    # Resume global
    all_ok = print_summary(all_residuals, stamp)

    # Validation + bias_correction
    if not args.skip_validation:
        banner("PHASE 2 : Validation globale")
        print("  Lancement de check_calibration.py...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "check_calibration.py")],
                       cwd=REPO)
        print()

        gt_path = REPO / "configs" / "perception" / "gt_test.json"
        if gt_path.exists():
            print("  Validation 3D contre ground truth (gt_test.json)...")
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "check_perception.py"),
                 "--gt", str(gt_path)],
                cwd=REPO,
            )
            print()
        else:
            print(f"  [INFO] {gt_path.name} absent, validation 3D sautee.")
            print("         Tu peux poser ton cube a une position mesuree au pied a coulisse,")
            print("         creer ce fichier (cf gt_test_example.json), et relancer :")
            print(f"         python scripts/check_perception.py --gt {gt_path}")
            print()

        banner("PHASE 3 : bias_correction.json")
        if all_ok:
            print("  ✓ Tous les residus respectent les criteres OK.")
            print("  Tu peux DESACTIVER bias_correction.json (dx=dy=dz=0).")
            print("  Le decalage est maintenant porte par la calibration elle-meme,")
            print("  plus besoin de compensation systematique.")
            print()
            if ask_yes_no("  Desactiver bias_correction.json maintenant ?", default_no=False):
                disable_bias_correction(stamp)
            else:
                print("  bias_correction.json laisse en l'etat (peut creer un sur-correction)")
                print("  ATTENTION : avec residus OK + bias toujours actif, le pipeline")
                print("  surcorrigera. Pense a le desactiver manuellement bientot :")
                print("    python -c \"import json; p='configs/perception/bias_correction.json'; "
                      "d=json.load(open(p)); d['dy_mm']=0; json.dump(d, open(p,'w'), indent=2)\"")
        else:
            print("  ✗ Au moins une cam est en dessous des criteres OK.")
            print("  bias_correction.json reste actif comme filet de securite.")
            print()
            print("  Cams a refaire :")
            for i, res in all_residuals.items():
                if res.get("mean_trans_mm"):
                    role = res["role"]
                    _, is_ok = verdict_for_cam(role, res["mean_trans_mm"], res["max_trans_mm"])
                    if not is_ok:
                        print(f"    python scripts/recalibrate_handeye.py --cams {i}")
            print()
            print("  Causes possibles d'un residu trop eleve :")
            print("    - pas assez de diversite angulaire (vise >65deg ecart moyen)")
            print("    - damier deforme ou impression de mauvaise qualite")
            print("    - mauvais eclairage (reflets sur le damier)")
            print("    - structure 3D de la barriere qui a bouge (cam mecaniquement deplacee)")

    banner("TERMINE")
    print(f"  Backups conserves : configs/*.before_B3_{stamp}.backup.json")
    print(f"  Pour revert toutes les cams :")
    print(f"    for i in 0 1 2; do")
    print(f"      cp configs/handeye_cam_$i.before_B3_{stamp}.backup.json configs/handeye_cam_$i.json")
    print(f"    done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n!! Interrompu par utilisateur.")
        sys.exit(130)
