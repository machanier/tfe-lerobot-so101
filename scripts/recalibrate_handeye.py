#!/usr/bin/env python3
"""Recalibration hand-eye complete des trois cameras.

Objectif : ramener les residus hand-eye au plancher de bruit du SO-101 afin
d'eliminer le biais Y residuel observe au refinement, et de pouvoir desactiver
la compensation bias_correction.json.

Usage :
    python scripts/recalibrate_handeye.py

Le script orchestre l'ensemble de la procedure :
  1. Sauvegardes horodatees des calibrations actuelles (handeye_cam_*.json,
     extrinsic_capture_cam_*.json, bias_correction.json).
  2. Pour cam_0, cam_1 et cam_2 successivement :
       - affichage de la procedure adaptee (eye-to-hand ou eye-in-hand) ;
       - attente d'une confirmation ;
       - capture via calibrate_extrinsic.py --index <i> (fenetre OpenCV) ;
       - resolution via solve_handeye_cam.py --index <i> ;
       - affichage des residus et du verdict (OK, ACCEPTABLE ou INSUFFISANT).
  3. Validation via check_calibration.py, puis check_perception.py si le
     fichier gt_test.json est present.
  4. Si tous les residus sont OK, proposition de desactiver bias_correction.json.

Criteres de succes (cf. Tsai-Lenz 1989, Park-Martin 1994 et plancher SO-101) :
  - cam_0, cam_1 (eye-to-hand) : moyenne <= 5mm, max <= 12mm
  - cam_2 (eye-in-hand)        : moyenne <= 3mm, max <= 6mm

Damier : 9x6 asymetrique par defaut, cases de 22mm.

Exemples d'usage :
    # Recalibrer les trois cameras :
    python scripts/recalibrate_handeye.py

    # Recalibrer uniquement cam_0 :
    python scripts/recalibrate_handeye.py --cams 0

    # Damier non standard :
    python scripts/recalibrate_handeye.py --cols 7 --rows 7 --square-size 25

Entrees : configurations existantes dans configs/. Sorties : calibrations mises
a jour et sauvegardes horodatees dans configs/.

Annulation : Ctrl+C a tout moment. Les sauvegardes conservees dans configs/
permettent de revenir a l'etat precedent en copiant les fichiers .before_B3_*.backup
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
    """Attend une confirmation clavier. Renvoie True si confirme, False sinon."""
    try:
        input(prompt + " [Entree pour continuer, Ctrl+C pour annuler] : ")
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
    """Lance une sous-commande en propageant stdin et stdout, ce qui est
    necessaire pour les fenetres OpenCV interactives de calibrate_extrinsic.py."""
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"  Echec de {label} (code de retour {rc}).")
    return rc


def read_residuals(index: int) -> dict | None:
    """Lit les residus depuis handeye_cam_<index>.json apres la resolution."""
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
        print(f"  [WARN] Lecture de {path.name} impossible : {e}")
        return None


def verdict_for_cam(role: str, mean_mm: float, max_mm: float) -> tuple[str, bool]:
    """Renvoie le verdict et un booleen is_ok qui indique si la calibration
    permet de desactiver bias_correction."""
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
    """Capture et resolution pour une camera. Renvoie un dictionnaire contenant
    les residus ou un statut d'echec."""
    role = ROLE_FOR_INDEX.get(index, "unknown")
    banner(f"Camera cam_{index} ({role})", char="#")

    if role == "eye_in_hand":
        print("  Procedure eye-in-hand (cam_2 montee sur la pince) :")
        print("    1. Poser le damier fixe sur la table (bien plat, bien eclaire,")
        print("       au moins 30cm de la base du robot pour eviter les zones d'exclusion).")
        print("    2. Le script lance la capture. Teleoperer le bras")
        print("       (avec le leader s'il est branche, sinon manuellement) pour amener")
        print("       cam_2 a 15-25 positions au-dessus du damier.")
        print("    3. Diversite angulaire >65deg : inclinaisons +/-30deg, rotations,")
        print("       distances variees entre 15 et 40cm de cam_2 au damier.")
        print("    4. Touche 'c' pour capturer chaque pose, 'q' pour terminer.")
    else:
        print("  Procedure eye-to-hand (cam_0 ou cam_1 fixes sur la barriere) :")
        print("    1. Coller le damier sur la pince fermee (bien centre, plat).")
        print("    2. La camera est fixe. Teleoperer (ou deplacer a la main)")
        print("       le bras pour amener le damier devant la camera.")
        print("    3. Diversite angulaire >65deg : inclinaisons +/-30deg, rotations,")
        print("       distances variees entre 30cm et 80cm de la camera au damier.")
        print("    4. Viser 50-70 poses (le solveur rejette les valeurs aberrantes,")
        print("       il en reste typiquement 25-30 utiles).")
        print("    5. Touche 'c' pour capturer chaque pose, 'q' pour terminer.")
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

    # Resolution
    print()
    print(f"  Resolution hand-eye cam_{index}...")
    cmd_solve = [
        sys.executable, str(REPO / "scripts" / "solve_handeye_cam.py"),
        "--index", str(index),
    ]
    if run_step(cmd_solve, f"solve cam_{index}") != 0:
        return {"solve_failed": True}

    # Lecture des residus
    res = read_residuals(index)
    if res is None or res.get("mean_trans_mm") is None:
        print(f"  [WARN] Residus illisibles pour cam_{index}.")
        return {"unreadable": True}

    verdict, _ = verdict_for_cam(role, res["mean_trans_mm"], res["max_trans_mm"])
    print()
    print(f"  --- Resultat cam_{index} ---")
    print(f"    moyenne translation : {res['mean_trans_mm']:.2f} mm")
    print(f"    max translation     : {res['max_trans_mm']:.2f} mm")
    print(f"    moyenne rotation    : {res['mean_rot_deg']:.2f} deg")
    print(f"    max rotation        : {res['max_rot_deg']:.2f} deg")
    print(f"    poses utiles        : {res['n_poses_used']}/{res['n_poses_total']}")
    print(f"    verdict             : {verdict}")
    res["role"] = role
    res["verdict"] = verdict
    return res


def disable_bias_correction(stamp: str):
    """Met dx, dy et dz a 0 dans bias_correction.json apres sauvegarde."""
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
    print(f"  [OK] {path.name} mis a dx=dy=dz=0 (sauvegarde conservee).")
    print(f"       Verification : python scripts/check_perception.py --gt configs/perception/gt_test.json")


def print_summary(all_residuals: dict, stamp: str):
    banner("Resume global")
    print(f"  Sauvegardes conservees avec le suffixe : before_B3_{stamp}")
    print()
    print(f"  {'cam':<6} {'role':<14} {'moyenne':>9} {'max':>9} {'verdict'}")
    print(f"  {'-'*6} {'-'*14} {'-'*9} {'-'*9} {'-'*30}")
    all_ok = True
    for i, res in all_residuals.items():
        if "mean_trans_mm" not in res:
            print(f"  cam_{i}  {ROLE_FOR_INDEX[i]:<14} (echec : {list(res.keys())[0]})")
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
        description="Recalibration hand-eye complete des trois cameras.",
    )
    p.add_argument("--cols", type=int, default=9,
                   help="Nombre de colonnes du damier (defaut : 9, asymetrique avec rows=6).")
    p.add_argument("--rows", type=int, default=6,
                   help="Nombre de lignes du damier (defaut : 6).")
    p.add_argument("--square-size", type=float, default=22.0,
                   help="Taille d'une case du damier en mm (defaut : 22.0).")
    p.add_argument("--cams", nargs="+", type=int, default=[0, 1, 2],
                   help="Indices des cameras a recalibrer (defaut : 0 1 2).")
    p.add_argument("--skip-validation", action="store_true",
                   help="Saute les etapes check_calibration, check_perception et bias_correction (defaut : desactive).")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    banner("Recalibration hand-eye complete")
    print(f"  damier             : {args.cols} x {args.rows} cases @ {args.square_size}mm")
    print(f"  cameras            : cam_{', cam_'.join(map(str, args.cams))}")
    print(f"  suffixe sauvegarde : before_B3_{stamp}")
    print()
    print("  Motivation de la procedure :")
    print("    Le pipeline actuel presente un biais Y residuel d'environ +40mm au")
    print("    refinement, signe d'une calibration hand-eye degradee. bias_correction.json")
    print("    compense la moyenne (-30mm) mais le residu varie selon la position.")
    print("    Cette recalibration vise a ramener les residus au plancher SO-101 :")
    print("    - cam_0, cam_1 (eye-to-hand) : moyenne <= 5mm, max <= 12mm")
    print("    - cam_2        (eye-in-hand) : moyenne <= 3mm, max <= 6mm")
    print()
    print("  Duree estimee : 1h30 a 2h de manipulation (capture et resolution par camera).")
    print()
    if not confirm("  Pret a demarrer ?"):
        print("Annule.")
        return

    # Sauvegardes initiales
    banner("Phase 1 : sauvegarde des calibrations actuelles", char="-")
    for i in args.cams:
        backup_file(REPO / "configs" / f"handeye_cam_{i}.json",
                    f"before_B3_{stamp}")
        backup_file(REPO / "configs" / f"extrinsic_capture_cam_{i}.json",
                    f"before_B3_{stamp}")

    # Capture et resolution par camera
    all_residuals: dict[int, dict] = {}
    for i in args.cams:
        res = calibrate_one_cam(i, args)
        all_residuals[i] = res
        if "aborted" in res:
            print(f"\n  Procedure abandonnee a cam_{i}. Les autres cameras n'ont pas ete traitees.")
            print("   Les sauvegardes sont conservees. Reprise possible avec :")
            remaining = [c for c in args.cams if c >= i]
            print(f"   python scripts/recalibrate_handeye.py --cams {' '.join(map(str, remaining))}")
            return

    # Resume global
    all_ok = print_summary(all_residuals, stamp)

    # Validation et bias_correction
    if not args.skip_validation:
        banner("Phase 2 : validation globale")
        print("  Lancement de check_calibration.py...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "check_calibration.py")],
                       cwd=REPO)
        print()

        gt_path = REPO / "configs" / "perception" / "gt_test.json"
        if gt_path.exists():
            print("  Validation 3D contre la reference (gt_test.json)...")
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "check_perception.py"),
                 "--gt", str(gt_path)],
                cwd=REPO,
            )
            print()
        else:
            print(f"  [INFO] {gt_path.name} absent, validation 3D sautee.")
            print("         Poser le cube a une position mesuree au pied a coulisse,")
            print("         creer ce fichier (cf. gt_test_example.json), puis relancer :")
            print(f"         python scripts/check_perception.py --gt {gt_path}")
            print()

        banner("Phase 3 : bias_correction.json")
        if all_ok:
            print("  Tous les residus respectent les criteres OK.")
            print("  La compensation bias_correction.json peut etre desactivee (dx=dy=dz=0).")
            print("  Le decalage est desormais porte par la calibration elle-meme ;")
            print("  la compensation systematique n'est plus necessaire.")
            print()
            if ask_yes_no("  Desactiver bias_correction.json maintenant ?", default_no=False):
                disable_bias_correction(stamp)
            else:
                print("  bias_correction.json laisse en l'etat.")
                print("  Avec des residus OK et la compensation toujours active, le pipeline")
                print("  surcorrigera. Penser a la desactiver manuellement :")
                print("    python -c \"import json; p='configs/perception/bias_correction.json'; "
                      "d=json.load(open(p)); d['dy_mm']=0; json.dump(d, open(p,'w'), indent=2)\"")
        else:
            print("  Au moins une camera est en dessous des criteres OK.")
            print("  bias_correction.json reste actif comme filet de securite.")
            print()
            print("  Cameras a recalibrer :")
            for i, res in all_residuals.items():
                if res.get("mean_trans_mm"):
                    role = res["role"]
                    _, is_ok = verdict_for_cam(role, res["mean_trans_mm"], res["max_trans_mm"])
                    if not is_ok:
                        print(f"    python scripts/recalibrate_handeye.py --cams {i}")
            print()
            print("  Causes possibles d'un residu trop eleve :")
            print("    - diversite angulaire insuffisante (viser >65deg d'ecart moyen)")
            print("    - damier deforme ou impression de mauvaise qualite")
            print("    - mauvais eclairage (reflets sur le damier)")
            print("    - structure de la barriere deplacee (camera bougee mecaniquement)")

    banner("Termine")
    print(f"  Sauvegardes conservees : configs/*.before_B3_{stamp}.backup.json")
    print(f"  Pour restaurer toutes les cameras :")
    print(f"    for i in 0 1 2; do")
    print(f"      cp configs/handeye_cam_$i.before_B3_{stamp}.backup.json configs/handeye_cam_$i.json")
    print(f"    done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrompu par l'utilisateur.")
        sys.exit(130)
