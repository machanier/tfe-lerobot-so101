#!/usr/bin/env python3
"""
experiment_campaign.py - Campagne experimentale pour le memoire TFE (P4).

OBJECTIF : produire un dataset de mesures reproductibles pour la section
"Resultats" du chapitre 6 du memoire. Pour chaque position d'objet, on lance
N essais et on collecte :
  - succes  (True/False)            : le robot a-t-il saisi l'objet ?
  - n_attempts (int)                : combien de tentatives (1 = succes direct, 2 = avec retry)
  - duration_s (float)              : duree totale du pick-and-place
  - attempts_log (list)             : pince reelle, marge, succes par tentative

Sortie :
  - outputs/experiments/campaign_YYYYMMDD_HHMMSS.json  (toutes les donnees)
  - outputs/experiments/campaign_YYYYMMDD_HHMMSS.csv   (vue tabulaire)
  - Statistiques agregees affichees a l'ecran (taux de succes, temps moyen, ...)

PROTOCOLE :
  1. Place la boite de depose et mesure sa position (configs/scene.json).
  2. Choisis 3 positions d'objet (par exemple : centre, gauche, droite).
  3. Lance ce script avec --n-per-position 10 et --positions centre gauche droite.
  4. A chaque essai, le script te demande de replacer l'objet a la position
     indiquee (puis ENTREE).
  5. Le robot tente la saisie. Resultats logges + sauves.
  6. A la fin, recap statistiques et fichiers de sortie pour analyse memoire.

Usage typique :
  python scripts/experiment_campaign.py \\
      --target orange_cube --detector hf \\
      --n-per-position 10 \\
      --positions centre gauche droite

Note : a executer avec le robot allume et tous les peripheriques (cameras +
hub USB) en place. Verifie que `python scripts/check_calibration.py` passe
au prealable.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import FOLLOWER_PORT  # noqa: E402
from src.pipeline import PickAndPlacePipeline, PipelineConfig  # noqa: E402


def make_pipeline(args) -> PickAndPlacePipeline:
    """Cree un nouveau pipeline pour un essai (etat reset)."""
    cfg = PipelineConfig(
        target_label=args.target,
        detector_kind=args.detector,
        motor_port=args.port,
        max_velocity_rad_s=args.max_velocity,
        grip_close_pct=args.grip_close,
        dry_run=False,                      # campagne reelle
        closed_loop=True,                   # toujours actif pour comparaison juste
        display=args.display,
        max_grasp_retries=args.max_retries,
    )
    # Seuil de detection saisie : par defaut on herite du defaut CALIBRE de
    # PipelineConfig (8.0 pour le cube 30mm). On n'override que si
    # --grasp-threshold est explicitement passe (analyse de sensibilite).
    # Evite le faux negatif de l'ancien defaut 15.0 (cf calibrate_grasp_threshold.py :
    # une saisie reussie a 14% etait classee RATEE car 14-5=9% < 15%).
    if args.grasp_threshold is not None:
        cfg.grasp_success_threshold_pct = args.grasp_threshold
    if args.grasp_lateral_offset is not None:
        cfg.grasp_lateral_offset_mm = args.grasp_lateral_offset
    return PickAndPlacePipeline(cfg)


def run_trial(args, position_name: str, trial_idx: int) -> dict:
    """Execute UN essai et renvoie un dict avec les metriques."""
    print()
    print(">" * 70)
    print(f">>> ESSAI {trial_idx} a la position '{position_name}'")
    print(">" * 70)
    if not args.no_prompt:
        try:
            input(f"    Place l'objet a la position '{position_name}' "
                  f"puis ENTREE (Ctrl+C pour stop campagne) : ")
        except (EOFError, KeyboardInterrupt):
            raise

    t0 = time.time()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "position": position_name,
        "trial": trial_idx,
        "success": None,
        "n_attempts": None,
        "attempts_log": [],
        "duration_s": None,
        "grasp_pose_error_mm": None,   # residu IK prise retenue (erreur de pose atteinte)
        "cam2_correction_mm": None,    # amplitude correction cam_2 (precision perception)
        "error": None,
    }
    try:
        pipeline = make_pipeline(args)
        pipeline.run()
        record["success"] = bool(getattr(pipeline, "_grasp_final_success", False))
        record["n_attempts"] = int(getattr(pipeline, "_grasp_total_attempts", 0))
        record["attempts_log"] = list(getattr(pipeline, "_grasp_attempts_log", []))
        # Metriques de precision (Y-free, internes) : None si jamais atteintes
        # (ex. echec perception avant planif / pas de correction cam_2).
        record["grasp_pose_error_mm"] = getattr(pipeline, "_grasp_pose_error_mm", None)
        record["cam2_correction_mm"] = getattr(pipeline, "_cam2_correction_mm", None)
    except KeyboardInterrupt:
        record["error"] = "interrupted"
        record["success"] = False
        raise   # remonte au caller pour arret propre
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["success"] = False
        print(f"    !! Exception : {record['error']}")
        traceback.print_exc()
    finally:
        record["duration_s"] = round(time.time() - t0, 2)

    tag = "OK" if record["success"] else "ECHEC"
    n = record["n_attempts"] or 0
    print(f"    >>> {position_name} essai {trial_idx} : {tag} "
          f"en {n} tentative(s), {record['duration_s']:.1f}s")
    return record


def print_aggregate_stats(results: list[dict]):
    """Affiche les statistiques agregees (a recopier dans le memoire)."""
    print()
    print("=" * 70)
    print(" STATISTIQUES AGREGEES (pour memoire chapitre 6)")
    print("=" * 70)
    if not results:
        print("  Aucun resultat (essais interrompus avant tout run).")
        return

    # Par position
    by_pos: dict[str, list[dict]] = {}
    for r in results:
        by_pos.setdefault(r.get("position", "?"), []).append(r)

    for pos, trials in by_pos.items():
        n = len(trials)
        ok = sum(1 for t in trials if t.get("success"))
        rate = 100.0 * ok / n if n else 0.0
        # Tentatives moyennes parmi les succes (1.0 si tous direct, ~1.5 si retry moitié)
        succ_attempts = [t.get("n_attempts") or 1 for t in trials if t.get("success")]
        avg_attempts = (sum(succ_attempts) / len(succ_attempts)) if succ_attempts else float("nan")
        durations = [t.get("duration_s") or 0 for t in trials if t.get("duration_s")]
        avg_dur = (sum(durations) / len(durations)) if durations else 0.0
        # Retries utilises (n_attempts > 1)
        n_with_retry = sum(1 for t in trials if (t.get("n_attempts") or 0) > 1)
        print(f"  {pos:<15} : {ok:>2}/{n:>2} OK ({rate:>5.1f}%)"
              f"  tentatives moyennes sur succes = {avg_attempts:.2f}"
              f"  temps moyen = {avg_dur:.1f}s"
              f"  retries utilises = {n_with_retry}/{n}")

    # Global
    n_total = len(results)
    n_ok = sum(1 for r in results if r.get("success"))
    rate_g = 100.0 * n_ok / n_total if n_total else 0.0
    durations = [r.get("duration_s") or 0 for r in results if r.get("duration_s")]
    avg_dur_g = (sum(durations) / len(durations)) if durations else 0.0
    n_retry_used = sum(1 for r in results if (r.get("n_attempts") or 0) > 1)
    n_retry_saved = sum(1 for r in results
                        if r.get("success") and (r.get("n_attempts") or 0) > 1)
    print()
    print(f"  GLOBAL   : {n_ok:>2}/{n_total:>2} OK ({rate_g:.1f}%)  "
          f"temps moyen = {avg_dur_g:.1f}s")
    print(f"  RETRY    : {n_retry_used} essais avec retry,"
          f" dont {n_retry_saved} rattrapes grace au retry")
    print("=" * 70)


def save_results(results: list[dict], args, out_dir: Path) -> tuple[Path, Path]:
    """Sauve resultats en JSON (complet) + CSV (vue tabulaire pour Excel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"campaign_{stamp}.json"
    csv_path = out_dir / f"campaign_{stamp}.csv"

    with open(json_path, "w") as f:
        json.dump({
            "args": vars(args),
            "n_results": len(results),
            "results": results,
        }, f, indent=2, default=str)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        target = getattr(args, "target", "")
        detector = getattr(args, "detector", "")
        writer.writerow(["objet", "detecteur",
                         "timestamp", "position", "trial", "success",
                         "n_attempts", "duration_s",
                         "grasp_pose_error_mm", "cam2_correction_mm",
                         "last_gripper_pct", "last_margin_pct",
                         "error",
                         # colonnes MANUELLES (a remplir a la main dans Excel) :
                         "orientation", "depose_boite", "mode_echec", "notes"])
        for r in results:
            last = r["attempts_log"][-1] if r.get("attempts_log") else {}
            gpe = r.get("grasp_pose_error_mm")
            c2c = r.get("cam2_correction_mm")
            writer.writerow([
                target,
                detector,
                r.get("timestamp"),
                r.get("position"),
                r.get("trial"),
                int(bool(r.get("success"))),
                r.get("n_attempts"),
                f"{r.get('duration_s') or 0:.2f}",
                f"{gpe:.2f}" if gpe is not None else "",
                f"{c2c:.2f}" if c2c is not None else "",
                f"{last.get('gripper_pct', ''):.1f}" if last.get("gripper_pct") is not None else "",
                f"{last.get('marge_pct', ''):+.1f}" if last.get("marge_pct") is not None else "",
                r.get("error") or "",
                "", "", "", "",   # orientation, depose_boite, mode_echec, notes (manuel)
            ])
    return json_path, csv_path


def main():
    p = argparse.ArgumentParser(
        description="Campagne experimentale pick-and-place (TFE chapitre 6).",
    )
    p.add_argument("--target", default="orange_cube",
                   help="Label objet a saisir (defaut: orange_cube).")
    p.add_argument("--detector", default="hf", choices=["hsv", "hf"],
                   help="Detecteur (defaut: hf = OWL-ViTv2).")
    p.add_argument("--n-per-position", type=int, default=10,
                   help="Nombre d'essais par position (defaut: 10).")
    p.add_argument("--positions", nargs="+",
                   default=["centre", "gauche", "droite"],
                   help="Noms logiques des positions (defaut: centre gauche droite).")
    p.add_argument("--port", default=FOLLOWER_PORT,
                   help="Port USB du follower.")
    p.add_argument("--max-velocity", type=float, default=0.5,
                   help="Vitesse articulaire max rad/s (defaut: 0.5).")
    p.add_argument("--grip-close", type=float, default=5.0,
                   help="Consigne fermeture pince 0-100 (defaut: 5 = presque ferme).")
    p.add_argument("--max-retries", type=int, default=1,
                   help="Nb max de retry par essai (defaut: 1 = 2 tentatives total).")
    p.add_argument("--grasp-threshold", type=float, default=None,
                   help="Seuil de detection saisie (%% au-dessus de grip-close). "
                        "Defaut = herite du defaut CALIBRE de PipelineConfig "
                        "(8.0 pour le cube 30mm). Ne le force que pour une analyse "
                        "de sensibilite (l'ancien 15.0 causait des faux negatifs).")
    p.add_argument("--grasp-lateral-offset", type=float, default=None,
                   help="Decalage lateral saisie en mm (pince asymetrique). Defaut=8 "
                        "(cube 30mm). Regle par objet : rectangle/cylindre plus large "
                        "peut demander une autre valeur (cf reglage prealable en single-shot).")
    p.add_argument("--output-dir", default="outputs/experiments",
                   help="Repertoire de sortie (defaut: outputs/experiments).")
    p.add_argument("--no-prompt", action="store_true",
                   help="N'attend pas l'ENTREE entre essais (mode automatique, "
                        "a eviter sauf si tu as un convoyeur).")
    p.add_argument("--display", action="store_true",
                   help="Affiche les cameras pendant chaque essai (ralentit).")
    args = p.parse_args()

    print("=" * 70)
    print(" CAMPAGNE EXPERIMENTALE TFE - pick-and-place")
    print("=" * 70)
    print(f"  cible            : {args.target}")
    print(f"  detecteur        : {args.detector}")
    print(f"  positions        : {args.positions}")
    print(f"  essais/position  : {args.n_per_position}")
    print(f"  total essais     : {args.n_per_position * len(args.positions)}")
    print(f"  max retries      : {args.max_retries}")
    seuil_txt = (f"{args.grasp_threshold}%" if args.grasp_threshold is not None
                 else "8% (defaut calibre PipelineConfig)")
    print(f"  seuil saisie OK  : pince > consigne + {seuil_txt}")
    print("=" * 70)
    print()
    if not args.no_prompt:
        try:
            input("Pret a commencer la campagne ? ENTREE = go, Ctrl+C = stop : ")
        except (EOFError, KeyboardInterrupt):
            print("\nAnnule.")
            return

    out_dir = REPO / args.output_dir
    results: list[dict] = []
    try:
        for pos_name in args.positions:
            print()
            print("#" * 70)
            print(f"#  POSITION : {pos_name}  ({args.n_per_position} essais)")
            print("#" * 70)
            for trial in range(1, args.n_per_position + 1):
                record = run_trial(args, pos_name, trial)
                results.append(record)
    except KeyboardInterrupt:
        print("\n!! Campagne interrompue par utilisateur (Ctrl+C).")
    except Exception as e:
        print(f"\n!! Erreur fatale : {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # TOUJOURS sauver ce qu'on a, meme si interrompu en cours de route
        if results:
            json_path, csv_path = save_results(results, args, out_dir)
            print()
            print(f">> {len(results)} essais enregistres :")
            print(f"   {json_path}")
            print(f"   {csv_path}")
            print_aggregate_stats(results)
        else:
            print("Aucun resultat a sauvegarder.")


if __name__ == "__main__":
    main()
