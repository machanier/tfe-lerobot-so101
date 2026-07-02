#!/usr/bin/env python3
"""
experiment_campaign_il.py - Campagne de tests pour la methode IMITATION LEARNING (ACT).

Pendant de scripts/experiment_campaign.py (pipeline classique), adapte a ACT :
  * 1 essai = 1 episode lerobot-record pilote par la policy (pas de teleop, pas de leader).
  * Metriques surtout manuelles : ACT n'expose ni residu IK, ni correction cam_2, ni
    nombre de replans -> le resultat est note apres chaque essai.
  * Automatiques (parse depuis la sortie de lerobot-record + horloge) : duree, frequence
    de boucle moyenne, nombre de decrochages cam_2, "cam_2 gelee ?", nombre de frames.
  * Sauve CSV + JSON, calcule le taux de reussite + intervalle de confiance Wilson 95 %
    (calcul pur, sans scipy).
  * Memes positions proche/mi/loin (rayon ~20/30/40 cm) que la campagne classique,
    pour remplir la colonne imitation du tableau de comparaison.

Chaque essai sauve sa propre video (front + poignet) sous outputs/experiments/<campagne>/,
comme trace reproductible de chaque essai.

Usage :
    # test principal in-distribution (cube, 3 distances x 10 = 30 essais) :
    python scripts/experiment_campaign_il.py --positions proche mi loin --n-per-position 10

    # sonde de generalisation (positions hors-distribution OU autre objet, ~3 essais) :
    python scripts/experiment_campaign_il.py --probe --object balle_bleue --positions mi --n-per-position 3

Pilotage : ACT ne s'arrete jamais seul -> appuie sur FLECHE DROITE quand le bras est
revenu a home (essai fini). ECHAP coupe l'episode. Ctrl+C entre 2 essais stoppe la campagne.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    FOLLOWER_ID,
    FOLLOWER_PORT,
    HF_USER,
    IL_TASK,
    il_cameras_flag,
)

# Stades de la tache, du plus court au plus complet (style ALOHA/ACT).
STADES = ["aucun", "reach", "contact", "grasp", "lift", "transport", "place"]

# Codes d'echec specifiques IL (les E1-E6 de la campagne classique = perception/IK).
MODES_ECHEC = {
    "I1": "n'approche pas / reste a la base",
    "I2": "approche mais mal aligne (cube hors machoires)",
    "I3": "ferme a vide",
    "I4": "saisit puis lache en transit",
    "I5": "rate la depose (hors boite)",
    "I6": "camera gelee / materiel (essai biaise)",
    "I7": "comportement erratique / oscillation",
    "I8": "saisit mais l'objet s'echappe a la fermeture (ejecte par la force de serrage, avant le transit)",
}

COLUMNS = [
    "policy", "probe", "timestamp", "objet", "position", "trial",
    "success", "stade_atteint", "score_partiel", "n_reapproches",
    "duration_s", "n_frames", "loop_hz_moyen", "cam2_fail_count", "cam2_frozen",
    "error", "mode_echec", "notes",
]


def score_partiel(stade):
    """Score partiel facon SmolVLA (meme robot SO-101) : 0.5 a la prise, +0.5 a la depose."""
    if stade == "place":
        return 1.0
    if stade in ("grasp", "lift", "transport"):
        return 0.5
    return 0.0


def wilson_ci(k, n, z=1.96):
    """IC Wilson 95 % pour une proportion k/n (math pur, pas de scipy)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_cmd(args, root):
    """Commande lerobot-record pour 1 episode pilote par la policy (calquee sur eval_policy.py)."""
    return [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={args.port}",
        f"--robot.id={FOLLOWER_ID}",
        il_cameras_flag(),
        f"--dataset.repo_id={HF_USER}/eval_campaign_il",  # lerobot EXIGE le prefixe eval_ avec une policy
        f"--dataset.root={root}",
        f"--dataset.single_task={IL_TASK}",
        "--dataset.num_episodes=1",
        f"--dataset.episode_time_s={args.episode_time}",
        "--dataset.push_to_hub=false",
        f"--display_data={'true' if args.display else 'false'}",
        f"--policy.path={args.policy_path}",
    ]


_RE_HZ = re.compile(r"running slower \(([\d.]+) Hz\)")
_RE_CAM2_FAIL = re.compile(r"Error reading frame in background thread for OpenCVCamera\(2\)")
_RE_CAM2_DEAD = re.compile(r"exceeded maximum consecutive read failures")
_RE_FRAMES = re.compile(r"Map:\s*\d+%.*?\|\s*(\d+)/(\d+)")


def run_episode(cmd):
    """Lance lerobot-record, relaie la sortie en direct, et parse les metriques auto."""
    hz_vals, cam2_fail, cam2_dead, n_frames = [], 0, False, None
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end="")  # relais direct : tu vois le robot tourner
        m = _RE_HZ.search(line)
        if m:
            hz_vals.append(float(m.group(1)))
        if _RE_CAM2_FAIL.search(line):
            cam2_fail += 1
        if _RE_CAM2_DEAD.search(line):
            cam2_dead = True
        m = _RE_FRAMES.search(line)
        if m:
            n_frames = int(m.group(1))
    proc.wait()
    # Hz moyen hors demarrage (on jette les mesures < 5 Hz = warmup de la boucle).
    stable = [h for h in hz_vals if h >= 5.0]
    loop_hz = round(sum(stable) / len(stable), 1) if stable else None
    return {
        "loop_hz_moyen": loop_hz,
        "cam2_fail_count": cam2_fail,
        "cam2_frozen": int(cam2_dead),
        "n_frames": n_frames,
        "returncode": proc.returncode,
    }


def _flush_stdin():
    """Vide le buffer clavier (touches tapees pendant l'episode/encodage) avant les questions,
    pour que les reponses ne se melangent pas avec ce qui a ete tape trop tot."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def ask(prompt, default=""):
    try:
        r = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return r if r else default


def _to_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def ask_manual(record):
    """Metriques notees a la main apres l'essai. La reussite est deduite du stade
    (place = depose dans la boite = reussite) : une seule saisie, pas de contradiction."""
    _flush_stdin()  # jette les touches tapees pendant l'encodage : reponses comptees a partir d'ici
    print("\n--- Observations (episode termine ; ENTREE = defaut) ---")
    print("   stades, du moins au plus loin :  " + " > ".join(STADES[1:]))
    print("   (place = deposee dans la boite = reussite)")
    stade = ask("  Stade atteint le plus loin : ", "aucun").lower()
    if stade not in STADES:
        stade = "aucun"
    record["stade_atteint"] = stade
    record["success"] = 1 if stade == "place" else 0      # reussite DEDUITE du stade
    record["score_partiel"] = score_partiel(stade)
    record["n_reapproches"] = _to_int(ask("  Nb de re-approches dans l'episode [0] : ", "0"))
    if not record["success"]:
        print("   codes :", ", ".join(f"{k}={v}" for k, v in MODES_ECHEC.items()))
        record["mode_echec"] = ask("  Mode d'echec [I1-I8] : ", "")
    record["notes"] = ask("  Notes libres : ", "")
    print(f"   -> note : stade={stade}  reussite={'OUI' if record['success'] else 'non'}")


def run_trial(args, campaign_dir, position, trial, idx, total):
    print(f"\n{'=' * 60}\n>>> ESSAI {idx}/{total} - position '{position}' (essai {trial})\n{'=' * 60}")
    if not args.no_prompt:
        ask(
            f"  Place le cube a '{position}', remets le BRAS A HOME, puis ENTREE "
            "(Ctrl+C = stop campagne) : "
        )
    rec = {c: None for c in COLUMNS}
    rec.update(
        policy=Path(args.policy_path).name,
        probe=int(args.probe),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        objet=args.object,
        position=position,
        trial=trial,
        error="",
    )
    root = campaign_dir / f"{position}_t{trial:02d}"
    t0 = time.time()
    ran_ok = True
    try:
        auto = run_episode(build_cmd(args, root))
        rc = auto.pop("returncode", 0)
        rec.update(auto)
        if rc != 0:
            rec["error"] = (str(rec["error"]) + f" returncode={rc}").strip()
            ran_ok = False
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
        ran_ok = False
        print(f"  [erreur] {e}")
    finally:
        rec["duration_s"] = round(time.time() - t0, 1)
    if ran_ok and not args.no_prompt:
        ask_manual(rec)
    status = "OK" if rec.get("success") else ("ECHEC" if not ran_ok else "—")
    print(
        f">>> '{position}' essai {trial} : {status} | "
        f"{rec['duration_s']}s | {rec.get('loop_hz_moyen')} Hz | "
        f"cam2_fail={rec.get('cam2_fail_count')} "
        f"{'(GELEE)' if rec.get('cam2_frozen') else ''}"
    )
    return rec, ran_ok


def save_results(records, args, campaign_dir, stamp):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"campaign_il_{stamp}"
    with open(base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in records:
            w.writerow([r.get(c, "") for c in COLUMNS])
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(
            {"args": vars(args), "n": len(records), "records": records,
             "videos_dir": str(campaign_dir)},
            f, indent=2, default=str,
        )
    print(f"\nResultats -> {base}.csv / .json   (videos par essai : {campaign_dir})")


def aggregate(records):
    print(f"\n{'=' * 60}\n  SYNTHESE ({len(records)} essais)\n{'=' * 60}")
    by_pos = {}
    for r in records:
        by_pos.setdefault(r["position"], []).append(r)
    for pos, rs in by_pos.items():
        n = len(rs)
        k = sum(int(bool(r.get("success"))) for r in rs)
        lo, hi = wilson_ci(k, n)
        score = sum((r.get("score_partiel") or 0) for r in rs) / n
        frozen = sum(int(bool(r.get("cam2_frozen"))) for r in rs)
        print(
            f"  [{pos:>10}] {k}/{n} = {100 * k / n:4.0f}%  "
            f"IC95 Wilson [{100 * lo:.0f}-{100 * hi:.0f}%]  "
            f"score partiel moy {score:.2f}  cam2_gelee {frozen}/{n}"
        )
    n = len(records)
    k = sum(int(bool(r.get("success"))) for r in records)
    lo, hi = wilson_ci(k, n)
    print(f"  {'-' * 54}")
    print(f"  GLOBAL : {k}/{n} = {100 * k / n:.0f}%  IC95 Wilson [{100 * lo:.0f}-{100 * hi:.0f}%]")
    # Repartition par stade (combien d'essais ont atteint AU MOINS chaque stade).
    order = {s: i for i, s in enumerate(STADES)}
    print("  Etage atteint (cumulatif) :")
    for s in STADES[1:]:
        c = sum(1 for r in records if order.get(r.get("stade_atteint"), 0) >= order[s])
        print(f"     {s:>10} : {c}/{n}")


def main():
    p = argparse.ArgumentParser(
        description="Campagne de tests IL/ACT (analogue de experiment_campaign.py)"
    )
    p.add_argument("--policy-path", default="outputs/colab_model_lowres")
    p.add_argument("--positions", nargs="+", default=["proche", "mi", "loin"])
    p.add_argument("--n-per-position", type=int, default=10)
    p.add_argument("--object", default="orange_cube")
    p.add_argument("--episode-time", type=int, default=60)
    p.add_argument("--probe", action="store_true",
                   help="marque les essais comme sonde de generalisation (OOD / autre objet)")
    p.add_argument("--port", default=FOLLOWER_PORT)
    p.add_argument("--output-dir", default="outputs/experiments")
    p.add_argument("--no-prompt", action="store_true", help="enchaine sans questions (mode auto)")
    p.add_argument("--display", action="store_true", help="affiche les cameras (charge ++ sur le hub)")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path(args.output_dir) / f"campaign_il_{stamp}"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.positions) * args.n_per_position
    print(
        f"Campagne IL : policy={args.policy_path}  objet={args.object}  "
        f"{'[SONDE GENERALISATION] ' if args.probe else ''}"
        f"{len(args.positions)} positions x {args.n_per_position} = {total} essais"
    )
    print("Rappel : FLECHE DROITE quand le bras est revenu a home (essai fini).\n")

    records = []
    try:
        idx = 0
        for pos in args.positions:
            trial = 0
            while trial < args.n_per_position:
                trial += 1
                idx += 1
                rec, ran_ok = run_trial(args, campaign_dir, pos, trial, idx, total)
                if not ran_ok:
                    # Echec au branchement (hoquet moteur/cam) ou abandon -> NON enregistre.
                    try:
                        ans = input("  Episode en echec. Reessayer CET essai ? "
                                    "[O/n] (Ctrl+C = stop campagne) : ").strip().lower()
                    except EOFError:
                        ans = "n"
                    if ans in ("", "o", "oui", "y"):
                        trial -= 1   # on refait le meme numero d'essai
                        idx -= 1
                        continue
                    print("  -> essai saute (non enregistre).")
                    continue
                records.append(rec)
    except KeyboardInterrupt:
        print("\n[Campagne interrompue - sauvegarde de ce qui a ete collecte]")
    finally:
        if records:
            save_results(records, args, campaign_dir, stamp)
            aggregate(records)
        else:
            print("\n[Aucun essai note - rien a sauvegarder]")


if __name__ == "__main__":
    main()
