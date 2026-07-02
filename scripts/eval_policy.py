#!/usr/bin/env python3
"""
eval_policy.py – Evaluer une politique entrainee sur le robot reel (LeRobot).

Usage (defaut : dernier checkpoint du job ACT sur l'objet orange) :
    python scripts/eval_policy.py

    ou en personnalisant :
    python scripts/eval_policy.py \
        --policy-path outputs/train/act_so101_orange_cube/checkpoints/last/pretrained_model \
        --episodes 10

La commande reprend celle de l'enregistrement, mais sans bras leader : les
actions sont generees par le reseau (--policy.path). N episodes d'evaluation
sont enregistres afin de mesurer le taux de succes.

Coherence du decor : les positions des cameras, l'eclairage, la calibration et
l'objet doivent etre identiques a ceux de l'enregistrement des demonstrations,
sinon la policy sort de sa distribution d'entrainement et echoue. L'identite des
cles cameras "front"/"wrist" est garantie par config.il_cameras_flag().
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from config import (
    FOLLOWER_ID,
    IL_POLICY_TYPE,
    IL_REPO_ID,
    IL_TASK,
    il_cameras_flag,
    pick_ports,
)


def main():
    dataset_slug = IL_REPO_ID.split("/")[-1]
    default_ckpt = f"outputs/train/{IL_POLICY_TYPE}_{dataset_slug}/checkpoints/last/pretrained_model"

    parser = argparse.ArgumentParser(description="Evaluer une politique LeRobot sur le robot")
    parser.add_argument("--policy-path", type=str, default=default_ckpt,
                        help=f"Chemin du checkpoint (defaut: {default_ckpt})")
    parser.add_argument("--task", type=str, default=IL_TASK,
                        help=f"Instruction (defaut: '{IL_TASK}')")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Nombre d'episodes d'evaluation (defaut: 10)")
    parser.add_argument("--repo-id", type=str, default=f"{IL_REPO_ID.split('/')[0]}/eval_{dataset_slug}",
                        help="Repo id du dataset d'eval (prefixe 'eval_')")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Envoyer le dataset d'eval sur le Hub (defaut: local)")
    parser.add_argument("--display", action="store_true",
                        help="Afficher les cameras via rerun (utile pour le debogage, mais ralentit la boucle de controle)")
    parser.add_argument("--episode-time", type=int, default=60,
                        help="Duree maximale d'un essai en secondes (defaut: 60 ; a augmenter si la boucle MPS est lente)")
    args = parser.parse_args()

    # Chaque evaluation reenregistre un dataset : on nettoie un eventuel
    # dataset d'eval existant, sinon LeRobot refuse d'ecrire par-dessus
    # (FileExistsError). Par securite, la suppression n'est effectuee que si
    # le nom commence par "eval" (jamais les demonstrations).
    hf_base = os.environ.get("HF_LEROBOT_HOME")
    if hf_base:
        ds_root = pathlib.Path(hf_base) / args.repo_id
    else:
        hf_home = os.environ.get("HF_HOME") or (pathlib.Path.home() / ".cache" / "huggingface")
        ds_root = pathlib.Path(hf_home) / "lerobot" / args.repo_id
    if ds_root.exists() and ds_root.name.startswith("eval"):
        print(f"Nettoyage du dataset d'eval existant : {ds_root}")
        shutil.rmtree(ds_root)

    follower_port, _ = pick_ports()
    if not follower_port:
        print("Port USB du follower introuvable.")
        print("  Liste : ls /dev/tty.usbmodem*   ou   lerobot-find-port")
        sys.exit(1)

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={FOLLOWER_ID}",
        il_cameras_flag(),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--display_data={'true' if args.display else 'false'}",
        f"--policy.path={args.policy_path}",
    ]

    print(f"Evaluation de la policy : {args.policy_path}")
    print(f"  Episodes: {args.episodes}   Dataset d'eval: {args.repo_id}")
    print(f"  Cameras:  {il_cameras_flag()}")
    print("  (pas de bras leader : le reseau pilote le robot)")
    print("  Clavier : -> episode suivant | Echap = arret\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nEvaluation arretee.")
    except FileNotFoundError:
        print("Commande 'lerobot-record' introuvable.")
        print("  Activer le venv au prealable : source venv/bin/activate")
        sys.exit(1)


if __name__ == "__main__":
    main()
