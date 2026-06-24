#!/usr/bin/env python3
"""
dataset_info.py – Afficher l'etat d'un dataset LeRobot (sans lancer d'enregistrement)

Usage:
    python scripts/dataset_info.py                          # repo par defaut (config.IL_REPO_ID)
    python scripts/dataset_info.py --repo-id maxence/so101_test

Pratique pour savoir combien d'episodes sont deja enregistres / sauves.
"""

import argparse
import json
import os
import pathlib
import sys

from config import IL_REPO_ID


def main():
    parser = argparse.ArgumentParser(description="Etat d'un dataset LeRobot local")
    parser.add_argument("--repo-id", type=str, default=IL_REPO_ID,
                        help=f"Repo id du dataset (defaut: {IL_REPO_ID})")
    args = parser.parse_args()

    # HF_LEROBOT_HOME, sinon HF_HOME/lerobot, sinon ~/.cache/huggingface/lerobot
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        root = pathlib.Path(base)
    else:
        hf_home = os.environ.get("HF_HOME") or (pathlib.Path.home() / ".cache" / "huggingface")
        root = pathlib.Path(hf_home) / "lerobot"

    ds = root / args.repo_id
    info_path = ds / "meta" / "info.json"

    if not info_path.exists():
        print(f"Aucun dataset trouve pour '{args.repo_id}'")
        print(f"  cherche dans : {ds}")
        print("  (rien encore enregistre, ou repo-id different)")
        sys.exit(0)

    info = json.loads(info_path.read_text())
    cams = [k for k in info.get("features", {}) if k.startswith("observation.images")]
    print(f"Dataset  : {ds}")
    print(f"  episodes : {info.get('total_episodes')}")
    print(f"  frames   : {info.get('total_frames')}")
    print(f"  fps      : {info.get('fps')}")
    print(f"  cameras  : {cams}")


if __name__ == "__main__":
    main()
