#!/usr/bin/env python3
"""
export_episode_frames.py – Exporter des images d'un episode LeRobot en PNG

Utile pour choisir une frame a mettre dans le memoire (ex. prise TPU).

Usage:
    python scripts/export_episode_frames.py --episode 16
    python scripts/export_episode_frames.py --episode 16 --camera wrist --num 20
    python scripts/export_episode_frames.py --episode 7 --repo-id maxence/so101_test

Les PNG sont ecrits dans outputs/il_frames/ep<NN>/ (front + wrist par defaut),
echantillonnes regulierement sur toute la duree de l'episode.
"""

import argparse
import glob
import pathlib

import cv2
import pandas as pd

from config import IL_REPO_ID


def _dataset_root(repo_id):
    import os
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        root = pathlib.Path(base)
    else:
        hf_home = os.environ.get("HF_HOME") or (pathlib.Path.home() / ".cache" / "huggingface")
        root = pathlib.Path(hf_home) / "lerobot"
    return root / repo_id


def main():
    parser = argparse.ArgumentParser(description="Exporter des frames d'un episode LeRobot")
    parser.add_argument("--episode", type=int, required=True, help="Index de l'episode")
    parser.add_argument("--repo-id", type=str, default=IL_REPO_ID)
    parser.add_argument("--camera", type=str, default="both",
                        choices=["front", "wrist", "both"])
    parser.add_argument("--num", type=int, default=12,
                        help="Nombre de frames echantillonnees (defaut: 12)")
    parser.add_argument("--out", type=str, default=None,
                        help="Dossier de sortie (defaut: outputs/il_frames/ep<NN>)")
    args = parser.parse_args()

    # Bornes globales de l'episode (depuis meta/episodes)
    root = _dataset_root(args.repo_id)
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    ep_meta = pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index").sort_index()
    if args.episode not in ep_meta.index:
        raise SystemExit(f"Episode {args.episode} absent (episodes: {ep_meta.index.min()}..{ep_meta.index.max()})")
    from_idx = int(ep_meta.loc[args.episode, "dataset_from_index"])
    to_idx = int(ep_meta.loc[args.episode, "dataset_to_index"])
    length = to_idx - from_idx
    print(f"Episode {args.episode}: frames {from_idx}..{to_idx} ({length} frames, ~{length/30:.1f}s)")

    cams = ["front", "wrist"] if args.camera == "both" else [args.camera]
    keys = [f"observation.images.{c}" for c in cams]

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path("outputs/il_frames") / f"ep{args.episode:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import lourd (lerobot/torch) seulement maintenant
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id)

    n = max(1, min(args.num, length))
    step = max(1, length // n)
    idxs = list(range(from_idx, to_idx, step))[:n]

    for gi in idxs:
        item = ds[gi]
        for cam, key in zip(cams, keys):
            t = item[key]  # (3,H,W) float [0,1] RGB
            arr = (t.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            local = gi - from_idx
            path = out_dir / f"ep{args.episode:02d}_{cam}_f{local:04d}.png"
            cv2.imwrite(str(path), bgr)

    print(f"OK : {len(idxs)} frames x {len(cams)} cam -> {out_dir}/")


if __name__ == "__main__":
    main()
