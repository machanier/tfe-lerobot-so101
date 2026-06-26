#!/usr/bin/env python3
"""
downscale_dataset.py – Crée une COPIE basse-résolution d'un dataset LeRobot.

Le dataset source reste intact. Le nouveau dataset (mêmes états/actions, mêmes
épisodes) a juste des images redimensionnées -> inférence plus rapide à l'éval.

Usage:
    python scripts/downscale_dataset.py --src maxence/so101_orange_cube \
        --dst Machanier/so101_orange_cube_lowres --width 320 --height 240
    # test rapide : --max-episodes 2   |   envoi Hub : --push
"""

import argparse
import copy
import glob
import pathlib

import cv2
import numpy as np

# Features ajoutées automatiquement par LeRobot (à ne PAS passer à create()).
_AUTO = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def main():
    p = argparse.ArgumentParser(description="Copie basse-resolution d'un dataset LeRobot")
    p.add_argument("--src", required=True, help="repo_id source")
    p.add_argument("--dst", required=True, help="repo_id destination")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--max-episodes", type=int, default=None, help="limiter (pour tester)")
    p.add_argument("--push", action="store_true", help="pousser sur le Hub a la fin")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(args.src)
    fps = src.meta.fps
    robot_type = src.meta.robot_type

    # Features de sortie = celles du source, MAIS shapes images redimensionnees,
    # et sans les features auto (LeRobot les rajoute).
    src_feats = src.meta.info["features"]
    img_keys = [k for k, v in src_feats.items() if v["dtype"] in ("video", "image")]
    features = {}
    for k, v in src_feats.items():
        if k in _AUTO:
            continue
        v = copy.deepcopy(v)
        if k in img_keys:
            v["shape"] = [args.height, args.width, 3]  # [H, W, C]
        features[k] = v

    dst = LeRobotDataset.create(
        repo_id=args.dst, fps=fps, features=features,
        robot_type=robot_type, use_videos=True,
    )

    # Bornes (frame de debut/fin) par episode, depuis les metadonnees source.
    root = pathlib.Path(src.root)
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    import pandas as pd
    ep = pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index").sort_index()
    n_ep = len(ep) if args.max_episodes is None else min(args.max_episodes, len(ep))
    print(f"Source: {args.src} ({len(ep)} episodes) -> {args.dst} en {args.width}x{args.height}")

    for e in range(n_ep):
        fr = int(ep.loc[e, "dataset_from_index"])
        to = int(ep.loc[e, "dataset_to_index"])
        for gi in range(fr, to):
            item = src[gi]
            frame = {}
            for k in img_keys:
                t = item[k]  # (3,H,W) float [0,1]
                arr = (t.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                frame[k] = cv2.resize(arr, (args.width, args.height), interpolation=cv2.INTER_AREA)
            frame["observation.state"] = item["observation.state"].numpy()
            frame["action"] = item["action"].numpy()
            frame["task"] = item["task"]
            dst.add_frame(frame)
        dst.save_episode()
        if (e + 1) % 10 == 0 or e + 1 == n_ep:
            print(f"  episode {e + 1}/{n_ep}")

    dst.finalize()
    print(f"OK -> dataset basse-res cree localement : {args.dst}")
    if args.push:
        dst.push_to_hub()
        print(f"Pousse sur le Hub : {args.dst}")


if __name__ == "__main__":
    main()
