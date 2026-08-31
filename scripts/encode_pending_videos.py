#!/usr/bin/env python
"""Force-encode videos for episodes saved before a batch_encoding_size threshold was reached.

Recovery tool for a lerobot-record crash/interrupt: with --dataset.video_encoding_batch_size=N,
raw per-frame images for completed episodes stay on disk until N episodes accumulate (see
LeRobotDataset.save_episode / _batch_save_episode_video). If recording dies before hitting that
threshold, this reads the on-disk dataset directly and encodes whatever's pending.

Usage:
    python scripts/encode_pending_videos.py --repo-id local/cubes --root ./datasets/sanity
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--root", required=True)
args = parser.parse_args()

dataset = LeRobotDataset(args.repo_id, root=args.root)
print(f"Dataset has {dataset.num_episodes} saved episodes.")

# CAUTION: video files are packed, multi-episode chunks with running chunk/file indices —
# re-encoding a range that was ALREADY encoded by a completed batch is not verified safe
# (risk of duplicating/corrupting that chunk). Only run this once, and only over episodes
# you're sure weren't already encoded (e.g. the whole dataset, if it crashed before ever
# reaching video_encoding_batch_size once — nothing encoded yet in that case).
dataset._batch_save_episode_video(0, dataset.num_episodes)
print("Done.")
