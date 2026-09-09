#!/usr/bin/env python
"""Real-robot eval client for the GAM bimanual YAM HTTP policy server
(see `host_gam_server.py` for the server-side contract).

===============================================================================
ACTION SPACE — JOINT ANGLES (no IK needed)
===============================================================================

The GAM policy outputs 14-D ABSOLUTE joint targets:

    [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]

These are commanded directly — no IK, no EEF conversion. Much simpler than the
MolmoAct2 EEF-space server.

===============================================================================
CLICKS — FIRST-FRAME INTERACTIVE ANNOTATION
===============================================================================

At the start of each episode, a matplotlib window pops up showing the first
camera frame. The user clicks 4 points, one per slot:

    1. pick_1   — where to pick the first object
    2. place_1  — where to place it
    3. pick_2   — where to pick the second object
    4. place_2  — where to place it

These clicks are sent to the server on every subsequent frame of that episode.
When the server has --track-clicks enabled, CoTracker3 online tracks these
clicks across frames so they follow object motion (matching training-time
precomputation). The client calls POST /reset between episodes to re-seed.

===============================================================================
DEPTH + INTRINSICS + EXTRINSICS
===============================================================================

The RealSense camera is configured with `use_depth=True`. Each frame, `read_depth()`
provides uint16 mm depth aligned to the RGB stream. Intrinsics and extrinsics are
fixed constants matching the training setup.

===============================================================================
WIRE PROTOCOL
===============================================================================

    POST /act  ->  action inference
      request:  {rgb, depth, state, K, E, clicks, state_hist?, instruction?, timestamp?}
      response: {actions: (16, 14), dt_ms, timestamp?}

Usage:

    python scripts/eval_gam_policy.py \\
      --robot.left_arm_port=1235 --robot.right_arm_port=1234 \\
      --robot.cameras='{
    top: {"type": "intelrealsense", "serial_number_or_name": "262522075787",
          "width": 1280, "height": 720, "fps": 15, "use_depth": true}
    }' \\
      --server_url=https://untaken-eskimo-penholder.ngrok-free.dev/act \\
      --task="Put apple into the pan and the can into the bowl." \\
      --actions_per_chunk=16 \\
      --duration_s=300 --record_dataset=true
"""

import json
import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.bi_yam_follower.bi_yam_follower import BiYamFollower
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import BiYamFollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say

SIDES = ("left", "right")
CAM_KEY = "top"  # single head camera

SLOT_NAMES = ["pick_1", "place_1", "pick_2", "place_2"]
SLOT_COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0)]  # red, green, blue, orange

VIDEO_TILE_WH = (640, 360)


# ---- Joint state helpers ----

def _state14_from_obs(obs: dict) -> np.ndarray:
    """Build the 14-D joint state: [left_j0..j5, left_gripper, right_j0..j5, right_gripper]."""
    row = []
    for side in SIDES:
        for i in range(6):
            row.append(obs[f"{side}_joint_{i}.pos"])
        row.append(obs[f"{side}_gripper.pos"])
    return np.asarray(row, dtype=np.float32)


# ---- Click annotation UI ----

def _collect_clicks(rgb: np.ndarray, slot_names: list[str], slot_colors: list[tuple]) -> list[list[float]]:
    """Show a matplotlib window and let the user click one point per slot.
    Returns [[u, v], ...] in pixel coordinates."""
    import importlib

    import matplotlib

    # Pick the first GUI backend whose bindings are actually importable. Tk is preferred
    # (matplotlib's default interactive backend), but it needs the system `python3-tk`
    # package -- `tkinter` is stdlib, so a venv can't supply it, and without that package
    # `matplotlib.use("TkAgg")` succeeds while the failure surfaces later as a confusing
    # ModuleNotFoundError from the first `plt.subplots()` call. Importing the backend
    # module here forces that failure up front so we can fall back to Qt (PyQt5 is already
    # installed in this venv) instead of dying at the click prompt mid-episode.
    for backend in ("TkAgg", "QtAgg"):
        try:
            importlib.import_module(f"matplotlib.backends.backend_{backend.lower()}")
        except ImportError:
            continue
        matplotlib.use(backend, force=True)
        break
    else:
        raise ImportError(
            "No usable matplotlib GUI backend for the click prompt. Install one with "
            "`sudo apt install python3-tk` (Tk) or `uv pip install pyqt5` (Qt)."
        )
    import matplotlib.pyplot as plt

    clicks = []
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(rgb)
    ax.set_title(f"Click: {slot_names[0]} (1/{len(slot_names)})")
    ax.axis("off")

    # Show existing clicks as colored dots
    scatter_artists = []
    text_artists = []

    def onclick(event):
        if event.inaxes != ax or event.xdata is None:
            return
        u, v = float(event.xdata), float(event.ydata)
        idx = len(clicks)
        if idx >= len(slot_names):
            return
        clicks.append([u, v])
        color = np.array(slot_colors[idx]) / 255.0
        sc = ax.scatter(u, v, c=[color], s=120, edgecolors="white", linewidths=1.5, zorder=5)
        txt = ax.text(u + 8, v - 8, slot_names[idx], color=color, fontsize=10,
                      fontweight="bold", zorder=6)
        scatter_artists.append(sc)
        text_artists.append(txt)

        if len(clicks) < len(slot_names):
            ax.set_title(f"Click: {slot_names[len(clicks)]} ({len(clicks)+1}/{len(slot_names)})")
        else:
            ax.set_title("All clicks recorded — close window to continue")
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    if len(clicks) != len(slot_names):
        raise RuntimeError(
            f"Expected {len(slot_names)} clicks, got {len(clicks)}. "
            "Please click exactly one point per slot."
        )
    print(f"[eval] Clicks recorded: {dict(zip(slot_names, clicks))}")
    return clicks


# ---- Dataclass configs ----

@dataclass
class DatasetRecordConfig:
    repo_id: str
    root: str | Path | None = None
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    private: bool = True
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1


@dataclass
class EvalGamPolicyConfig:
    robot: BiYamFollowerConfig
    record_dataset: bool = False
    server_url: str = "https://untaken-eskimo-penholder.ngrok-free.dev/act"
    task: str = "Put apple into the pan and the can into the bowl."
    actions_per_chunk: int = 16
    control_hz: float = 15.0  # match camera fps
    duration_s: float | None = None
    play_sounds: bool = True
    # Home position for 6 joints (gripper handled separately)
    reset_joint_pos: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_duration_s: float = 4.0
    reset_hz: float = 20.0
    max_episodes: int = 50
    output_dir: str = "eval_logs"
    video_fps: float | None = None
    # Camera calibration — correct intrinsics for aligned D435 (1280x720)
    intrinsic_fx: float = 907.90
    intrinsic_fy: float = 905.86
    intrinsic_cx: float = 643.41
    intrinsic_cy: float = 373.91
    # cam2base extrinsic — 12 floats, row-major 3x4 (rotation | translation)
    cam2base: tuple[float, ...] = (
        -0.98940003,  0.06120000, -0.13210000,  0.00894280,
         0.12830000,  0.79490000, -0.59299999, -0.04316070,
         0.06870000, -0.60360003, -0.79430002,  0.41122191,
    )


def _current_arm_joint_pos(obs: dict, side: str) -> np.ndarray:
    return np.array([obs[f"{side}_joint_{i}.pos"] for i in range(6)])


def _slow_reset(
    robot: BiYamFollower,
    robot_action_processor,
    target_joint_pos: np.ndarray,
    duration_s: float,
    hz: float,
) -> None:
    """Linearly interpolate both arms' 6 joints to target + open grippers."""
    obs = robot.get_observation()
    start = {side: _current_arm_joint_pos(obs, side) for side in SIDES}
    gripper_start = {side: obs[f"{side}_gripper.pos"] for side in SIDES}

    n_steps = max(1, int(duration_s * hz))
    period_s = 1.0 / hz
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = step / n_steps
        action: dict[str, float] = {}
        for side in SIDES:
            q = (1 - alpha) * start[side] + alpha * target_joint_pos
            for j, val in enumerate(q):
                action[f"{side}_joint_{j}.pos"] = float(val)
            action[f"{side}_gripper.pos"] = float((1 - alpha) * gripper_start[side] + alpha * 1.0)

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)
        busy_wait(period_s - (time.perf_counter() - t0))


def _request_with_retries(fn, retries: int = 5, backoff_s: float = 2.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logging.warning(f"request attempt {attempt}/{retries} failed: {e!r}; retrying in {backoff_s}s")
            time.sleep(backoff_s)
    raise RuntimeError(f"request failed after {retries} attempts") from last_exc


# ---- Keyboard input ----

def _start_key_listener() -> queue.Queue:
    q: queue.Queue = queue.Queue()

    def _reader():
        while True:
            try:
                line = input()
            except EOFError:
                break
            q.put(line.strip().lower())

    threading.Thread(target=_reader, daemon=True).start()
    return q


def _wait_for_key(q: queue.Queue, valid: set[str]) -> str:
    while True:
        line = q.get()
        if line in valid:
            return line
        print(f"[eval] unrecognized input {line!r}; expected one of {sorted(valid)}")


def _poll_key(q: queue.Queue, valid: set[str]) -> str | None:
    result = None
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line in valid:
            result = line
        else:
            print(f"[eval] unrecognized input {line!r}; expected one of {sorted(valid)}")
    return result


# ---- Video recording ----

def _compose_frame(obs: dict) -> np.ndarray:
    """Single camera frame for the tiled mp4."""
    img = cv2.resize(np.asarray(obs[CAM_KEY]), VIDEO_TILE_WH)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _top_frame(obs: dict) -> np.ndarray:
    return cv2.cvtColor(np.asarray(obs[CAM_KEY]), cv2.COLOR_RGB2BGR)


class EpisodeRecorder:
    def __init__(self, raw_path: Path, raw_top_path: Path, fps: float, obs: dict):
        shape = _compose_frame(obs).shape
        top_shape = _top_frame(obs).shape
        h, w = shape[:2]
        top_h, top_w = top_shape[:2]

        self.raw_path = raw_path
        self.raw_top_path = raw_top_path
        self.fps = fps
        self._writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._top_writer = cv2.VideoWriter(
            str(raw_top_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (top_w, top_h)
        )
        self._top_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
        self.n_frames = 0

    def write(self, obs: dict) -> None:
        self._writer.write(_compose_frame(obs))
        self._top_writer.write(_top_frame(obs))
        self.n_frames += 1

    def close(self) -> None:
        self._writer.release()
        self._top_writer.release()

    def finalize(self, final_path: Path, final_top_path: Path, outcome: str) -> None:
        self.close()
        self.raw_path.replace(final_path)
        self.raw_top_path.replace(final_top_path)


# ---- Session logging ----

class SessionLog:
    def __init__(self, output_dir: str, timestamp: str | None = None):
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / self.timestamp
        self.videos_dir = self.session_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_dir / "log.jsonl"
        self.summary_path = self.session_dir / "summary.txt"
        self.records: list[dict] = []
        logging.info(f"Session log directory: {self.session_dir}")

    def record_episode(self, episode: int, outcome: str, video_path: Path,
                       n_frames: int, fps: float) -> None:
        entry = {
            "episode": episode,
            "outcome": outcome,
            "video": str(video_path.relative_to(self.session_dir)),
            "n_frames": n_frames,
            "duration_s": n_frames / fps if fps else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.records.append(entry)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._write_summary(final=False)

    def _write_summary(self, final: bool) -> None:
        n = len(self.records)
        n_success = sum(1 for r in self.records if r["outcome"] == "success")
        rate = (n_success / n * 100.0) if n else 0.0
        lines = [
            f"{'FINAL' if final else 'Running'} summary -- {n} episode(s), {n_success} success, "
            f"success rate {rate:.1f}%",
            "",
        ]
        for r in self.records:
            lines.append(f"  ep {r['episode']:03d}: {r['outcome']:7s} ({r['video']})")
        self.summary_path.write_text("\n".join(lines) + "\n")

    def finalize(self) -> str:
        self._write_summary(final=True)
        n = len(self.records)
        n_success = sum(1 for r in self.records if r["outcome"] == "success")
        rate = (n_success / n * 100.0) if n else 0.0
        return f"{n} episode(s), {n_success} success, success rate {rate:.1f}%"


_CONTROLS = {
    "start": "b: begin episode 1 (homes first)   |   r: home/reset now   |   z: quit",
    "running": "s: mark SUCCESS   |   f: mark FAILURE",
    "waiting": "n: next episode   |   r: home/reset now   |   z: finish session",
}


def _print_controls(state: str) -> None:
    print(f"[eval] controls -- {_CONTROLS[state]}")


def _home(cfg: EvalGamPolicyConfig, robot: BiYamFollower, robot_action_processor) -> None:
    log_say("Resetting to home position", cfg.play_sounds, blocking=True)
    _slow_reset(
        robot, robot_action_processor,
        np.asarray(cfg.reset_joint_pos, dtype=np.float32),
        cfg.reset_duration_s, cfg.reset_hz,
    )


def _cleanup_and_exit(cfg, robot, robot_action_processor, session, reason):
    print(f"\n[eval] {reason} -- homing arms and closing out the session.")
    try:
        _home(cfg, robot, robot_action_processor)
    except Exception as e:
        logging.warning(f"home-on-cleanup failed: {e!r}")
    summary = session.finalize()
    print(f"[eval] {summary}")
    print(f"[eval] session log: {session.session_dir}")
    try:
        robot.disconnect()
    except Exception as e:
        logging.warning(f"disconnect-on-cleanup failed: {e!r}")


def _get_depth_m(robot: BiYamFollower) -> np.ndarray:
    """Read depth from the RealSense camera, return as float32 metres."""
    cam = robot.cameras[CAM_KEY]
    depth_mm = cam.read_depth()  # uint16, millimetres
    return depth_mm.astype(np.float32) / 1000.0


def _build_dataset_record_config(cfg: EvalGamPolicyConfig, timestamp: str) -> DatasetRecordConfig:
    task_slug = "".join(c if c.isalnum() else "_" for c in cfg.task.strip().lower()).strip("_")
    task_slug = "_".join(filter(None, task_slug.split("_")))
    repo_id = f"local/{timestamp}_{task_slug}" if task_slug else f"local/{timestamp}"
    root = Path(cfg.output_dir) / timestamp / "dataset"
    return DatasetRecordConfig(
        repo_id=repo_id, root=root, fps=int(round(cfg.control_hz)),
        video_encoding_batch_size=10**6,
    )


def _create_dataset(dataset_cfg, robot, teleop_action_processor, robot_observation_processor):
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_cfg.video,
        ),
    )
    features = combine_feature_dicts(features, robot.extra_dataset_features or {})
    for old_key, new_key in (robot.dataset_feature_renames or {}).items():
        if old_key in features:
            features[new_key] = features.pop(old_key)

    return LeRobotDataset.create(
        dataset_cfg.repo_id,
        dataset_cfg.fps,
        root=dataset_cfg.root,
        robot_type=robot.name,
        features=features,
        use_videos=dataset_cfg.video,
        image_writer_processes=dataset_cfg.num_image_writer_processes,
        image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=dataset_cfg.video_encoding_batch_size,
    )


def _run_episode(
    cfg: EvalGamPolicyConfig,
    robot: BiYamFollower,
    robot_action_processor,
    requests,
    json_numpy,
    key_q: queue.Queue,
    episode: int,
    session: SessionLog,
    clicks: list[list[float]],
    prev_state: np.ndarray | None,
    dataset: LeRobotDataset | None,
    robot_observation_processor,
    INTRINSIC_K: np.ndarray,
    EXTRINSIC_E: np.ndarray,
) -> tuple[str, "EpisodeRecorder", list[list[float]]]:
    """Home, then run the policy loop until s/f is typed.
    Returns (outcome, recorder, clicks)."""
    _home(cfg, robot, robot_action_processor)

    video_fps = cfg.video_fps or cfg.control_hz
    raw_path = session.session_dir / f"_raw_tmp_episode_{episode:03d}.mp4"
    raw_top_path = session.session_dir / f"_raw_tmp_episode_{episode:03d}_top.mp4"
    obs = robot.get_observation()
    recorder = EpisodeRecorder(raw_path, raw_top_path, video_fps, obs)

    # ---- Collect clicks for this episode ----
    rgb_first = np.asarray(obs[CAM_KEY])
    print(f"[eval] Annotate clicks for episode {episode} (4 clicks: pick_1, place_1, pick_2, place_2)")
    clicks = _collect_clicks(rgb_first, SLOT_NAMES, SLOT_COLORS)

    # Reset server-side CoTracker state for the new episode
    reset_url = cfg.server_url.replace("/act", "/reset")
    try:
        reset_resp = requests.post(
            reset_url,
            headers={"ngrok-skip-browser-warning": "1"},
            timeout=15,
        )
        reset_data = reset_resp.json()
        if reset_data.get("tracking"):
            logging.info("Server CoTracker reset OK — click tracking ACTIVE")
        else:
            logging.info("Server reset OK — click tracking not available (static clicks)")
    except Exception as e:
        logging.warning(f"Failed to reset server tracker (continuing anyway): {e!r}")

    log_say(f"Starting episode {episode}: {cfg.task}", cfg.play_sounds, blocking=True)
    print(f"[eval] episode {episode} RUNNING")
    _print_controls("running")

    period_s = 1.0 / cfg.control_hz
    t_start = time.perf_counter()
    step = 0
    outcome: str | None = None
    time_capped = False
    cur_state = _state14_from_obs(obs)

    while outcome is None:
        elapsed = time.perf_counter() - t_start
        if cfg.duration_s is not None and elapsed >= cfg.duration_s:
            if not time_capped:
                print(f"[eval] episode {episode}: duration_s={cfg.duration_s} reached — "
                      "type 's' or 'f' to label.")
                time_capped = True
            key = _wait_for_key(key_q, {"s", "f"})
            outcome = "success" if key == "s" else "failure"
            break

        obs = robot.get_observation()
        recorder.write(obs)

        rgb = np.asarray(obs[CAM_KEY])  # (720, 1280, 3) uint8 RGB
        depth_m = _get_depth_m(robot)   # (720, 1280) float32 metres
        cur_state = _state14_from_obs(obs)

        # Build state history (2 steps)
        if prev_state is None:
            state_hist = np.stack([cur_state, cur_state])  # (2, 14)
        else:
            state_hist = np.stack([prev_state, cur_state])  # (2, 14)

        # ---- Compress images to keep payload small over ngrok ----
        import base64 as _b64
        # JPEG-encode RGB (~100 KB vs ~3.5 MB raw base64)
        _, rgb_jpg = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        rgb_jpg_b64 = _b64.b64encode(rgb_jpg.tobytes()).decode("ascii")
        # PNG-encode depth as uint16 mm (lossless, ~200 KB vs ~4.7 MB raw base64)
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        _, depth_png = cv2.imencode(".png", depth_mm)
        depth_png_b64 = _b64.b64encode(depth_png.tobytes()).decode("ascii")

        payload = {
            "rgb_jpg": rgb_jpg_b64,
            "depth_png": depth_png_b64,
            "state": cur_state,
            "state_hist": state_hist,
            "K": INTRINSIC_K.astype(np.float32),
            "E": EXTRINSIC_E.astype(np.float32),
            "clicks": clicks,
            "instruction": cfg.task,
            "reset": step == 0,  # seed CoTracker on first frame
            "debug_viz": True,
        }

        t0 = time.perf_counter()
        body = json_numpy.dumps(payload)
        if step == 0:
            logging.info(f"Payload size: {len(body) / 1024:.1f} KB")
        resp = _request_with_retries(
            lambda: requests.post(
                cfg.server_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "1",
                },
                timeout=60,
            )
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"server returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        resp_data = json_numpy.loads(resp.text)
        if "error" in resp_data:
            raise RuntimeError(f"server returned an error: {resp_data['error']}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        actions = np.asarray(resp_data["actions"], dtype=np.float32)  # (16, 14)
        tracked = resp_data.get("tracked_clicks")
        track_info = ""
        if tracked is not None:
            track_info = " tracked_clicks=" + ",".join(
                f"({c[0]:.0f},{c[1]:.0f})" for c in tracked
            )
        logging.info(
            f"episode={episode} step={step} round_trip_ms={dt_ms:.1f} "
            f"server_dt_ms={resp_data.get('dt_ms', float('nan')):.1f}{track_info}"
        )

        # Save debug viz image if returned
        debug_viz_b64 = resp_data.get("debug_viz")
        if debug_viz_b64:
            debug_dir = session.session_dir / "debug_viz" / f"episode_{episode:03d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            viz_bytes = _b64.b64decode(debug_viz_b64)
            viz_path = debug_dir / f"step_{step:05d}.jpg"
            viz_path.write_bytes(viz_bytes)
            if step == 0:
                logging.info(f"Debug viz saving to {debug_dir}")

        n_exec = min(cfg.actions_per_chunk, len(actions))
        for i in range(n_exec):
            row_start = time.perf_counter()

            key = _poll_key(key_q, {"s", "f"})
            if key is not None:
                outcome = "success" if key == "s" else "failure"
                break

            # Directly command joint angles — no IK needed!
            row = actions[i]
            action: dict[str, float] = {}
            for side_idx, side in enumerate(SIDES):
                offset = side_idx * 7
                for j in range(6):
                    action[f"{side}_joint_{j}.pos"] = float(row[offset + j])
                action[f"{side}_gripper.pos"] = float(row[offset + 6])

            robot_obs = robot.get_observation()
            recorder.write(robot_obs)
            prev_state = cur_state
            cur_state = _state14_from_obs(robot_obs)

            if dataset is not None:
                observation_frame = build_dataset_frame(
                    dataset.features, robot_observation_processor(robot_obs), prefix=OBS_STR
                )
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": cfg.task})

            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)
            busy_wait(period_s - (time.perf_counter() - row_start))

        step += 1

    print(f"[eval] episode {episode} marked {outcome.upper()} — homing.")
    log_say(f"Episode {outcome}", cfg.play_sounds, blocking=True)
    _home(cfg, robot, robot_action_processor)

    return outcome, recorder, clicks


@parser.wrap()
def eval_policy(cfg: EvalGamPolicyConfig):
    init_logging()
    logging.info(cfg)

    import json_numpy
    import requests

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    # GAM policy outputs joint angles only — no EEF poses in the action dict,
    # so the dataset must not expect them.
    cfg.robot.record_eef_pose = False
    robot = BiYamFollower(cfg.robot)
    robot.connect()

    # Force one observation to trigger lazy imports before json_numpy patch
    obs = robot.get_observation()
    _state14_from_obs(obs)

    json_numpy.patch()

    # ---- Build calibration matrices from config ----
    INTRINSIC_K = np.array([
        [cfg.intrinsic_fx, 0.0, cfg.intrinsic_cx],
        [0.0, cfg.intrinsic_fy, cfg.intrinsic_cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    EXTRINSIC_E = np.asarray(cfg.cam2base, dtype=np.float64).reshape(3, 4)
    logging.info(f"Intrinsics K: fx={cfg.intrinsic_fx} fy={cfg.intrinsic_fy} "
                 f"cx={cfg.intrinsic_cx} cy={cfg.intrinsic_cy}")
    logging.info(f"Extrinsic E (cam2base 3x4):\n{EXTRINSIC_E}")

    # Health check
    health = _request_with_retries(
        lambda: requests.get(
            cfg.server_url,
            headers={"ngrok-skip-browser-warning": "1"},
            timeout=15,
        ).json()
    )
    logging.info(f"Server health: {health}")
    expected_action_dim = 14
    if health.get("action_dim") != expected_action_dim:
        raise RuntimeError(
            f"Server action_dim={health.get('action_dim')}, expected {expected_action_dim}"
        )

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.record_dataset:
        dataset_cfg = _build_dataset_record_config(cfg, session_timestamp)
        logging.info(f"Recording dataset RAW to {dataset_cfg.root}")
        dataset = _create_dataset(dataset_cfg, robot, teleop_action_processor,
                                  robot_observation_processor)
    else:
        dataset_cfg = None
        dataset = None
    dataset_closed = False

    def close_dataset(interrupted: bool = False):
        nonlocal dataset_closed
        if dataset_closed or dataset is None:
            dataset_closed = True
            return
        if interrupted:
            ep_idx = dataset.num_episodes
            for key in dataset.meta.video_keys:
                img_dir = dataset._get_image_file_path(
                    episode_index=ep_idx, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    shutil.rmtree(img_dir)
        dataset.finalize()
        logging.info(
            f"Dataset saved RAW at {dataset_cfg.root} — {dataset.num_episodes} episode(s). "
            f"Run: python scripts/encode_pending_videos.py "
            f"--repo-id {dataset_cfg.repo_id} --root {dataset_cfg.root}"
        )
        dataset_closed = True

    session = SessionLog(cfg.output_dir, timestamp=session_timestamp)
    key_q = _start_key_listener()

    try:
        print(f"\n[eval] Session ready ({cfg.max_episodes} episode(s) max).")
        _print_controls("start")
        while True:
            key = _wait_for_key(key_q, {"b", "r", "z"})
            if key == "r":
                _home(cfg, robot, robot_action_processor)
                _print_controls("start")
                continue
            break

        if key == "z":
            close_dataset()
            _cleanup_and_exit(cfg, robot, robot_action_processor, session,
                              "quit before any episode")
            return

        episode = 1
        clicks = None
        prev_state = None
        while True:
            outcome, recorder, clicks = _run_episode(
                cfg, robot, robot_action_processor, requests, json_numpy,
                key_q, episode, session, clicks, prev_state,
                dataset, robot_observation_processor,
                INTRINSIC_K, EXTRINSIC_E,
            )

            final_path = session.videos_dir / f"episode_{episode:03d}_{outcome}.mp4"
            final_top_path = session.videos_dir / f"episode_{episode:03d}_{outcome}_top.mp4"
            recorder.finalize(final_path, final_top_path, outcome)
            if dataset is not None:
                dataset.save_episode()
            session.record_episode(episode, outcome, final_path, recorder.n_frames, recorder.fps)
            print(f"[eval] saved {final_path}")
            print(f"[eval] saved {final_top_path}")

            if episode >= cfg.max_episodes:
                print(f"[eval] max_episodes={cfg.max_episodes} reached.")
                break

            print(f"[eval] episode {episode} logged. Reset scene, then:")
            _print_controls("waiting")
            while True:
                key = _wait_for_key(key_q, {"n", "r", "z"})
                if key == "r":
                    _home(cfg, robot, robot_action_processor)
                    _print_controls("waiting")
                    continue
                break

            if key == "z":
                print(f"[eval] finishing early at episode {episode}.")
                break
            episode += 1
            prev_state = None  # reset history between episodes

        summary = session.finalize()
        print(f"[eval] {summary}")
        print(f"[eval] session log: {session.session_dir}")
        close_dataset()
        robot.disconnect()

    except KeyboardInterrupt:
        close_dataset(interrupted=True)
        _cleanup_and_exit(cfg, robot, robot_action_processor, session, "Ctrl+C received")

    except Exception:
        close_dataset(interrupted=True)
        _cleanup_and_exit(cfg, robot, robot_action_processor, session,
                          "evaluation aborted by an error")
        raise


def main():
    eval_policy()


if __name__ == "__main__":
    main()
