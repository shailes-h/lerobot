#!/usr/bin/env python
"""Interactive episode review/delete tool.

Shows the first/mid/last frame (from one camera) of each episode in `--episodes` in a local
web page, lets you click Keep/Delete per episode, and once every episode has a decision,
deletes the ones marked Delete and writes the result to `<root>_filtered` (a sibling
directory of `--root`) — the original dataset is never modified.

Usage:
    python scripts/interactive_episode_filter.py \
        --root ./datasets/transfer --repo-id local/transfer \
        --episodes 5,17,18,42,46 \
        [--camera observation.images.top] [--port 8765]

Notes:
- Reuses the same pandas/pyarrow nested-list workaround for `dataset_tools._load_episode_with_stats`
  that earlier one-off deletion scripts in this repo used (pandas 2.3.3 / pyarrow 25.0.1 can't
  read this project's deeply-nested per-arm EEF-pose stat columns).
- Frame extraction uses PyAV directly (no system ffmpeg dependency).
- The server exits automatically once the deletion (or a "keep everything" no-op) is done.
"""

import argparse
import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

import lerobot.datasets.dataset_tools as dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, DEFAULT_VIDEO_PATH


def _load_episode_with_stats_pa(src_dataset, episode_idx):
    """pyarrow-based replacement for dataset_tools._load_episode_with_stats.

    pandas.read_parquet chokes on this project's nested-list EEF-pose stat columns
    (TypeError: data type '...[pyarrow]' not understood) — read with pyarrow directly
    instead and convert only the stats/* columns to numpy arrays (required by
    aggregate_stats()/compute_stats.py's _validate_stat_value).
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]
    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    table = pq.read_table(parquet_path)
    idx_col = table.column("episode_index").to_pylist()
    row_i = idx_col.index(episode_idx)
    row = {}
    for name in table.column_names:
        value = table.column(name)[row_i].as_py()
        if name.startswith("stats/") and isinstance(value, list):
            value = np.array(value)
        row[name] = value
    return row


dataset_tools._load_episode_with_stats = _load_episode_with_stats_pa


def extract_frame_jpeg_b64(video_path: Path, target_t: float) -> str | None:
    """Decode the first frame at/after `target_t` (seconds) and return it as a base64 JPEG."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    container.seek(int(target_t / stream.time_base), stream=stream)
    result = None
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t >= target_t:
            buf = io.BytesIO()
            frame.to_image().save(buf, format="JPEG", quality=85)
            result = base64.b64encode(buf.getvalue()).decode("ascii")
            break
    container.close()
    return result


def build_episode_frames(dataset: LeRobotDataset, episodes: list[int], camera: str) -> dict:
    frames = {}
    for ep in episodes:
        ep_meta = dataset.meta.episodes[ep]
        chunk_idx = ep_meta[f"videos/{camera}/chunk_index"]
        file_idx = ep_meta[f"videos/{camera}/file_index"]
        from_ts = ep_meta[f"videos/{camera}/from_timestamp"]
        to_ts = ep_meta[f"videos/{camera}/to_timestamp"]
        video_path = dataset.root / DEFAULT_VIDEO_PATH.format(
            video_key=camera, chunk_index=chunk_idx, file_index=file_idx
        )
        targets = {"start": from_ts + 0.2, "mid": (from_ts + to_ts) / 2, "end": to_ts - 0.2}
        frames[ep] = {label: extract_frame_jpeg_b64(video_path, t) for label, t in targets.items()}
        print(f"  extracted frames for episode {ep}")
    return frames


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Episode Review</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1b1b1f; color: #eee; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; }}
  .episode {{ border: 1px solid #444; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
  .episode.decided-keep {{ border-color: #2ecc71; }}
  .episode.decided-delete {{ border-color: #e74c3c; opacity: 0.6; }}
  .frames {{ display: flex; gap: 8px; margin: 10px 0; }}
  .frames img {{ width: 260px; border-radius: 4px; }}
  .frames .label {{ text-align: center; font-size: 12px; color: #aaa; }}
  .buttons button {{ font-size: 15px; padding: 8px 20px; margin-right: 10px; border: none; border-radius: 6px; cursor: pointer; }}
  .keep {{ background: #2ecc71; color: #063; }}
  .delete {{ background: #e74c3c; color: #300; }}
  .status {{ font-size: 13px; color: #aaa; margin-left: 8px; }}
  #submit-bar {{ position: sticky; bottom: 0; background: #1b1b1f; padding: 16px 0; border-top: 1px solid #444; }}
  #submit {{ font-size: 16px; padding: 10px 28px; background: #3498db; color: #fff; border: none; border-radius: 6px; cursor: pointer; }}
  #submit:disabled {{ background: #555; cursor: not-allowed; }}
  #result {{ margin-top: 16px; white-space: pre-wrap; font-family: monospace; }}
</style>
</head>
<body>
<h1>Review episodes from {root} (camera: {camera})</h1>
<div id="episodes"></div>
<div id="submit-bar">
  <button id="submit" disabled>Submit decisions</button>
  <span id="progress"></span>
</div>
<div id="result"></div>
<script>
const data = {data_json};
const decisions = {{}};
const container = document.getElementById("episodes");

for (const ep of Object.keys(data)) {{
  const div = document.createElement("div");
  div.className = "episode";
  div.id = "ep-" + ep;
  const framesHtml = ["start", "mid", "end"].map(label => {{
    const b64 = data[ep][label];
    return `<div><div class="label">${{label}}</div>` +
      (b64 ? `<img src="data:image/jpeg;base64,${{b64}}">` : "<div>no frame</div>") + `</div>`;
  }}).join("");
  div.innerHTML = `<b>Episode ${{ep}}</b>
    <div class="frames">${{framesHtml}}</div>
    <div class="buttons">
      <button class="keep" onclick="decide(${{ep}}, 'keep')">Keep</button>
      <button class="delete" onclick="decide(${{ep}}, 'delete')">Delete</button>
      <span class="status" id="status-${{ep}}">undecided</span>
    </div>`;
  container.appendChild(div);
}}

function updateProgress() {{
  const total = Object.keys(data).length;
  const decided = Object.keys(decisions).length;
  document.getElementById("progress").innerText = `${{decided}}/${{total}} decided`;
  document.getElementById("submit").disabled = decided < total;
}}

function decide(ep, choice) {{
  decisions[ep] = choice;
  const div = document.getElementById("ep-" + ep);
  div.className = "episode decided-" + choice;
  document.getElementById("status-" + ep).innerText = choice;
  updateProgress();
}}

document.getElementById("submit").onclick = async () => {{
  document.getElementById("submit").disabled = true;
  document.getElementById("result").innerText = "Deleting... this can take a few minutes (video re-encoding).";
  const resp = await fetch("/submit", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(decisions),
  }});
  const text = await resp.text();
  document.getElementById("result").innerText = text;
}};

updateProgress();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--repo-id", required=True, help="repo_id to load the dataset with")
    parser.add_argument("--episodes", required=True, help="Comma-separated zero-indexed episode indices")
    parser.add_argument("--camera", default="observation.images.top", help="Camera key to preview")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    episodes = sorted(int(e) for e in args.episodes.split(","))

    print(f"Loading dataset {args.repo_id} from {root} ...")
    dataset = LeRobotDataset(args.repo_id, root=root, episodes=episodes)
    print(f"Loaded. total_episodes={dataset.meta.total_episodes}")

    print("Extracting preview frames (start/mid/end) for each episode...")
    frames = build_episode_frames(dataset, episodes, args.camera)

    page_html = PAGE_TEMPLATE.format(
        root=str(root), camera=args.camera, data_json=json.dumps(frames)
    )

    result_holder = {}
    shutdown_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # quiet

        def do_GET(self):
            if self.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            body = page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/submit":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            decisions = json.loads(self.rfile.read(length))
            to_delete = sorted(int(ep) for ep, choice in decisions.items() if choice == "delete")
            kept = sorted(int(ep) for ep, choice in decisions.items() if choice == "keep")
            print(f"\nDecisions received. Delete: {to_delete}  Keep: {kept}")

            output_dir = root.parent / f"{root.name}_filtered"
            try:
                if not to_delete:
                    text = "No episodes marked for deletion — nothing to do. No new dataset was written."
                else:
                    print(f"Deleting {to_delete} -> writing to {output_dir}")
                    new_dataset = dataset_tools.delete_episodes(
                        dataset,
                        episode_indices=to_delete,
                        output_dir=str(output_dir),
                        repo_id=f"local/{output_dir.name}",
                    )
                    text = (
                        f"Done.\nDeleted episodes: {to_delete}\n"
                        f"New dataset: {output_dir}\n"
                        f"New episode count: {new_dataset.meta.total_episodes}\n"
                        f"New frame count: {new_dataset.meta.total_frames}\n"
                        f"Original dataset at {root} was not modified."
                    )
                print(text)
            except Exception as e:  # noqa: BLE001
                text = f"ERROR during deletion: {e!r}"
                print(text)

            result_holder["text"] = text
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            shutdown_event.set()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nOpen {url} to review episodes.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    shutdown_event.wait()
    server.shutdown()
    print("\nServer stopped.")


if __name__ == "__main__":
    main()
