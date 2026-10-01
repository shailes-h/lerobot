# RoboColosseum VLA evals — runbook

How to serve a RoboColosseum fine-tune, hook up the real YAM robot, and run an eval session.

## 0. How it fits together

- **Checkpoints live on HF**, in [`RoboColosseum/BimanualYAM-models`](https://huggingface.co/RoboColosseum/BimanualYAM-models),
  laid out as `<model>/<task>/`, e.g. `gr00t/dustpan/`. Each folder is self-contained: weights plus
  every training-time config the loader needs (GR00T's `yam_config.py`, G0.5's `.hydra/config.yaml`,
  pi0.5's `train_config.py`, LingBot's `lingbotvla_cli.yaml` + `robot_config_yam_dustpan.yaml`, …).
- **`scripts/rc_serve_policy.py`** is the one policy server. You give it a model and a task; it pulls
  that folder into the standard HF cache (`~/.cache/huggingface`, or `$HF_HOME`) on first use and
  serves it. Every model answers the same json_numpy `GET`/`POST /act` contract (14-D joint state
  in, `(chunk, 14)` absolute joint actions out).
- **`third_party/`** holds the 5 upstream VLA repos as pinned git submodules. They are never
  modified — nothing is written into them, and no config lives in them. Each model has its own
  venv (table below): inside the submodule where that repo already gitignores `.venv`, otherwise
  under `envs/` (gitignored here).
- **`scripts/eval_yam_joint_policy.py`** is the robot-side client. Run it in the robot env
  (`/home/rc1/lerobot/.venv`), not in any model venv.

| `--policy` | Model | Venv | Chunk | VRAM | Status |
|---|---|---|---|---|---|
| `gr00t` | GR00T N1.7 | `third_party/Isaac-GR00T/.venv` | 16 | ~6.2 GB | ✅ verified from HF |
| `g05` | Galaxea G0.5 | `third_party/GalaxeaVLA/.venv` | 32 | ~11.4 GB | ✅ verified from HF |
| `molmoact2` | MolmoAct2 | `third_party/molmoact2/.venv` | 30 | ~10.8 GB | ✅ verified from HF |
| `pi05` | π0.5 (openpi, JAX) | `third_party/openpi/.venv` | 50 | ~18 GB (JAX preallocates) | ✅ verified from HF |
| `lingbot` | LingBot-VLA v2 | `envs/lingbot-vla-v2/.venv` | 50 | not measured | ✅ verified from HF |

## 1. Start a policy server

Run **one at a time** (section 5), with that model's own venv:

```bash
cd /home/rc1/lerobot

third_party/Isaac-GR00T/.venv/bin/python scripts/rc_serve_policy.py --policy=gr00t     --task=dustpan
third_party/GalaxeaVLA/.venv/bin/python  scripts/rc_serve_policy.py --policy=g05       --task=dustpan
third_party/molmoact2/.venv/bin/python   scripts/rc_serve_policy.py --policy=molmoact2 --task=dustpan
third_party/openpi/.venv/bin/python      scripts/rc_serve_policy.py --policy=pi05      --task=dustpan
envs/lingbot-vla-v2/.venv/bin/python     scripts/rc_serve_policy.py --policy=lingbot   --task=dustpan
```

The first run per checkpoint downloads it (5–25 GB); later runs load straight from the cache.
Each start also checks HF for a newer `main` and fetches only changed files. The first `/act`
call is slow while the model warms up (pi0.5: ~30 s of XLA compilation); later calls are fast.
Once it prints `[rc_serve_policy] serving '<policy>' on http://0.0.0.0:8000/act`, check it from
another terminal:

```bash
wget -qO- http://localhost:8000/act
# -> {"action_dim":14,"camera_keys":["top","left","right"],"state_dim":14}
```

### Picking the checkpoint

| Flag | Meaning |
|---|---|
| `--task=<task>` | Serve `<model>/<task>/` from `--hf_repo`, e.g. `dustpan` (later: `microwave`, `drawer`, `cups`). |
| `--hf_repo=<org>/<repo>` | Repo laid out as `<model>/<task>/`. Default `RoboColosseum/BimanualYAM-models`. |
| `--ckpt_path=<ref>` | Any other checkpoint; overrides the two above. A local dir, or `hf://<org>/<repo>[/<subfolder>][@<revision>]` (an `https://huggingface.co/...` URL works too). |
| `--port`, `--device` | Default `8000`, `cuda:0`. |
| `--norm_tag` | MolmoAct2 only, see below. |

Examples:

```bash
# Pin a specific HF revision
... --policy=gr00t --ckpt_path=hf://RoboColosseum/BimanualYAM-models/gr00t/dustpan@<commit>

# Baseline: the official MolmoAct2 YAM release (general YAM data, not Dustpan-specific)
third_party/molmoact2/.venv/bin/python scripts/rc_serve_policy.py --policy=molmoact2 \
  --ckpt_path=hf://allenai/MolmoAct2-BimanualYAM --norm_tag=yam_dual_molmoact2
```

The repo is private: log in once with `hf auth login` (any model venv has the `hf` CLI) before
the first download.

## 2. Hook up the real robot

- Power on both YAM arms, connect the 3 RealSense cameras (top `262522074294`, left wrist
  `260322271881`, right wrist `260322275072` — same serials/mounting as the recordings).
- Set up the scene the way training saw it.
- The robot env (`/home/rc1/lerobot/.venv`) is `uv`-created and has no `pip` binary — use
  `uv pip install <pkg>`, not `pip`/`python -m pip`.

## 3. Run the eval client

```bash
python scripts/eval_yam_joint_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.record_eef_pose=false \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://localhost:8000/act \
  --task="Clean the table." \
  --actions_per_chunk=8 \
  --max_episodes=10 \
  --record_dataset=true
```

Notes:
- `--robot.record_eef_pose=false` is **required** with `--record_dataset=true`, or the first frame
  write crashes with `KeyError: 'left_eef.x'` (section 6.3).
- `--task` here is the language instruction and must be exactly what the checkpoint was trained on —
  `"Clean the table."` for Dustpan.
- `--actions_per_chunk`: about half the model's chunk, or a fixed 8 across all models for a fair
  comparison.

## 4. Session controls (stdin, type + Enter)

| Key | Meaning |
|---|---|
| `b` | Home the arms, begin episode 1 |
| `s` / `f` | While an episode is running: mark it success / failure (homes and holds) |
| `n` | After resetting the scene: start the next episode (homes first) |
| `r` | Re-home on demand |
| `z` | Quit / finish the session |

Output lands in `eval_logs/<timestamp>/`: `videos/episode_NNN_<outcome>[_top].mp4`, `log.jsonl`,
`summary.txt`. With `--record_dataset=true` the raw dataset is at `eval_logs/<timestamp>/dataset/`;
encode it afterward:

```bash
python scripts/encode_pending_videos.py \
  --repo-id local/<timestamp>_clean_the_table \
  --root eval_logs/<timestamp>/dataset
```

## 5. Switching models / resource limits

**GPU (24 GB)**: one policy server at a time. pi0.5 alone takes ~18 GB (JAX reserves most of the
GPU up front), so nothing else fits next to it. Workflow: Ctrl+C the client, stop the server,
check `nvidia-smi` is near 0 MiB, start the next one.

**System RAM (30 GB)**: tight. Stop every server before anything else heavy:

```bash
pgrep -f "[r]c_serve_policy" | xargs -r kill
```

**Disk**: checkpoints accumulate in `~/.cache/huggingface/hub`. Remove ones you no longer need
with `hf cache delete` (interactive) rather than deleting cache folders by hand.

## 6. Troubleshooting / rebuilding an env

`rc_serve_policy.py` carries every code-side fix itself — re-cloning a submodule fresh still works.

### 6.1 Rebuilding a model's `.venv`

GR00T, G0.5, MolmoAct2:

```bash
cd third_party/<Isaac-GR00T|GalaxeaVLA|molmoact2>
uv venv --python <3.12|3.10|3.12> .venv
UV_HTTP_TIMEOUT=300 uv sync            # --frozen for GalaxeaVLA (its lock has an unrelated
                                        # aarch64-only resolution failure otherwise)
uv pip install flask json_numpy hf_transfer   # server deps, not in any upstream pyproject
```

pi0.5 (openpi ships its own `uv.lock`):

```bash
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=300 uv sync
uv pip install flask json_numpy hf_transfer
```

LingBot — its repo has no `.gitignore`, so the venv lives outside it, and its sources are put on
the path with a `.pth` file instead of `pip install -e` (which would write build files into the
submodule). Same package pins as upstream `tools/create_train_env.sh`, re-applied in the same order
because its two requirement files conflict (`mlflow` wants `pyarrow<20`):

```bash
cd /home/rc1/lerobot
export VIRTUAL_ENV=envs/lingbot-vla-v2/.venv UV_HTTP_TIMEOUT=300
R=third_party/lingbot-vla-v2
uv venv --python 3.12 $VIRTUAL_ENV
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0
uv pip install -r $R/requirements.txt
uv pip install --no-deps numpydantic==1.9.0 \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install -r $R/requirements-depth.txt
uv pip install -r $R/requirements.txt     # restore the core pins depth reqs bumped
uv pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
uv pip install huggingface_hub==0.34.0 flask json_numpy hf_transfer
SP=$($VIRTUAL_ENV/bin/python -c "import site;print(site.getsitepackages()[0])")
A=$(realpath $R)
printf '%s\n' "$A" "$A/lingbotvla/models/vla/vision_models/lingbot-depth" \
  "$A/lingbotvla/models/vla/vision_models/MoGe" > $SP/lingbot_vla_v2_src.pth
```

The model hardcodes `flash_attention_2`, so `flash-attn` must be installed; the prebuilt 2.8.3
wheel above does run on this GPU (sm_120).

`uv sync` in a repo that ships no `uv.lock` (molmoact2) writes one into the submodule — delete it
afterward so the submodule stays clean (`git -C third_party/<repo> status` should be empty).

GalaxeaVLA build quirks:
- If `uv sync` fails with "Invalid zip file structure", the LFS-tracked torchcodec wheel isn't
  materialized: put a portable `git-lfs` binary on `PATH`, then `git lfs install --local && git lfs pull`.
- `egl-probe` needs `cmake` (`uv pip install cmake`, `.venv/bin` on `PATH`) and
  `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
- `flash-attn-4`'s CUTLASS backend doesn't support this GPU (RTX 5090, sm_120):
  `uv pip uninstall flash-attn-4 nvidia-cutlass-dsl` so it falls back to SDPA.

### 6.2 What each loader does with the checkpoint folder

Useful if a new task's checkpoint fails to load:

- **GR00T** — imports `yam_config.py` from the folder (registers the `NEW_EMBODIMENT` modality
  config), and letterboxes the 640x360 top camera to 640x480 like training.
- **G0.5** — the folder is the training run dir (`model.pt`, `.hydra/config.yaml`,
  `dataset_stats.json`, `action_tokenizer.pt`). The shared Qwen3.5 processor comes from
  `OpenGalaxea/G05` on HF. Three upstream config-loading bugs are worked around in
  `_load_g05_config_from_run_dir` (unresolved `${tokenizer}` interpolations, cwd-relative
  `oc.load:` resolvers, `hf_processor_path` not redirected) — keep those fixes if the loader is
  ever rewritten.
- **MolmoAct2** — older exports need two bf16 source patches to `modeling_molmoact2.py`; when they
  apply, the server builds an overlay in `~/.cache/rc_serve_policy/molmoact2/` (symlinks to the
  checkpoint plus the patched file) instead of editing the HF cache. The current Dustpan export
  needs none and loads directly.
- **π0.5** — loads `CONFIG` from the folder's `train_config.py`; openpi is unmodified. The JAX
  `params/` are used (EMA weights). The config's repack transform only runs in training, so the
  server sends images straight in openpi's Aloha layout (`cam_high`/`cam_left_wrist`/
  `cam_right_wrist`, channel-first) with raw joint units (`adapt_to_pi=False`).
- **LingBot** — `lingbotvla_cli.yaml` hardcodes the training machine's paths, so the server writes
  a copy into `~/.cache/rc_serve_policy/lingbot/` with those paths pointed at HF cache snapshots of
  `robbyant/lingbot-vla-v2-6b`, `Qwen/Qwen3-VL-4B-Instruct` and `Ruicheng/moge-2-vitb-normal`
  (configs/tokenizer/aux files only — all weights come from our checkpoint). Robot config and norm
  stats come from the folder. Denoising uses 10 steps (the checkpoint's `num_steps`, also upstream's
  default); the request's `num_steps` is ignored.

`~/.cache/rc_serve_policy/` is disposable; it's rebuilt on the next start. The server also sends
Python bytecode there (`sys.pycache_prefix`), so importing a backend never writes `__pycache__/`
into a submodule.

### 6.3 `eval_yam_joint_policy.py` gotchas

- `json_numpy.patch()` must run **after** the first `robot.get_observation()`.
  `BiYamFollower.get_observation()` lazily imports `scipy` → `numpy.testing`, which runs its own
  `json.loads()` at import and crashes under the patched decoder.
- `--robot.record_eef_pose=false` is required with `--record_dataset=true`: otherwise the dataset
  schema includes `*_eef.*` columns this joint-only client never fills.

### 6.4 Slow downloads

The server enables `hf_transfer` automatically when it's installed in the venv (section 6.1),
which matters here: latency to the HF CDN is ~250–500 ms, so single-connection downloads crawl.
`hf_transfer` can't resume a partially downloaded file — an interrupted download restarts that
file from zero (finished files are kept).
