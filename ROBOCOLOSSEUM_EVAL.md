# RoboColosseum Dustpan VLA evals — runbook

This is the eval-PC-specific runbook: how to start a policy server, hook up the real YAM
robot, and run an eval session, for whichever RoboColosseum Dustpan fine-tune you want to
test. Background/rationale for how this was all wired up lives in
`EVAL_LOCAL_ROBOCOLOSSEUM.md`; this doc is the "just tell me the commands" version, kept in
sync with what's actually been verified working on this machine.

## 0. What's already set up here

- **`third_party/`**: `Isaac-GR00T`, `GalaxeaVLA`, `molmoact2` as git submodules, each with its
  own `.venv` (created with `uv venv`, dependencies installed with `uv sync` + a couple of
  manual `uv pip install` additions — see section 6 if you need to rebuild one from scratch).
  `openpi` and `lingbot-vla-v2` are present as submodules but have no env / checkpoint yet
  (still blocked upstream — see section 7).
- **`ckpts/`**: `gr00t/`, `g05/`, `molmoact2/` — the actual downloaded/converted checkpoints for
  the 3 models that are ready. `ckpts/molmoact2_raw/` is the original 59GB FSDP-sharded
  checkpoint MolmoAct2 was converted from — safe to delete once you've confirmed
  `ckpts/molmoact2/` works (see section 6.3), it's not used at serve time.
- **`scripts/rc_serve_policy.py`**: one server script, `--policy {gr00t,g05,molmoact2}` picks
  the backend. Every backend answers the identical json_numpy `GET`/`POST /act` contract.
- **`scripts/eval_yam_joint_policy.py`**: the robot-side client (14-D joint state/action, no
  IK) — run this in your normal robot lerobot env, not any of the model venvs above.

## 1. Start a policy server

Run **one at a time** (see section 5 for why). Each command uses that model's own `.venv`:

```bash
# GR00T N1.7  (~6.2 GB VRAM, action chunk 16)
cd /home/rc1/lerobot
third_party/Isaac-GR00T/.venv/bin/python scripts/rc_serve_policy.py \
  --policy=gr00t --ckpt_path=ckpts/gr00t --port=8000

# G0.5  (~11.4 GB VRAM, action chunk 32)
third_party/GalaxeaVLA/.venv/bin/python scripts/rc_serve_policy.py \
  --policy=g05 --ckpt_path=ckpts/g05/checkpoints/step_8520.pt --port=8000

# MolmoAct2  (~10.8 GB VRAM, action chunk 30)
third_party/molmoact2/.venv/bin/python scripts/rc_serve_policy.py \
  --policy=molmoact2 --ckpt_path=ckpts/molmoact2 --port=8000
```

Wait for `[rc_serve_policy] serving '<policy>' on http://0.0.0.0:8000/act`, then check it from
another terminal:

```bash
wget -qO- http://localhost:8000/act
# -> {"action_dim":14,"camera_keys":["top","left","right"],"state_dim":14}
```

## 2. Hook up the real robot

- Power on both YAM arms, connect the 3 RealSense cameras (top `262522074294`, left wrist
  `260322271881`, right wrist `260322275072` — same serials/mounting as the Dustpan
  recordings).
- Set up the scene the way training saw it (dustpan + whatever's being cleaned).
- The robot-side env is a plain `uv`-created venv (`/home/rc1/lerobot/.venv`, no `pip` binary —
  use `uv pip install <pkg>` if anything's missing there, not `pip`/`python -m pip`).

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
- `--robot.record_eef_pose=false` is **required** — without it, `--record_dataset=true` will
  crash the first time an episode tries to write a frame (`KeyError: 'left_eef.x'`). See
  section 6.4 for why.
- `--task` must be exactly `"Clean the table."` — the only instruction string every checkpoint
  was fine-tuned on. Do not add "using the dust pan" or similar.
- `--actions_per_chunk`: roughly half the model's own chunk, or a fixed value (8) across all
  models for a fair comparison. GR00T chunk=16 → 8; MolmoAct2 chunk=30 → 15; G0.5 chunk=32 →
  16 (or 8, to match the others).

## 4. Session controls (stdin, type + Enter)

| Key | Meaning |
|---|---|
| `b` | Home the arms, begin episode 1 |
| `s` / `f` | While an episode is running: mark it success / failure (homes and holds) |
| `n` | After resetting the scene: start the next episode (homes first) |
| `r` | Re-home on demand |
| `z` | Quit / finish the session |

Output lands in `eval_logs/<timestamp>/`: `videos/episode_NNN_<outcome>[_top].mp4`,
`log.jsonl`, `summary.txt`. If `--record_dataset=true`, the raw (unencoded) dataset is at
`eval_logs/<timestamp>/dataset/` — encode it afterward:

```bash
python scripts/encode_pending_videos.py \
  --repo-id local/<timestamp>_clean_the_table \
  --root eval_logs/<timestamp>/dataset
```

## 5. Switching models / resource limits

**GPU (24 GB total)**: run **one policy server at a time**. Confirmed empirically, not just
by the spec sheet — GR00T (6.2GB) + G0.5 (11.4GB) + MolmoAct2 (10.8GB) together is ~28GB and
OOMs; any *two* of the three fit, but the intended real-eval workflow is one at a time anyway
(Ctrl+C the client, stop the server, start the next model's server, repeat).

**System RAM (30 GB total, tight)**: CPU-side work that loads a full model into host memory
(e.g. re-running the MolmoAct2 FSDP→HF conversion in section 6.3) can OOM-kill silently if a
GPU policy server is also running and holding a few GB of host RAM. Stop all `rc_serve_policy.py`
processes before doing anything like that:

```bash
ps aux | grep "[r]c_serve_policy" | awk '{print $2}' | xargs -r kill -9
```

## 6. Troubleshooting reference / rebuilding an env from scratch

Everything below was hit and fixed once already; `scripts/rc_serve_policy.py` already has the
code-side fixes baked in (they're not upstream repo patches — re-cloning a submodule fresh and
re-running `rc_serve_policy.py` against it should still work). Kept here in case an env needs
rebuilding or a checkpoint needs re-converting.

### 6.1 Rebuilding a model's `.venv`

```bash
cd third_party/<Isaac-GR00T|GalaxeaVLA|molmoact2>
uv venv --python <3.12|3.10|3.12> .venv
UV_HTTP_TIMEOUT=300 uv sync            # add --frozen for GalaxeaVLA (its lock has an
                                        # unrelated aarch64-only resolution failure otherwise)
uv pip install flask json_numpy        # rc_serve_policy.py's server deps, not in any
                                        # upstream pyproject.toml
```

GalaxeaVLA-specific build quirks:
- `configs/vla/robotwin/robotwin_dist_muon.yaml`'s LFS-tracked wheel
  (`scripts/deployment/dgpu/wheels/torchcodec-*.whl`) needs `git lfs pull` first if `uv sync`
  fails with "Invalid zip file structure" — a portable `git-lfs` binary works fine without
  root: `wget` the release tarball, put `git-lfs` on `PATH`, `git lfs install --local && git
  lfs pull`.
- `egl-probe`'s build needs `cmake` (`uv pip install cmake` into the venv, put `.venv/bin` on
  `PATH` for the sync) and `CMAKE_POLICY_VERSION_MINIMUM=3.5` (its `CMakeLists.txt` predates
  modern CMake's policy versioning).
- `flash-attn-4`'s CUTLASS JIT backend does not support this GPU's compute capability (RTX
  5090 / Blackwell, sm_120) — `uv pip uninstall flash-attn-4 nvidia-cutlass-dsl` so it falls
  back to the repo's own SDPA attention path (already has a clean fallback, just needs the
  import to fail).

### 6.2 G0.5-specific checkpoint layout

`load_g05` needs the checkpoint laid out as an actual Hydra run dir, not just the 3 files HF
ships:

```
ckpts/g05/
├── .hydra/config.yaml          # from the training run dir, NOT on HF -- scp from crane
├── dataset_stats.json
├── action_tokenizer.pt
├── hf_processor -> qwen3_5_2b_base_processor      # symlink
├── qwen3_5_2b_base_processor/  # from OpenGalaxea/G05 on HF (~22MB), shared across all G0.5 ckpts
└── checkpoints/
    └── step_8520.pt            # the actual weights (renamed from HF's model.pt)
```

If re-fetching: `.hydra/` comes from
`crane:.../g05-dustpan-crane6-gbs32/.hydra`;
`qwen3_5_2b_base_processor/` comes from `huggingface_hub.snapshot_download("OpenGalaxea/G05",
allow_patterns=["qwen3_5_2b_base_processor/*"])`.

Three bugs fixed in `rc_serve_policy.py`'s `_load_g05_config_from_run_dir`/`load_g05` that are
easy to reintroduce if this loader is ever rewritten:
1. `model.tokenizer` and `model.model_arch.AT_CONFIG` in the saved `.hydra/config.yaml` are
   OmegaConf interpolations (`${tokenizer}`, `${model.tokenizer.vq_config}`), not literal
   dicts. Upstream's `_apply_action_tokenizer_sidecar` writes through them with
   `OmegaConf.update(..., merge=False)`, which silently drops the interpolation target's other
   keys. Force-resolve both to concrete dicts *before* that sidecar runs.
2. The saved config's `oc.load:configs/data/parts_meta/r1lite.yaml`-style resolvers are
   relative to GalaxeaVLA's own repo root — `chdir` into it for the load window.
3. `_apply_hf_processor_sidecar` only redirects `hf_processor_path` to a local
   `run_dir/hf_processor` sidecar when `pretrained_model_path` is already non-null; ours is
   `null`, so it's a no-op — `load_g05` checks for `run_dir/hf_processor` and points
   `hf_processor_path` at it directly instead.

### 6.3 Re-converting a MolmoAct2 checkpoint (FSDP → HF format)

Training checkpoints land as raw FSDP-sharded `torch.distributed.checkpoint` dumps
(`.distcp` shards + `model_and_optim/`), not directly loadable. To convert a new one:

```bash
# 1. One-time env (py3.11, separate from the root serving env):
cd third_party/molmoact2/experiments
uv venv --python 3.11 .venv
UV_HTTP_TIMEOUT=300 uv sync
uv pip install torchmetrics   # missing from experiments/pyproject.toml

# 2. rsync the checkpoint (large -- ~60GB for a ~4-5B param model incl. optimizer state;
#    model + optimizer are interleaved per .distcp file, no cheap way to fetch model-only):
rsync -ah --info=progress2 \
  crane:<run_dir>/config.yaml \
  crane:<run_dir>/step<N> \
  ckpts/molmoact2_raw/

# 3. Stop any running rc_serve_policy.py servers first (RAM headroom -- see section 5),
#    then convert (point straight at the step dir, NOT the parent run dir -- the script
#    resolves the step dir for the config load but not for the actual weight load, so passing
#    the parent throws FileNotFoundError: '.../model_and_optim' is not a distributed
#    checkpoint folder):
cd /home/rc1/lerobot/third_party/molmoact2
experiments/.venv/bin/python -m experiments.olmo.hf_model.convert_molmoact2_to_hf \
  /home/rc1/lerobot/ckpts/molmoact2_raw/step<N> \
  /home/rc1/lerobot/ckpts/molmoact2 \
  --attn_implementation sdpa
```

Takes a couple minutes once running in isolation (RAM, not GPU, is the constraint here).
Confirm the norm tag afterward (`python -c "import json;
print(list(json.load(open('ckpts/molmoact2/norm_stats.json'))['metadata_by_tag']))"` — should
print `['yam_dustpan']`; `load_molmoact2` in `rc_serve_policy.py` hardcodes that tag).

### 6.4 `eval_yam_joint_policy.py` gotchas

- `json_numpy.patch()` must run **after** the first `robot.get_observation()` call, not
  before. `BiYamFollower.get_observation()` does its own internal FK for EEF-pose logging
  fields (independent of this client's own joint-only control path), which lazily imports
  `scipy` → `numpy.testing`, and `numpy.testing` does an unrelated `json.loads()` at import
  time that crashes under the patched decoder hook if json_numpy has already patched `json`.
- `--robot.record_eef_pose=false` is required with `--record_dataset=true` (see section 3) --
  without it, `BiYamFollower`'s native `action_features` includes `*_eef.*`/`*_eef_delta.*`
  columns that this joint-only client never populates.
- The robot-side venv (`/home/rc1/lerobot/.venv`) is `uv`-created and has no `pip` binary --
  use `uv pip install <pkg>`, not `pip install` or `python -m pip install` (both silently
  install into the wrong environment / fail with "No module named pip").

## 7. Not ready yet

- **π0.5 (openpi)**: submodule present, `pi05_yam_dustpan` `TrainConfig` already pasted into
  `third_party/openpi/src/openpi/training/config.py`, but no env or checkpoint set up yet.
- **LingBot-VLA v2**: submodule present, its `yam_dustpan` robot config + norm stats already
  placed at `third_party/lingbot-vla-v2/configs/robot_configs/` /
  `third_party/lingbot-vla-v2/assets/norm_stats/`, but no env or checkpoint yet (also the
  heaviest env to build — needs `flash-attn`/`conda`, per `EVAL_LOCAL_ROBOCOLOSSEUM.md`
  section 4).

Check `EVAL_LOCAL_ROBOCOLOSSEUM.md` section 6 for current checkpoint-readiness status before
starting either.
