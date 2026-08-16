# Running MolmoAct2 on YAM Arms (Async Inference)

A minimal guide to run the **MolmoAct2** policy on **bimanual YAM arms** using
LeRobot's async inference.

It uses two machines:

- **Server** — a GPU box that runs the policy (`policy_server`) and returns action chunks.
- **Client** — the robot machine that talks to the YAM arms + RealSense cameras
  (`robot_client`), streams observations to the server, and executes the actions.

```
 Client (robot machine, py3.10)                 Server (GPU machine, py3.12)
 ┌───────────────────────────┐                  ┌───────────────────────────┐
 │ YAM follower arms (CAN)    │                  │                           │
 │ RealSense cameras          │ ── observations ─▶  MolmoAct2 policy (CUDA)  │
 │ robot_client               │ ◀── action chunks ─                         │
 └───────────────────────────┘      gRPC          └───────────────────────────┘
```

> Repo: [github.com/SuveenE/lerobot](https://github.com/SuveenE/lerobot)
>
> Branches: client = [`bimanual-yam-arms-support`](https://github.com/SuveenE/lerobot/tree/bimanual-yam-arms-support) (Python **3.10**),
> server = [`molmoact2-yam-async-inference-server`](https://github.com/SuveenE/lerobot/tree/molmoact2-yam-async-inference-server) (Python **3.12**, already updated to use `uv`).

---

## 0. Clone the repo (both machines)

```bash
git clone https://github.com/SuveenE/lerobot.git
cd lerobot
```

Both machines use `uv` — install it first if needed
([uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1. Server (GPU machine)

Branch `molmoact2-yam-async-inference-server` ships a `uv.lock`, so install is one command.
Requires a CUDA GPU (cu128 wheels, NVIDIA driver ≥ 570.86).

```bash
git checkout molmoact2-yam-async-inference-server && git pull

# Sync the locked env with the async + molmoact2 extras
uv sync --python 3.12 --locked --extra async --extra molmoact2

# Start the policy server (leave it running)
uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8000 \
  --fps=30
```

Note the server machine's IP address — the client connects to `<SERVER_IP>:8000`.

---

## 2. Client (robot machine, Python 3.10)

The client only needs the YAM hardware + camera deps — it never loads the policy,
so the `molmoact2` extra is **not** required here.

### Install (one-time)

You can use `uv` on the client too. This branch has no `uv.lock`, so just create a
3.10 virtualenv with `uv` and install in editable mode.

```bash
git checkout bimanual-yam-arms-support && git pull
git submodule update --init --recursive   # pulls the i2rt submodule

# Create + activate a Python 3.10 env with uv
uv venv --python 3.10
source .venv/bin/activate

# i2rt drives the YAM arms (local submodule), then LeRobot with yam + async + realsense
uv pip install -e ./i2rt
uv pip install -e ".[yam,async,intelrealsense]"
```

<details>
<summary>Prefer pip/conda instead of uv?</summary>

```bash
conda create -y -n lerobot-client python=3.10 && conda activate lerobot-client
pip install -e ./i2rt
pip install -e ".[yam,async,intelrealsense]"
```
</details>

### Step 2a — Start the YAM follower servers (terminal 1)

First reset the CAN interfaces (brings every `can*` interface down then up at
bitrate 1000000) — do this whenever the arms were just powered on or the bus is stuck:

```bash
bash i2rt/scripts/reset_all_can.sh
```

For eval we only need the follower arms (no leader/teaching handles):

```bash
python -m lerobot.scripts.setup_bi_yam_servers --eval
```

This launches the right (port `1234`) and left (port `1235`) follower arm servers.
Keep it running.

### Step 2b — Run the robot client (terminal 2)

Point `--server_address` at the GPU server's IP and port. Replace the RealSense
`serial_number_or_name` values with yours (`lerobot-find-cameras`).

```bash
python -m lerobot.async_inference.robot_client \
  --server_address https://untaken-eskimo-penholder.ngrok-free.dev/act \
  --robot.type bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras '{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 640, "height": 360, "fps": 30}
}' \
  --task "Pack everything into the box." \
  --policy_type molmoact2 \
  --pretrained_name_or_path="" \
  --policy_device cuda \
  --actions_per_chunk 15 \
  --chunk_size_threshold 0.0 \
  --aggregate_fn_name weighted_average \
  --debug_visualize_queue_size True \
  --policy_config_overrides '[
    "--checkpoint_path=allenai/MolmoAct2-BimanualYAM",
    "--norm_tag=yam_dual_molmoact2",
    "--inference_action_mode=continuous",
    "--normalize_gripper=false"
  ]'
```

### Find Serial Numbers for RealSense Cameras
```bash
rs-enumerate-devices | grep -i "serial"
```

---

## Key flags

| Flag | Meaning |
| --- | --- |
| `--server_address` | GPU server `IP:port` the client connects to. |
| `--pretrained_name_or_path=""` | Empty → **HF-original mode**: the MolmoAct2 weights on the Hub are in plain `transformers` format (no LeRobot `config.json`), so the server builds the policy config from scratch entirely out of `--policy_config_overrides`. Set this to a LeRobot-saved checkpoint instead to load its `config.json` directly. |
| `--policy_config_overrides` → `--checkpoint_path` | Where MolmoAct2 weights are loaded from (HF repo id). |

## Notes

- Client and server are **separate environments / machines**: client = Python 3.10,
  server = Python 3.12.
- The `molmoact2` extra lives only on the server branch; the client never instantiates the policy.
- **Data recording is skipped** in this guide. To record eval rollouts, add the
  `--dataset.*` flags — see the example at the top of
  `src/lerobot/async_inference/robot_client.py`