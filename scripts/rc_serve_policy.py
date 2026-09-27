#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single local policy server for all 5 RoboColosseum Dustpan VLA fine-tunes -- GR00T N1.7,
G0.5, MolmoAct2, pi0.5, LingBot-VLA v2 (see EVAL_LOCAL_ROBOCOLOSSEUM.md).

Exposes the SAME json_numpy REST `/act` contract for every model (`GET /act` health check,
`POST /act` observation -> action chunk), so `scripts/eval_yam_joint_policy.py` never needs to
change: only `--policy` and `--ckpt_path` change between runs.

    GET  /act  -> {"state_dim": 14, "action_dim": 14, "camera_keys": ["top", "left", "right"]}
    POST /act  body: {"top", "left", "right": HxWx3 uint8 RGB, "instruction": str,
                       "state": (14,) float32, "num_steps"?: int}
               resp: {"actions": (N, 14) float32 ABSOLUTE joints, "dt_ms": float}

Each of the 5 upstream repos lives under `third_party/<repo>/` as a pinned git submodule (see
`.gitmodules`) and ships its OWN inference API and serving protocol (ZMQ, websocket+msgpack,
or REST) with its own torch/JAX/transformers versions that conflict across repos -- see
`EVAL_LOCAL_ROBOCOLOSSEUM.md` section 0/4. So this script never imports more than one repo's
modules in a single process: `--policy` selects exactly one `_backend.*` loader below, which
lazily (1) inserts that repo's `third_party/<repo>/src` (or repo root) onto `sys.path` and (2)
imports it only at that point -- so simply importing this file, or running `--help`, never
requires any backend's dependencies to be installed. You still run this script inside THAT
model's own env (uv/conda per the table in section 4 of the guide); this file only removes the
need for 5 separate hand-rolled server scripts with 5 different client protocols.

Usage (run ONE model at a time; stop the previous server before starting the next so its VRAM
is freed -- see guide section 4):

```shell
# GR00T N1.7
python scripts/rc_serve_policy.py --policy=gr00t --ckpt_path=ckpts/gr00t --port=8000

# G0.5
python scripts/rc_serve_policy.py --policy=g05 --ckpt_path=ckpts/g05/model.pt --port=8000

# MolmoAct2
python scripts/rc_serve_policy.py --policy=molmoact2 --ckpt_path=ckpts/molmoact2 --port=8000

# pi0.5
python scripts/rc_serve_policy.py --policy=pi05 --ckpt_path=ckpts/pi05 --port=8000

# LingBot-VLA v2
python scripts/rc_serve_policy.py --policy=lingbot --ckpt_path=ckpts/lingbot-vla-v2 --port=8000
```

Then point the client at it: `--server_url=http://localhost:8000/act`.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"

CAMERA_KEYS = ("top", "left", "right")
STATE_DIM = 14
ACTION_DIM = 14
# The exact string every checkpoint was fine-tuned on -- see EVAL_LOCAL_ROBOCOLOSSEUM.md
# section 3. Passed as the default `instruction`/prompt if a request omits it.
DEFAULT_INSTRUCTION = "Clean the table."
# Joint state/action layout every backend must speak in, at the http boundary (see
# `eval_yam_joint_policy.py`'s `_state14_from_obs`): left_joint_0..5, left_gripper,
# right_joint_0..5, right_gripper.
JOINT_KEYS = [f"left_joint_{i}.pos" for i in range(6)] + ["left_gripper.pos"]
JOINT_KEYS += [f"right_joint_{i}.pos" for i in range(6)] + ["right_gripper.pos"]


class PolicyBackend(Protocol):
    """What every `_backend.*` loader hands back to `serve()`: a single `infer` closure that
    already has the checkpoint loaded, and speaks in exactly this script's plain-numpy
    contract -- 14-D joint state in, `(chunk, 14)` absolute joint actions out. All
    repo-specific normalization / image resizing / chunking happens inside this closure."""

    def __call__(
        self, images: dict[str, np.ndarray], instruction: str, state: np.ndarray, num_steps: int | None
    ) -> np.ndarray: ...


def _add_to_path(*paths: Path) -> None:
    for p in paths:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _require_submodule(repo_dir: Path) -> None:
    if not repo_dir.exists() or not any(repo_dir.iterdir()):
        raise RuntimeError(
            f"{repo_dir} is missing/empty -- run `git submodule update --init {repo_dir.relative_to(REPO_ROOT)}`."
        )


# ---------------------------------------------------------------------------
# GR00T N1.7 -- third_party/Isaac-GR00T
# ---------------------------------------------------------------------------


def load_gr00t(ckpt_path: str, device: str) -> PolicyBackend:
    """`Gr00tPolicy` (see `gr00t/policy/gr00t_policy.py`) needs a `NEW_EMBODIMENT` modality
    config registered for the YAM robot's 14-D joint state/action layout -- copy the two files
    named in EVAL_LOCAL_ROBOCOLOSSEUM.md section 0 to `runs/configs/gr00t/` before running this
    (`yam_config.py` registers the embodiment tag, `yam_modality.json` is the modality spec
    used at fine-tuning time). `yam_config.py` registers `EmbodimentTag.NEW_EMBODIMENT` as a
    module-level side effect on import (confirmed in section 7 of the guide), so importing it
    with `runs/configs/gr00t/` on `sys.path` is enough -- no explicit call needed.

    State/action are a 4-part dict (`PARTS = ["left_arm", "left_gripper", "right_arm",
    "right_gripper"]` in `yam_config.py`, each a sub-range of the flat 14-D vector), cameras
    are plain `top`/`left`/`right`, language key is `observation["language"]["annotation.human.task_description"]`
    (section 7). The arm parts are `ActionRepresentation.RELATIVE` -- GR00T's own processor
    converts relative arm actions back to absolute, so this loader must NOT do that itself
    (matches section 4: "converted back to absolute by the processor")."""
    repo_dir = THIRD_PARTY / "Isaac-GR00T"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir, REPO_ROOT / "runs" / "configs" / "gr00t")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    import yam_config  # noqa: F401  (side-effect import: registers EmbodimentTag.NEW_EMBODIMENT)

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=ckpt_path, device=device)

    def infer(images, instruction, state, num_steps):
        left_q, left_g = state[0:6], state[6:7]
        right_q, right_g = state[7:13], state[13:14]
        # `top` is trained/served at 640x360 (16:9) while both wrist cams are 640x480 (4:3) --
        # see EVAL_LOCAL_ROBOCOLOSSEUM.md section 3 ("GR00T server letterboxes it to 640x480
        # internally"). The eval-time transform (SmallestMaxSize(256) then a center crop)
        # resizes preserving aspect ratio, so mismatched input aspect ratios produce
        # mismatched output widths across cameras that then fail to `torch.stack` together.
        # Letterbox `top` to 480 height (black bars top/bottom) to match, exactly as training
        # saw it.
        top = images["top"]
        pad_total = 480 - top.shape[0]
        pad_top, pad_bottom = pad_total // 2, pad_total - pad_total // 2
        top_letterboxed = np.pad(top, ((pad_top, pad_bottom), (0, 0), (0, 0))) if pad_total > 0 else top
        video_images = {**images, "top": top_letterboxed}
        observation = {
            "video": {cam: video_images[cam][None, None] for cam in CAMERA_KEYS},  # (1, 1, H, W, 3)
            "state": {
                "left_arm": left_q[None, None].astype(np.float32),
                "left_gripper": left_g[None, None].astype(np.float32),
                "right_arm": right_q[None, None].astype(np.float32),
                "right_gripper": right_g[None, None].astype(np.float32),
            },
            "language": {"annotation.human.task_description": [[instruction]]},
        }
        action, _info = policy.get_action(observation)
        # Each action[part]: (1, T, part_dim) absolute -- processor already un-relativized the
        # arm parts. Re-flatten to this script's flat 14-D joint contract.
        chunk = action["left_arm"].shape[1]
        actions = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
        actions[:, 0:6] = action["left_arm"][0]
        actions[:, 6] = action["left_gripper"][0, :, 0]
        actions[:, 7:13] = action["right_arm"][0]
        actions[:, 13] = action["right_gripper"][0, :, 0]
        return actions

    return infer


# ---------------------------------------------------------------------------
# G0.5 -- third_party/GalaxeaVLA
# ---------------------------------------------------------------------------


def _load_g05_config_from_run_dir(run_dir: Path, ckpt_path: str):
    """Reimplements `g05.utils.checkpoint.ckpt_utils.load_config_from_run_dir`, with one fix
    applied to two fields: on this checkpoint's saved `.hydra/config.yaml`, both
    `model.tokenizer` (`${tokenizer}`, pointing at the root-level `tokenizer` block) and
    `model.model_arch.AT_CONFIG` (`${model.tokenizer.vq_config}`) are unresolved
    interpolations, not literal dicts. The upstream function's
    `_apply_action_tokenizer_sidecar` does `OmegaConf.update(cfg, "<dotted path ending in
    ...vq_config.ckpt_dir or ...AT_CONFIG.ckpt_dir>", ..., merge=False)` for both, and writing
    through a dotted path whose intermediate node is an unresolved interpolation makes
    OmegaConf materialize only the written leaf, silently dropping the interpolation target's
    other keys (`_target_`/`vqvae_type` included) -- which then breaks
    `model.model_arch.action_tokenizer = ${model.tokenizer._target_}` and
    `InputPreprocessor`'s `vq_config` (missing `vqvae_type`) respectively. Force-resolving both
    to concrete dicts before any sidecar patch walks into them (the one addition versus
    upstream, marked below) sidesteps both."""
    from omegaconf import OmegaConf
    from g05.utils.checkpoint.ckpt_utils import (
        _apply_action_tokenizer_sidecar,
        _apply_hf_processor_sidecar,
        _patch_g05_compat,
        _register_hydra_builtin_resolvers,
    )

    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    _patch_g05_compat(cfg)

    # --- fix (not in upstream `load_config_from_run_dir`): see docstring above. ---
    if OmegaConf.is_interpolation(cfg.model, "tokenizer"):
        cfg.model.tokenizer = OmegaConf.create(OmegaConf.to_container(cfg.model.tokenizer, resolve=True))
    if OmegaConf.is_interpolation(cfg.model.model_arch, "AT_CONFIG"):
        cfg.model.model_arch.AT_CONFIG = OmegaConf.create(
            OmegaConf.to_container(cfg.model.model_arch.AT_CONFIG, resolve=True)
        )

    ckpt_stem = Path(ckpt_path).stem
    cfg.run_dir = str(run_dir)
    cfg.output_dir = str(Path(run_dir) / f"eval_{ckpt_stem}")
    cfg.exp_name = Path(run_dir).name
    cfg.logger.task = "eval"
    cfg.logger.experiment_name = f"eval_{Path(run_dir).name}"
    cfg.logger.mode = "disabled"
    cfg.ckpt_path = str(Path(ckpt_path).resolve())

    _apply_hf_processor_sidecar(cfg, run_dir)
    # --- fix (not in upstream `load_config_from_run_dir`): see docstring above. ---
    # `_apply_hf_processor_sidecar` only redirects `hf_processor_path` to `run_dir/hf_processor`
    # when `pretrained_model_path` is set (its non-null value is what it falls back to when no
    # local sidecar exists); when `pretrained_model_path` is already None -- our case -- it
    # leaves `hf_processor_path` untouched at its original, cwd-relative training-time value
    # (e.g. "checkpoints/qwen3_5_2b_base_processor"), which doesn't exist here. If a local
    # `run_dir/hf_processor` sidecar exists (see EVAL_LOCAL_ROBOCOLOSSEUM.md section 8 -- a
    # symlink to the shared `qwen3_5_2b_base_processor/` from `OpenGalaxea/G05` on HF), point
    # `hf_processor_path` at it directly.
    local_hf = Path(run_dir) / "hf_processor"
    if local_hf.exists():
        cfg.model.model_arch.hf_processor_path = str(local_hf)
    _apply_action_tokenizer_sidecar(cfg, run_dir)

    _register_hydra_builtin_resolvers()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_g05(ckpt_path: str, device: str) -> PolicyBackend:
    """Mirrors `scripts/serve_policy.py::setup()` (config resolved from the checkpoint's own
    run dir via Hydra's saved `.hydra/config.yaml`) but skips its websocket/msgpack transport
    -- this script's `serve()` is the transport, this loader only builds the in-process
    `PolicyInferencer`. `ckpt_path` must therefore point at a checkpoint that still has its
    training run's `.hydra/config.yaml` findable by walking up to 5 parent dirs (see
    `find_run_dir`) -- e.g. `<run_dir>/checkpoints/step_8520.pt`, with `<run_dir>/.hydra/`
    and `<run_dir>/{dataset_stats.json,action_tokenizer.pt}` alongside it. The
    `configs/data/yam_dustpan.yaml`/`configs/task/yam_dustpan.yaml` dropped in from
    `serving/g05/` are what that saved config's Hydra `defaults:` resolve against at compose
    time, not read directly here.

    Confirmed keys (section 7 of the guide, verified against `configs/data/yam_dustpan.yaml`):
    state/action parts are `left_arm`(6)/`left_gripper`(1)/`right_arm`(6)/`right_gripper`(1);
    images are `head_rgb`(top, native 360x640 CHW)/`left_wrist_rgb`/`right_wrist_rgb`(both
    480x640 CHW) -- all resized to 224x224 internally by the processor. Arm actions are
    RelativeJointTransform (relative to current state); `PolicyInferencer.infer`'s
    postprocess step un-relativizes them using the saved processor config, so this loader
    does not redo that."""
    repo_dir = THIRD_PARTY / "GalaxeaVLA"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir)

    from g05.utils.checkpoint.ckpt_utils import find_run_dir
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.data.processor_utils import build_processors
    from g05.data_processor.processor.mixture_processor import MixtureProcessor
    from g05.data_processor.transforms.action_filter import BaseActionFilter
    from g05.models.g05.inferencer import PolicyInferencer, resolve_processor
    from g05.utils.common.pytorch_utils import dict_apply
    import torch

    register_default_resolvers()

    ckpt_path = str(Path(ckpt_path).resolve())  # before the chdir below, so it stays correct
    run_dir = find_run_dir(ckpt_path)

    # The saved config's `oc.load:configs/data/parts_meta/r1lite.yaml`-style resolvers use
    # paths relative to the GalaxeaVLA repo root (they assume its own scripts are run with
    # that as cwd) -- chdir for the config-load/model/processor construction window only, so
    # they resolve correctly regardless of what this server was launched from.
    import os

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        cfg = _load_g05_config_from_run_dir(run_dir, ckpt_path)

        model = load_model_from_checkpoint(
            cfg.model.model_arch, cfg.ckpt_path, device=device, extra_prefixes=["normalizer."], eval_mode=False
        )
        if cfg.model.get("model_weights_to_bf16", True):
            model = model.to(torch.bfloat16)
        model.apply_fp32_params()
        policy = model.eval()
        if hasattr(policy, "action_tokenizer"):
            policy.action_tokenizer.to(device)

        stats_path = Path(cfg.run_dir) / "dataset_stats.json"
        dataset_stats = load_dataset_stats_from_json(stats_path)
        processor = build_processors(cfg)
        processor.set_normalizer_from_stats(dataset_stats)
        processor.eval()

        def _neutralize_action_filter(p):
            safe_filter = BaseActionFilter()
            safe_filter.set_shape_meta(p.shape_meta)
            p.action_filter = safe_filter

        if isinstance(processor, MixtureProcessor):
            for emb_name in processor.processors:
                _neutralize_action_filter(processor.processors[emb_name])
        else:
            _neutralize_action_filter(processor)

        action_horizon = int(cfg.data.action_size)
        if isinstance(processor, MixtureProcessor):
            for sub_processor in processor.processors.values():
                sub_processor.action_horizon = action_horizon
        else:
            processor.action_horizon = action_horizon
    finally:
        os.chdir(prev_cwd)

    inferencer = PolicyInferencer(policy, processor, device)

    def infer(images, instruction, state, num_steps):
        left_q, left_g = state[0:6], state[6]
        right_q, right_g = state[7:13], state[13]
        raw_obs = {
            "images": {
                # HWC uint8 (this script's convention) -> CHW uint8 (this repo's convention).
                "head_rgb": np.ascontiguousarray(images["top"].transpose(2, 0, 1)),
                "left_wrist_rgb": np.ascontiguousarray(images["left"].transpose(2, 0, 1)),
                "right_wrist_rgb": np.ascontiguousarray(images["right"].transpose(2, 0, 1)),
            },
            "state": {
                "left_arm": left_q.astype(np.float32),
                "left_gripper": np.array([left_g], dtype=np.float32),
                "right_arm": right_q.astype(np.float32),
                "right_gripper": np.array([right_g], dtype=np.float32),
            },
            "task": instruction,
        }
        # Inlined from `scripts/serve_policy.py::build_obs_dict` (not imported directly --
        # that module lives at a bare `scripts/` path that would collide with this repo's own
        # `scripts/` on sys.path). Skips that function's `_validate_obs` call since this
        # loader already builds `raw_obs` to exactly match `processor.shape_meta`.
        p = resolve_processor(processor, raw_obs)
        num_obs_steps = p.num_obs_steps
        action_horizon_local = getattr(p, "action_horizon", num_obs_steps)
        expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
        expected_state_keys = {m["key"] for m in p.shape_meta["state"]}
        obs_dict = {
            "images": {
                k: torch.from_numpy(np.array(v, copy=True)).unsqueeze(0).expand(num_obs_steps, -1, -1, -1)
                for k, v in raw_obs["images"].items()
                if k in expected_img_keys
            },
            "state": {
                k: torch.from_numpy(np.array(v, copy=True)).unsqueeze(0).expand(num_obs_steps, -1).float()
                for k, v in raw_obs["state"].items()
                if k in expected_state_keys
            },
            "task": raw_obs["task"],
            "action": {m["key"]: torch.zeros(action_horizon_local, m["raw_shape"]) for m in p.shape_meta["action"]},
            "action_is_pad": torch.ones(action_horizon_local, dtype=torch.bool),
            "state_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
            "image_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
            "idx": 0,
        }

        out = inferencer.infer([obs_dict])[0]
        out.pop("_cot_text", None)
        out.pop("_absent_keys", None)
        # Each value: (batch=1, chunk, dim) torch tensor -> (chunk, dim) ndarray. `infer()` was
        # called with a single obs (batch size 1), so drop that leading dim.
        out = dict_apply(out, lambda x: x[0].numpy() if isinstance(x, torch.Tensor) else x)
        # out: dict of R1Lite-style part -> (chunk, dim) ndarrays, already absolute (see
        # docstring); re-flatten to the flat 14-D joint layout this script's contract uses.
        chunk = out["left_arm"].shape[0]
        actions = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
        actions[:, 0:6] = out["left_arm"]
        actions[:, 6] = out["left_gripper"][:, 0]
        actions[:, 7:13] = out["right_arm"]
        actions[:, 13] = out["right_gripper"][:, 0]
        return actions

    return infer


# ---------------------------------------------------------------------------
# MolmoAct2 -- third_party/molmoact2
# ---------------------------------------------------------------------------


_MOLMOACT2_NORM_TAG = "yam_dustpan"
_MOLMOACT2_DEFAULT_NUM_STEPS = 10


def _patch_molmoact2_modeling_for_bf16(local_dir: str) -> None:
    """Copied from `examples/yam/host_server_yam.py::_patch_modeling_for_bf16` -- two small,
    idempotent source patches to the checkpoint's own exported `modeling_molmoact2.py` so bf16
    inference works (upstream `predict_action` hardcodes `dtype=torch.float32` for a generator
    tensor, and casts actions back to numpy via a bf16-unsafe path). Safe to call on every
    load: each patch checks its own marker and no-ops if already applied."""
    import os

    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    path = os.path.join(local_dir, "modeling_molmoact2.py")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return
    new_src = src
    for needle, replacement, marker in patches:
        if marker in new_src or needle not in new_src:
            continue
        new_src = new_src.replace(needle, replacement, 1)
    if new_src != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)


def load_molmoact2(ckpt_path: str, device: str) -> PolicyBackend:
    """Mirrors `examples/yam/host_server_yam.py::Policy` (that example is already almost
    exactly this script's contract: 14-D joint state/action, `top`/`left`/`right` cameras) but
    loads straight from our local converted checkpoint (`ckpt_path`) instead of
    `snapshot_download`-ing `allenai/MolmoAct2-BimanualYAM`. Norm tag for this fine-tune is
    `yam_dustpan` (confirmed present as the sole tag in the checkpoint's own `norm_stats.json`
    after conversion -- see EVAL_LOCAL_ROBOCOLOSSEUM.md section 8)."""
    repo_dir = THIRD_PARTY / "molmoact2"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _patch_molmoact2_modeling_for_bf16(ckpt_path)

    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True, extra_special_tokens={})
    model = (
        AutoModelForImageTextToText.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    # Upstream `_move_inputs_to_device` only moves tensors, doesn't cast floats to the model
    # dtype -- with bf16 weights the processor's fp32 `pixel_values` then trips a dtype
    # mismatch. Same per-instance monkeypatch as the reference server.
    target_dtype = next(model.parameters()).dtype

    def _move_and_cast(inputs, dev, _target=target_dtype):
        out = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(dev)
                if value.is_floating_point() and value.dtype != _target:
                    value = value.to(_target)
            out[key] = value
        return out

    model._move_inputs_to_device = _move_and_cast

    def _to_pil(arr: np.ndarray) -> Image.Image:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @torch.inference_mode()
    def infer(images, instruction, state, num_steps):
        pil_images = [_to_pil(images[cam]) for cam in CAMERA_KEYS]  # order: top, left, right
        out = model.predict_action(
            processor=processor,
            images=pil_images,
            task=instruction,
            state=state.astype(np.float32),
            norm_tag=_MOLMOACT2_NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=num_steps or _MOLMOACT2_DEFAULT_NUM_STEPS,
            normalize_language=True,
            enable_cuda_graph=False,
        )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions

    return infer


# ---------------------------------------------------------------------------
# pi0.5 -- third_party/openpi
# ---------------------------------------------------------------------------


def load_pi05(ckpt_path: str, device: str) -> PolicyBackend:
    """Requires the `pi05_yam_dustpan` TrainConfig entry described in
    EVAL_LOCAL_ROBOCOLOSSEUM.md section 0 to already be present in this submodule's
    `src/openpi/training/config.py` (pull it from the HF `serving/` upload, or add it locally
    following the `pi05_droid`/`pi05_aloha` entries in that file).

    Confirmed (section 7): the config's `RepackTransform` runs FIRST and expects
    LeRobot-style external keys (`observation.images.top/left/right`, `observation.state`,
    `prompt`), converting them internally to the Aloha-style `cam_high`/`cam_left_wrist`/
    `cam_right_wrist` dict that `AlohaInputs` then consumes -- so this loader must send the
    external (left-hand-side) keys below, NOT the internal `cam_*` names. `adapt_to_pi=False`
    in the config, so no Aloha-space unit conversion is applied -- raw follower joint units
    are sent as-is."""
    repo_dir = THIRD_PARTY / "openpi"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir / "src")

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config("pi05_yam_dustpan")
    policy = policy_config.create_trained_policy(train_config, ckpt_path, pytorch_device=device)

    def infer(images, instruction, state, num_steps):
        obs = {
            "observation.images.top": images["top"],
            "observation.images.left": images["left"],
            "observation.images.right": images["right"],
            "observation.state": state.astype(np.float32),
            "prompt": instruction,
        }
        result = policy.infer(obs)
        # result["actions"]: (chunk, 14) absolute joint targets, per the pi05_yam_dustpan
        # data-transform config (no delta decoding needed -- trained on absolute joints).
        return np.asarray(result["actions"], dtype=np.float32)

    return infer


# ---------------------------------------------------------------------------
# LingBot-VLA v2 -- third_party/lingbot-vla-v2
# ---------------------------------------------------------------------------


def load_lingbot(ckpt_path: str, device: str) -> PolicyBackend:
    """Wraps `deploy.lingbot_vla_v2_policy.LingbotVLAv2Server` in-process (skipping its own
    websocket+msgpack `serve_forever()`), then calls `.reset(robo_name="yam_dustpan")` once to
    load the YAM robot config/norm stats from `configs/robot_configs/yam_dustpan.yaml` +
    `assets/norm_stats/yam_dustpan.json` before serving (confirmed present in section 7).

    Confirmed image keys: `observation.images.camera_top` / `camera_wrist_left` /
    `camera_wrist_right`, sourced from `top`/`left`/`right` respectively. Arm actions use
    `subtract_state: True, relative_type: joint` (relative to current state, same pattern as
    GR00T/G0.5) while the gripper action is absolute -- `server.infer`'s own `feature_transform
    .unapply` handles the un-relativize step, so this loader passes the raw model output
    through unchanged."""
    repo_dir = THIRD_PARTY / "lingbot-vla-v2"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir)

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(path_to_pi_model=ckpt_path, use_bf16=True)
    server.reset(robo_name="yam_dustpan")

    def infer(images, instruction, state, num_steps):
        left_q, left_g = state[0:6], state[6]
        right_q, right_g = state[7:13], state[13]
        observation = {
            "images": {
                "observation.images.camera_top": images["top"],
                "observation.images.camera_wrist_left": images["left"],
                "observation.images.camera_wrist_right": images["right"],
            },
            "state": {
                "left_arm": left_q.astype(np.float32),
                "left_gripper": float(left_g),
                "right_arm": right_q.astype(np.float32),
                "right_gripper": float(right_g),
            },
            "instruction": instruction,
        }
        out = server.infer(observation)
        chunk = out["left_arm"].shape[0]
        actions = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
        actions[:, 0:6] = out["left_arm"]
        actions[:, 6] = np.asarray(out["left_gripper"]).reshape(-1)
        actions[:, 7:13] = out["right_arm"]
        actions[:, 13] = np.asarray(out["right_gripper"]).reshape(-1)
        return actions

    return infer


_BACKENDS = {
    "gr00t": load_gr00t,
    "g05": load_g05,
    "molmoact2": load_molmoact2,
    "pi05": load_pi05,
    "lingbot": load_lingbot,
}


def serve(policy_name: str, ckpt_path: str, host: str, port: int, device: str) -> None:
    # Imported here, not at module top, so `--help` never needs Flask/json_numpy installed
    # and so json_numpy's process-wide monkeypatch of the stdlib `json` module (needed for
    # ndarray-in-JSON) happens only once the backend's own heavy imports (which may do their
    # own unrelated json parsing at import time) are already done -- same ordering rationale
    # as `eval_yam_http_policy.py`'s module docstring.
    import json_numpy
    from flask import Flask, jsonify, request

    print(f"[rc_serve_policy] loading {policy_name!r} from {ckpt_path!r} on {device!r} ...")
    t0 = time.perf_counter()
    infer = _BACKENDS[policy_name](ckpt_path, device)
    print(f"[rc_serve_policy] loaded in {time.perf_counter() - t0:.1f}s")

    json_numpy.patch()
    app = Flask(__name__)

    @app.route("/act", methods=["GET"])
    def health():
        return jsonify({"state_dim": STATE_DIM, "action_dim": ACTION_DIM, "camera_keys": list(CAMERA_KEYS)})

    @app.route("/act", methods=["POST"])
    def act():
        payload = request.get_json()
        try:
            images = {cam: np.asarray(payload[cam]) for cam in CAMERA_KEYS}
            state = np.asarray(payload["state"], dtype=np.float32)
            if state.shape != (STATE_DIM,):
                raise ValueError(f"expected state shape ({STATE_DIM},), got {state.shape}")
            instruction = payload.get("instruction", DEFAULT_INSTRUCTION)
            num_steps = payload.get("num_steps")

            t0 = time.perf_counter()
            actions = infer(images, instruction, state, num_steps)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"backend returned actions of shape {actions.shape}, expected (*, {ACTION_DIM})")
            return jsonify({"actions": actions, "dt_ms": dt_ms})
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": repr(e)})

    print(f"[rc_serve_policy] serving {policy_name!r} on http://{host}:{port}/act")
    app.run(host=host, port=port, threaded=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, choices=sorted(_BACKENDS), help="Which VLA fine-tune to serve.")
    parser.add_argument("--ckpt_path", required=True, help="Path to that model's downloaded checkpoint (see hf download in section 2 of the guide).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    serve(args.policy, args.ckpt_path, args.host, args.port, args.device)


if __name__ == "__main__":
    main()
