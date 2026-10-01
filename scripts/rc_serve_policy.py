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
change: only `--policy` and `--task` (or `--ckpt_path`) change between runs.

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

`--ckpt_path` is either a local directory or an HF reference
`hf://<org>/<repo>[/<subfolder>][@<revision>]` (a `https://huggingface.co/<org>/<repo>/tree/<rev>/<subfolder>`
URL works too). HF references are fetched with `snapshot_download` into the standard HF cache
(`$HF_HOME`, default `~/.cache/huggingface`) and loaded from there -- every training-time config a
backend needs ships inside that checkpoint folder, so nothing is read from or written to the
submodules.

Usage (run ONE model at a time; stop the previous server before starting the next so its VRAM
is freed):

```shell
# <model>/<task> from --hf_repo (default RoboColosseum/BimanualYAM-models)
python scripts/rc_serve_policy.py --policy=gr00t --task=dustpan
python scripts/rc_serve_policy.py --policy=g05 --task=dustpan --hf_repo=RoboColosseum/BimanualYAM-models

# any other checkpoint
python scripts/rc_serve_policy.py --policy=molmoact2 --ckpt_path=hf://allenai/MolmoAct2-BimanualYAM --norm_tag=yam_dual_molmoact2
```

Then point the client at it: `--server_url=http://localhost:8000/act`.
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"
# Keeps backend imports from writing __pycache__/ into the submodules (not all of them ignore it).
sys.pycache_prefix = str(Path.home() / ".cache" / "rc_serve_policy" / "pycache")

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


_HF_URL_PREFIXES = ("hf://", "https://huggingface.co/", "http://huggingface.co/")


def _hf_repo_files(repo_id: str, allow_patterns: list[str] | None = None, revision: str | None = None) -> Path:
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=allow_patterns))


def _hf_snapshot(repo_id: str, subfolder: str = "", revision: str | None = None) -> Path:
    path = _hf_repo_files(repo_id, [f"{subfolder}/*"] if subfolder else None, revision) / subfolder
    if not path.is_dir():
        raise FileNotFoundError(f"{subfolder!r} not found in {repo_id}@{revision or 'main'}")
    return path


def resolve_ckpt(ref: str) -> Path:
    """Local dir -> itself. HF reference -> its snapshot dir inside the HF cache (see module docstring)."""
    local = Path(ref).expanduser()
    if local.exists():
        return local.absolute()

    spec = ref
    for prefix in _HF_URL_PREFIXES:
        if spec.startswith(prefix):
            spec = spec[len(prefix) :]
            break
    spec, _, revision = spec.partition("@")
    parts = [p for p in spec.split("/") if p]
    if len(parts) < 2:
        raise FileNotFoundError(f"{ref!r} is neither an existing local path nor an HF reference")
    repo_id, rest = "/".join(parts[:2]), parts[2:]
    if len(rest) >= 2 and rest[0] in ("tree", "blob"):
        revision = revision or rest[1]
        rest = rest[2:]
    return _hf_snapshot(repo_id, "/".join(rest), revision or None)


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
    config registered for the YAM robot's 14-D joint state/action layout. The checkpoint folder
    ships the `yam_config.py` it was trained with, which registers it as a module-level side
    effect on import -- so putting the checkpoint dir on `sys.path` and importing it is enough.

    State/action are a 4-part dict (`PARTS = ["left_arm", "left_gripper", "right_arm",
    "right_gripper"]` in `yam_config.py`, each a sub-range of the flat 14-D vector), cameras
    are plain `top`/`left`/`right`, language key is `observation["language"]["annotation.human.task_description"]`
    (section 7). The arm parts are `ActionRepresentation.RELATIVE` -- GR00T's own processor
    converts relative arm actions back to absolute, so this loader must NOT do that itself
    (matches section 4: "converted back to absolute by the processor")."""
    repo_dir = THIRD_PARTY / "Isaac-GR00T"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir, Path(ckpt_path))

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


def _load_g05_config_from_run_dir(run_dir: Path, ckpt_path: str, hf_processor: Path):
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

    import tempfile

    cfg.run_dir = str(run_dir)
    # Never inside run_dir: it may be a read-only HF cache snapshot.
    cfg.output_dir = str(Path(tempfile.gettempdir()) / "rc_serve_policy" / f"eval_{Path(run_dir).name}")
    cfg.exp_name = Path(run_dir).name
    cfg.logger.task = "eval"
    cfg.logger.experiment_name = f"eval_{Path(run_dir).name}"
    cfg.logger.mode = "disabled"
    cfg.ckpt_path = str(Path(ckpt_path).resolve())

    _apply_hf_processor_sidecar(cfg, run_dir)
    # --- fix (not in upstream `load_config_from_run_dir`) ---
    # `_apply_hf_processor_sidecar` is a no-op when `pretrained_model_path` is null (ours),
    # leaving the cwd-relative training-time `hf_processor_path`, which doesn't exist here.
    cfg.model.model_arch.hf_processor_path = str(hf_processor)
    _apply_action_tokenizer_sidecar(cfg, run_dir)

    _register_hydra_builtin_resolvers()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_g05(ckpt_path: str, device: str) -> PolicyBackend:
    """Mirrors `scripts/serve_policy.py::setup()` (config resolved from the checkpoint's own
    run dir via Hydra's saved `.hydra/config.yaml`) but skips its websocket/msgpack transport
    -- this script's `serve()` is the transport, this loader only builds the in-process
    `PolicyInferencer`. `ckpt_path` is the checkpoint folder (the training run dir layout):
    `model.pt`, `.hydra/config.yaml` (the fully composed training config), `dataset_stats.json`
    and `action_tokenizer.pt` side by side. The Qwen3.5 processor is shared across all G0.5
    checkpoints and comes from `OpenGalaxea/G05` on HF unless the folder has its own
    `hf_processor/`.

    Confirmed keys (verified against the checkpoint's `data_yam_dustpan.yaml`):
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

    # `.absolute()`, not `.resolve()`: in an HF cache snapshot every file is a symlink into the
    # blob store, so resolving would lose the sibling `.hydra/` that `find_run_dir` looks for.
    # Must happen before the chdir below.
    ckpt = Path(ckpt_path).absolute()
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    ckpt_path = str(ckpt)
    run_dir = find_run_dir(ckpt_path)
    hf_processor = run_dir / "hf_processor"
    if not hf_processor.exists():
        hf_processor = _hf_snapshot("OpenGalaxea/G05", "qwen3_5_2b_base_processor")

    # The saved config's `oc.load:configs/data/parts_meta/r1lite.yaml`-style resolvers use
    # paths relative to the GalaxeaVLA repo root (they assume its own scripts are run with
    # that as cwd) -- chdir for the config-load/model/processor construction window only, so
    # they resolve correctly regardless of what this server was launched from.
    import os

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        cfg = _load_g05_config_from_run_dir(run_dir, ckpt_path, hf_processor)

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


def _molmoact2_bf16_overlay(ckpt_dir: Path) -> Path:
    """Source patches from `examples/yam/host_server_yam.py::_patch_modeling_for_bf16`, applied
    to the checkpoint's exported `modeling_molmoact2.py` so bf16 inference works (upstream
    `predict_action` hardcodes `dtype=torch.float32` for a generator tensor, and casts actions
    back to numpy via a bf16-unsafe path).

    The checkpoint dir is never modified (an HF cache snapshot's files are symlinks into shared
    blobs). Instead this returns an overlay dir under `~/.cache/rc_serve_policy/` that symlinks
    every checkpoint file except a patched copy of `modeling_molmoact2.py`."""
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
    modeling = ckpt_dir / "modeling_molmoact2.py"
    src = modeling.read_text(encoding="utf-8")
    new_src = src
    for needle, replacement, marker in patches:
        if marker in new_src or needle not in new_src:
            continue
        new_src = new_src.replace(needle, replacement, 1)
    if new_src == src:
        return ckpt_dir

    key = hashlib.sha1(f"{ckpt_dir}\n{new_src}".encode()).hexdigest()[:16]
    overlay = Path.home() / ".cache" / "rc_serve_policy" / "molmoact2" / key
    overlay.mkdir(parents=True, exist_ok=True)
    for entry in ckpt_dir.iterdir():
        link = overlay / entry.name
        if entry.name == modeling.name or link.is_symlink() or link.exists():
            continue
        link.symlink_to(entry.absolute())
    (overlay / modeling.name).write_text(new_src, encoding="utf-8")
    return overlay


def load_molmoact2(ckpt_path: str, device: str, norm_tag: str | None = None) -> PolicyBackend:
    """Mirrors `examples/yam/host_server_yam.py::Policy` (that example is already almost
    exactly this script's contract: 14-D joint state/action, `top`/`left`/`right` cameras).

    `ckpt_path` is a resolved local dir (see `resolve_ckpt`): our Dustpan fine-tune (norm tag
    `yam_dustpan`, the default) or the official `allenai/MolmoAct2-BimanualYAM` (norm tag
    `yam_dual_molmoact2`, general YAM data, not Dustpan-specific). `predict_action` reads
    `norm_stats.json` relative to `config._name_or_path`, so loading by bare repo id would crash
    at inference time -- hence always a local dir."""
    repo_dir = THIRD_PARTY / "molmoact2"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt_path = str(_molmoact2_bf16_overlay(Path(ckpt_path)))

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
            norm_tag=norm_tag or _MOLMOACT2_NORM_TAG,
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
    """The checkpoint folder ships `train_config.py` (the exact `TrainConfig` it was trained
    with, exposed as `CONFIG`), loaded straight from the folder -- openpi itself is unmodified.

    The config's `RepackTransform` (LeRobot keys -> `cam_*`) only runs in training, so this
    loader sends the Aloha-style `cam_high`/`cam_left_wrist`/`cam_right_wrist` dict
    `AlohaInputs` consumes. `adapt_to_pi=False`, so raw follower joint units are sent as-is."""
    repo_dir = THIRD_PARTY / "openpi"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir / "src")

    from openpi.policies import policy_config

    spec = importlib.util.spec_from_file_location("rc_pi05_train_config", Path(ckpt_path) / "train_config.py")
    train_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_config_module)
    policy = policy_config.create_trained_policy(train_config_module.CONFIG, ckpt_path, pytorch_device=device)

    def infer(images, instruction, state, num_steps):
        # `create_trained_policy` doesn't apply the config's (training-only) repack transform, so
        # send its output keys directly; `AlohaInputs` wants CHW images.
        obs = {
            "images": {
                "cam_high": np.ascontiguousarray(images["top"].transpose(2, 0, 1)),
                "cam_left_wrist": np.ascontiguousarray(images["left"].transpose(2, 0, 1)),
                "cam_right_wrist": np.ascontiguousarray(images["right"].transpose(2, 0, 1)),
            },
            "state": state.astype(np.float32),
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


_LINGBOT_BASE = "robbyant/lingbot-vla-v2-6b"
_LINGBOT_QWEN = "Qwen/Qwen3-VL-4B-Instruct"
_LINGBOT_MOGE = "Ruicheng/moge-2-vitb-normal"


def _lingbot_cli_overlay(ckpt_dir: Path) -> Path:
    """Upstream `LingbotVLAv2Server.load_vla` reads the training CLI config from
    `<ckpt>/../../../lingbotvla_cli.yaml`, and the checkpoint's copy hardcodes training-machine
    paths for the base model, Qwen3-VL and the depth/video alignment modules. Writes
    `<overlay>/lingbotvla_cli.yaml` with those paths pointed at HF cache snapshots, links
    `<overlay>/_/_/ckpt -> ckpt_dir`, and returns that `ckpt` path for `path_to_pi_model`.
    Only config/tokenizer/aux files are fetched from the base repos -- the fine-tuned weights
    all come from `ckpt_dir`."""
    import yaml

    base = _hf_repo_files(_LINGBOT_BASE, ["*.json", "depth/*", "dino_video/*"])
    qwen = _hf_repo_files(_LINGBOT_QWEN, ["*.json", "*.txt", "*.jinja"])
    moge = _hf_repo_files(_LINGBOT_MOGE, ["*.json", "*.pt"])

    cli = yaml.safe_load((ckpt_dir / "lingbotvla_cli.yaml").read_text())
    cli["model"]["model_path"] = str(base)
    cli["model"]["config_path"] = str(base)
    cli["model"]["tokenizer_path"] = str(qwen)
    align = cli["train"]["align_params"]
    align["depth"]["moge_path"] = str(moge / "model.pt")
    align["depth"]["morgbd_path"] = str(base / "depth" / "model.pt")
    align["video"]["ckpt_path"] = str(base / "dino_video" / "teacher_step_10000.pth")
    align["video"]["config_path"] = str(base / "dino_video" / "config.yaml")
    rendered = yaml.safe_dump(cli, sort_keys=False)

    key = hashlib.sha1(f"{ckpt_dir}\n{rendered}".encode()).hexdigest()[:16]
    overlay = Path.home() / ".cache" / "rc_serve_policy" / "lingbot" / key
    link = overlay / "_" / "_" / "ckpt"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(ckpt_dir.absolute(), target_is_directory=True)
    (overlay / "lingbotvla_cli.yaml").write_text(rendered)
    return link


def load_lingbot(ckpt_path: str, device: str) -> PolicyBackend:
    """Wraps `deploy.lingbot_vla_v2_policy.LingbotVLAv2Server` in-process (skipping its own
    websocket+msgpack `serve_forever()`). Everything comes from the checkpoint folder:
    `lingbotvla_cli.yaml` (training config, paths rewritten by `_lingbot_cli_overlay`),
    `robot_config_yam_dustpan.yaml` (14-D -> 55-D mapping) and `norm_stats.json`.

    Upstream `reset()` reads the robot config from a cwd-relative `configs/robot_configs/`
    (inside the submodule), so this loader does the same setup with the checkpoint's own file.
    `FeatureTransform` takes the raw dataset keys (`observation.state`,
    `observation.images.{top,left,right}`, `task`) and its `unapply` turns the relative arm
    actions back into absolute joints, so the output is passed through unchanged."""
    repo_dir = THIRD_PARTY / "lingbot-vla-v2"
    _require_submodule(repo_dir)
    _add_to_path(repo_dir)

    import torch
    import yaml
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.data.vla_data.utils import FeatureTransform

    ckpt_dir = Path(ckpt_path)
    server = LingbotVLAv2Server(
        path_to_pi_model=str(_lingbot_cli_overlay(ckpt_dir)),
        robot_norm_path=str(ckpt_dir / "norm_stats.json"),
        chunk_ret=True,
        use_length=-1,
        use_bf16=True,
    )

    robot_config = str(ckpt_dir / "robot_config_yam_dustpan.yaml")
    server.robot_config = yaml.safe_load(Path(robot_config).read_text())
    feature_transform = FeatureTransform(
        robot_config,
        server.data_config,
        server.config,
        server.processor,
        chunk_size=server.config.chunk_size,
        norm_stats_path=server.robot_norm_path,
    )
    server.vla.feature_transform = feature_transform
    server.action_key = feature_transform.org_features["actions"]

    def infer(images, instruction, state, num_steps):
        observation = {
            "observation.images.top": images["top"],
            "observation.images.left": images["left"],
            "observation.images.right": images["right"],
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "task": instruction,
        }
        out = server.infer(observation)
        return np.asarray(out["action"], dtype=np.float32)

    return infer


DEFAULT_HF_REPO = "RoboColosseum/BimanualYAM-models"
HF_MODEL_FOLDERS = {
    "gr00t": "gr00t",
    "g05": "g05",
    "molmoact2": "molmoact2",
    "pi05": "pi05",
    "lingbot": "lingbot-vla-v2",
}

_BACKENDS = {
    "gr00t": load_gr00t,
    "g05": load_g05,
    "molmoact2": load_molmoact2,
    "pi05": load_pi05,
    "lingbot": load_lingbot,
}


def serve(
    policy_name: str, ckpt_path: str, host: str, port: int, device: str, norm_tag: str | None = None
) -> None:
    # Imported here, not at module top, so `--help` never needs Flask/json_numpy installed
    # and so json_numpy's process-wide monkeypatch of the stdlib `json` module (needed for
    # ndarray-in-JSON) happens only once the backend's own heavy imports (which may do their
    # own unrelated json parsing at import time) are already done -- same ordering rationale
    # as `eval_yam_http_policy.py`'s module docstring.
    import json_numpy
    from flask import Flask, jsonify, request

    print(f"[rc_serve_policy] resolving {ckpt_path!r} ...")
    ckpt_path = str(resolve_ckpt(ckpt_path))
    print(f"[rc_serve_policy] loading {policy_name!r} from {ckpt_path!r} on {device!r} ...")
    t0 = time.perf_counter()
    # `norm_tag` only means something to molmoact2 (its checkpoint can hold multiple norm
    # tags -- ours vs. the officially released BimanualYAM one); every other backend picks its
    # norm stats up from its own config/run dir and has no equivalent override.
    if policy_name == "molmoact2":
        infer = load_molmoact2(ckpt_path, device, norm_tag=norm_tag)
    else:
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
    parser.add_argument("--hf_repo", default=DEFAULT_HF_REPO, help="HF repo laid out as <model>/<task>/.")
    parser.add_argument("--task", help="Task folder inside --hf_repo, e.g. 'dustpan'.")
    parser.add_argument(
        "--ckpt_path",
        help="Override: local dir or hf://<org>/<repo>[/<subfolder>][@<rev>]. Takes precedence over --hf_repo/--task.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--norm_tag",
        default=None,
        help=(
            "molmoact2 only: override the checkpoint's norm tag (default 'yam_dustpan' for "
            "our fine-tune; pass 'yam_dual_molmoact2' when --ckpt_path points at a local "
            "snapshot of the official allenai/MolmoAct2-BimanualYAM checkpoint instead)."
        ),
    )
    args = parser.parse_args()
    if args.ckpt_path:
        ckpt_ref = args.ckpt_path
    elif args.task:
        ckpt_ref = f"hf://{args.hf_repo}/{HF_MODEL_FOLDERS[args.policy]}/{args.task}"
    else:
        parser.error("pass --task (checkpoint from --hf_repo) or --ckpt_path")
    serve(args.policy, ckpt_ref, args.host, args.port, args.device, norm_tag=args.norm_tag)


if __name__ == "__main__":
    main()
