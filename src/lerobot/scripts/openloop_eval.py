#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Open-loop evaluation of a policy on a dataset.

This script evaluates a policy by comparing predicted actions against ground truth actions
from a dataset, without running in a simulation environment. It computes MSE and MAE metrics
and optionally generates trajectory plots.

Usage examples:

Evaluate a pretrained model on a dataset:
```
python -m lerobot.scripts.openloop_eval \
    --policy.path=lerobot/diffusion_pusht \
    --dataset.repo_id=lerobot/pusht \
    --eval.episode_ids 0 1 2 \
    --eval.action_horizon=16 \
    --eval.save_plot_dir=/tmp/openloop_eval
```

Evaluate a local checkpoint:
```
python -m lerobot.scripts.openloop_eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --dataset.repo_id=lerobot/pusht \
    --eval.episode_ids 0 \
    --eval.steps=200
```
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from pprint import pformat

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

logger = getLogger(__name__)


def plot_trajectory_results(
    state_across_time: np.ndarray | None,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    episode_id: int,
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_across_time: Array of state over time (optional, can be None)
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        episode_id: Episode ID
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed, skipping plot generation")
        return

    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))
    num_plots = len(indices_to_plot)

    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 3 * num_plots))

    if num_plots == 1:
        axes = [axes]

    fig.suptitle(f"Episode {episode_id} - Open-loop Evaluation", fontsize=14)

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # Plot state if available and dimensions match
        if state_across_time is not None and state_across_time.shape[1] > action_idx:
            ax.plot(state_across_time[:, action_idx], label="state", alpha=0.7)

        ax.plot(gt_action_across_time[:, action_idx], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, action_idx], label="pred action", linestyle="--", linewidth=2)

        # Mark inference points
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", markersize=6, label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", markersize=6)

        ax.set_title(f"Action Dimension {action_idx}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path, dpi=150)
    plt.close()
    logging.info(f"Saved plot to {save_plot_path}")


def evaluate_single_episode(
    policy: PreTrainedPolicy,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    episode_id: int,
    steps: int = 300,
    action_horizon: int = 16,
    save_plot_path: str | None = None,
    device: str = "cuda",
) -> tuple[float, float]:
    """
    Evaluate a single episode by comparing predicted actions to ground truth.

    Args:
        policy: The policy to evaluate
        dataset: The dataset containing ground truth
        preprocessor: Preprocessor pipeline for observations
        postprocessor: Postprocessor pipeline for actions
        episode_id: Episode index to evaluate
        steps: Maximum number of steps to evaluate
        action_horizon: Number of steps between policy inferences
        save_plot_path: Path to save the trajectory plot
        device: Device to run inference on

    Returns:
        Tuple of (MSE, MAE) for the episode
    """
    # Get episode boundaries
    ep_info = dataset.meta.episodes[episode_id]
    ep_start = ep_info["dataset_from_index"]
    ep_end = ep_info["dataset_to_index"]
    ep_length = ep_end - ep_start

    actual_steps = min(steps, ep_length)
    logging.info(f"Episode {episode_id}: using {actual_steps} steps (episode length: {ep_length})")

    pred_action_across_time = []
    gt_action_across_time = []
    state_across_time = []

    policy.reset()

    # Iterate through the episode with action_horizon steps
    for step_count in range(0, actual_steps, action_horizon):
        # Get the data point at this step
        data_idx = ep_start + step_count
        data_point = dataset[data_idx]

        # Collect ground truth actions for this chunk
        chunk_end = min(step_count + action_horizon, actual_steps)
        for j in range(step_count, chunk_end):
            gt_idx = ep_start + j
            gt_data = dataset[gt_idx]
            gt_action = gt_data["action"]
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.cpu().numpy()
            # Handle multi-step action format
            if gt_action.ndim > 1:
                gt_action = gt_action[0]  # Take first step if action has temporal dimension
            gt_action_across_time.append(gt_action)

            # Collect state if available
            if "observation.state" in gt_data:
                state = gt_data["observation.state"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                if state.ndim > 1:
                    state = state[-1]  # Take last observation if has temporal dimension
                state_across_time.append(state)

        # Prepare observation for policy
        obs = {}
        for key, value in data_point.items():
            if key.startswith("observation.") or key == "task":
                if isinstance(value, torch.Tensor):
                    # Add batch dimension
                    obs[key] = value.unsqueeze(0).to(device)
                elif isinstance(value, str):
                    obs[key] = value
                else:
                    obs[key] = torch.tensor(value).unsqueeze(0).to(device)

        # Apply preprocessor
        obs = preprocessor(obs)

        # Get action from policy
        with torch.inference_mode():
            action = policy.select_action(obs)

        # Apply postprocessor
        action = postprocessor(action)

        # Convert action to numpy
        if isinstance(action, dict):
            action_tensor = action.get("action", list(action.values())[0])
        else:
            action_tensor = action

        if isinstance(action_tensor, torch.Tensor):
            action_np = action_tensor.cpu().numpy()
        else:
            action_np = np.array(action_tensor)

        # Remove batch dimension if present
        if action_np.ndim == 3:
            action_np = action_np[0]  # (batch, horizon, dim) -> (horizon, dim)
        elif action_np.ndim == 1:
            action_np = action_np[np.newaxis, :]  # (dim,) -> (1, dim)

        # Append predicted actions for this chunk
        for j in range(chunk_end - step_count):
            if j < len(action_np):
                pred_action_across_time.append(action_np[j])
            else:
                # If action horizon is shorter than chunk, repeat last action
                pred_action_across_time.append(action_np[-1])

    # Convert to numpy arrays
    gt_action_across_time = np.array(gt_action_across_time)[:actual_steps]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]

    if len(state_across_time) > 0:
        state_across_time = np.array(state_across_time)[:actual_steps]
    else:
        state_across_time = None

    # Ensure shapes match
    min_len = min(len(gt_action_across_time), len(pred_action_across_time))
    gt_action_across_time = gt_action_across_time[:min_len]
    pred_action_across_time = pred_action_across_time[:min_len]

    if gt_action_across_time.shape != pred_action_across_time.shape:
        logging.warning(
            f"Shape mismatch: gt={gt_action_across_time.shape}, pred={pred_action_across_time.shape}"
        )
        # Try to align dimensions
        min_dim = min(gt_action_across_time.shape[-1], pred_action_across_time.shape[-1])
        gt_action_across_time = gt_action_across_time[..., :min_dim]
        pred_action_across_time = pred_action_across_time[..., :min_dim]

    # Calculate metrics
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

    logging.info(f"Episode {episode_id} - MSE: {mse:.6f}, MAE: {mae:.6f}")

    # Plot if requested
    if save_plot_path:
        plot_trajectory_results(
            state_across_time=state_across_time,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            episode_id=episode_id,
            action_horizon=action_horizon,
            save_plot_path=save_plot_path,
        )

    return mse, mae


@dataclass
class OpenLoopEvalConfig:
    """Configuration for open-loop evaluation."""

    episode_ids: list[int] = field(default_factory=lambda: [0])
    """List of episode IDs to evaluate."""

    steps: int = 300
    """Maximum number of steps to evaluate per episode."""

    action_horizon: int = 16
    """Number of steps between policy inferences."""

    save_plot_dir: str = "outputs/openloop_eval_plots"
    """Directory to save trajectory plots."""


@dataclass
class OpenLoopEvalPipelineConfig:
    """Pipeline configuration for open-loop evaluation."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""

    eval: OpenLoopEvalConfig = field(default_factory=OpenLoopEvalConfig)
    """Evaluation configuration."""

    policy: PreTrainedConfig | None = None
    """Policy configuration."""

    seed: int = 42
    """Random seed."""

    output_dir: str = "outputs/openloop_eval"
    """Output directory for results."""

    state_indices: list[int] | None = None
    """State indices to use (for slicing state dimensions)."""

    action_indices: list[int] | None = None
    """Action indices to use (for slicing action dimensions)."""

    image_keys: list[str] | None = None
    """Image keys to use (for selecting specific cameras)."""

    resize_size: tuple[int, int] | None = None
    """Resize size for images."""

    def __post_init__(self) -> None:
        # Parse policy path from CLI args
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            logger.warning(
                "No pretrained path was provided, policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def openloop_eval_main(cfg: OpenLoopEvalPipelineConfig):
    """Main function for open-loop evaluation."""
    logging.info(pformat(cfg))

    if cfg.policy is None:
        raise ValueError("A policy must be provided via --policy.path=<path>")

    # Check device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    # Load dataset metadata
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )

    # Apply feature slicing if indices are provided (to match training configuration)
    if cfg.state_indices is not None:
        state_feature = ds_meta.features.get("observation.state")
        if state_feature:
            original_dim = state_feature["shape"][0]
            sliced_dim = len(cfg.state_indices)
            logging.info(f"Slicing observation.state from {original_dim} to {sliced_dim} dimensions")
            ds_meta.info["features"]["observation.state"]["shape"] = [sliced_dim]
    
    if cfg.action_indices is not None:
        action_feature = ds_meta.features.get("action")
        if action_feature:
            original_dim = action_feature["shape"][0]
            sliced_dim = len(cfg.action_indices)
            logging.info(f"Slicing action from {original_dim} to {sliced_dim} dimensions")
            ds_meta.info["features"]["action"]["shape"] = [sliced_dim]
    
    if cfg.image_keys is not None:
        # Remove image keys not in the list
        all_camera_keys = ds_meta.camera_keys.copy()
        for key in all_camera_keys:
            if key not in cfg.image_keys:
                logging.info(f"Removing camera key: {key}")
                ds_meta.info["features"].pop(key, None)

    # Calculate delta_timestamps from policy config
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    # Load dataset
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )
    logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")

    # Create policy
    logging.info("Loading policy...")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )
    policy.eval()

    # Create preprocessor and postprocessor
    preprocessor_overrides = {
        "device_processor": {"device": str(cfg.policy.device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        state_indices=cfg.state_indices,
        action_indices=cfg.action_indices,
        image_keys=cfg.image_keys,
        resize_size=cfg.resize_size,
    )

    # Run evaluation
    all_mse = []
    all_mae = []

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for episode_id in cfg.eval.episode_ids:
            if episode_id >= dataset.num_episodes:
                logging.warning(
                    f"Episode ID {episode_id} is out of range (max: {dataset.num_episodes - 1}). Skipping."
                )
                continue

            save_plot_path = None
            if cfg.eval.save_plot_dir:
                save_plot_path = f"{cfg.eval.save_plot_dir}/episode_{episode_id}.png"

            mse, mae = evaluate_single_episode(
                policy=policy,
                dataset=dataset,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                episode_id=episode_id,
                steps=cfg.eval.steps,
                action_horizon=cfg.eval.action_horizon,
                save_plot_path=save_plot_path,
                device=str(cfg.policy.device),
            )

            all_mse.append(mse)
            all_mae.append(mae)

    # Report results
    if all_mse:
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae)
        logging.info("=" * 60)
        logging.info("Open-loop Evaluation Results:")
        logging.info(f"  Episodes evaluated: {len(all_mse)}")
        logging.info(f"  Average MSE: {avg_mse:.6f}")
        logging.info(f"  Average MAE: {avg_mae:.6f}")
        logging.info("=" * 60)

        # Per-episode results
        for i, episode_id in enumerate(cfg.eval.episode_ids):
            if i < len(all_mse):
                logging.info(f"  Episode {episode_id}: MSE={all_mse[i]:.6f}, MAE={all_mae[i]:.6f}")
    else:
        logging.warning("No valid episodes were evaluated.")

    logging.info("Open-loop evaluation complete.")


def main():
    init_logging()
    openloop_eval_main()


if __name__ == "__main__":
    main()
