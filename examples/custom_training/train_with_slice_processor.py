#!/usr/bin/env python

"""
Example script demonstrating how to train with custom slice processors.

This script shows how to use StateSliceProcessorStep, ActionSliceProcessorStep,
and ImageSelectProcessorStep to train a policy using only a subset of:
- State dimensions (e.g., only left arm joints)
- Action dimensions (e.g., only left arm joints)
- Camera views (e.g., only head and hand cameras)

Usage:
    python examples/custom_training/train_with_slice_processor.py \
        --dataset.repo_id=pick_bottle \
        --dataset.root=/path/to/your/dataset \
        --policy.type=act \
        --output_dir=outputs/train/pick_bottle_sliced

Or run directly with custom configuration:
    python examples/custom_training/train_with_slice_processor.py
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    StateSliceProcessorStep,
    ActionSliceProcessorStep,
    ImageSelectProcessorStep,
    StateActionSliceProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    OBS_STATE,
    ACTION,
)


def make_sliced_act_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    state_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
    image_keys: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Creates pre- and post-processing pipelines for ACT policy with dimension slicing.

    This function extends the standard ACT processor pipeline by adding:
    - StateSliceProcessorStep: to select specific state dimensions
    - ActionSliceProcessorStep: to select specific action dimensions
    - ImageSelectProcessorStep: to select specific camera views

    Args:
        config: The ACT policy configuration.
        dataset_stats: Dataset statistics for normalization.
        state_indices: List of state dimension indices to keep (e.g., [0,1,2,3,4,5,6] for left arm).
        action_indices: List of action dimension indices to keep.
        image_keys: List of image keys to keep (e.g., ["observation.images.head"]).

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines.
    """
    # Build input processing steps
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
    ]

    # Add image selection if specified
    if image_keys:
        input_steps.append(ImageSelectProcessorStep(image_keys=image_keys))

    # Add state slicing if specified
    if state_indices:
        input_steps.append(StateSliceProcessorStep(state_indices=state_indices))

    # Add action slicing if specified
    if action_indices:
        input_steps.append(ActionSliceProcessorStep(action_indices=action_indices))

    # Slice the dataset stats to match the sliced dimensions
    sliced_stats = slice_dataset_stats(
        dataset_stats,
        state_indices=state_indices,
        action_indices=action_indices,
    )

    # Continue with standard processing
    input_steps.extend([
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=sliced_stats,
            device=config.device,
        ),
    ])

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=sliced_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def slice_dataset_stats(
    stats: dict[str, dict[str, torch.Tensor]] | None,
    state_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """
    Slices dataset statistics to match the sliced state/action dimensions.

    Args:
        stats: Original dataset statistics.
        state_indices: Indices to keep for state.
        action_indices: Indices to keep for action.

    Returns:
        Sliced statistics dictionary.
    """
    if stats is None:
        return None

    sliced_stats = {}
    for key, stat_dict in stats.items():
        if key == OBS_STATE and state_indices:
            sliced_stats[key] = {
                stat_name: tensor[..., state_indices] if isinstance(tensor, torch.Tensor) else tensor
                for stat_name, tensor in stat_dict.items()
            }
        elif key == ACTION and action_indices:
            sliced_stats[key] = {
                stat_name: tensor[..., action_indices] if isinstance(tensor, torch.Tensor) else tensor
                for stat_name, tensor in stat_dict.items()
            }
        else:
            sliced_stats[key] = stat_dict

    return sliced_stats


def create_sliced_policy_features(
    original_features: dict[str, PolicyFeature],
    state_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
    image_keys: list[str] | None = None,
) -> dict[str, PolicyFeature]:
    """
    Creates policy features with sliced dimensions.

    Args:
        original_features: Original features from dataset.
        state_indices: Indices to keep for state.
        action_indices: Indices to keep for action.
        image_keys: Image keys to keep.

    Returns:
        Modified features dictionary.
    """
    new_features = {}

    for key, feature in original_features.items():
        # Filter images
        if key.startswith("observation.images.") or key == "observation.image":
            if image_keys is None or key in image_keys:
                new_features[key] = feature
            continue

        # Slice state
        if key == OBS_STATE and state_indices:
            new_shape = (len(state_indices),) + feature.shape[1:]
            new_features[key] = PolicyFeature(type=feature.type, shape=new_shape)
            continue

        # Slice action
        if key == ACTION and action_indices:
            new_shape = (len(action_indices),) + feature.shape[1:]
            new_features[key] = PolicyFeature(type=feature.type, shape=new_shape)
            continue

        # Keep other features unchanged
        new_features[key] = feature

    return new_features


@dataclass
class SlicedTrainConfig:
    """Configuration for training with sliced dimensions."""

    # Dataset configuration
    dataset_repo_id: str = "pick_bottle"
    dataset_root: str = "/mnt/data2/baohua/dataHub/lerobot/pick_bottle"

    # Slicing configuration
    state_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    action_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    image_keys: list[str] = field(default_factory=lambda: [
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ])

    # Training configuration
    policy_type: str = "act"
    batch_size: int = 8
    steps: int = 100000
    device: str = "cuda"
    output_dir: str = "outputs/train/pick_bottle_sliced"


def main():
    """Main training function with sliced dimensions."""
    logging.basicConfig(level=logging.INFO)

    # Configuration
    cfg = SlicedTrainConfig()

    logging.info(f"Training with sliced dimensions:")
    logging.info(f"  State indices: {cfg.state_indices}")
    logging.info(f"  Action indices: {cfg.action_indices}")
    logging.info(f"  Image keys: {cfg.image_keys}")

    # Load dataset metadata
    dataset_meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset_repo_id,
        root=Path(cfg.dataset_root),
    )

    # Get original features
    original_features = dataset_to_policy_features(dataset_meta.features)
    logging.info(f"Original features: {list(original_features.keys())}")

    # Create sliced features
    sliced_features = create_sliced_policy_features(
        original_features,
        state_indices=cfg.state_indices,
        action_indices=cfg.action_indices,
        image_keys=cfg.image_keys,
    )
    logging.info(f"Sliced features: {list(sliced_features.keys())}")

    # Log dimension changes
    if OBS_STATE in original_features and OBS_STATE in sliced_features:
        orig_dim = original_features[OBS_STATE].shape[0]
        new_dim = sliced_features[OBS_STATE].shape[0]
        logging.info(f"State dimension: {orig_dim} -> {new_dim}")

    if ACTION in original_features and ACTION in sliced_features:
        orig_dim = original_features[ACTION].shape[0]
        new_dim = sliced_features[ACTION].shape[0]
        logging.info(f"Action dimension: {orig_dim} -> {new_dim}")

    # Create policy config with sliced features
    input_features = {k: v for k, v in sliced_features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in sliced_features.items() if v.type == FeatureType.ACTION}

    policy_config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=cfg.device,
    )

    # Create policy
    policy_cls = get_policy_class(cfg.policy_type)
    policy = policy_cls(config=policy_config)
    policy.to(cfg.device)

    logging.info(f"Policy created with {sum(p.numel() for p in policy.parameters())} parameters")

    # Create sliced processors
    preprocessor, postprocessor = make_sliced_act_processors(
        config=policy_config,
        dataset_stats=dataset_meta.stats,
        state_indices=cfg.state_indices,
        action_indices=cfg.action_indices,
        image_keys=cfg.image_keys,
    )

    logging.info("Preprocessor steps:")
    for i, step in enumerate(preprocessor.steps):
        logging.info(f"  {i}: {step.__class__.__name__}")

    logging.info("Postprocessor steps:")
    for i, step in enumerate(postprocessor.steps):
        logging.info(f"  {i}: {step.__class__.__name__}")

    # Now you can use these in your training loop
    # For full training, integrate with lerobot_train.py or use the standard training pipeline

    logging.info("Setup complete! Ready for training.")
    logging.info(f"To train, you can now use the policy and processors in your training loop.")


if __name__ == "__main__":
    main()
