#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    ImageCropResizeProcessorStep,
    ImageSelectProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    StateActionSliceProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _slice_dataset_stats(
    stats: dict[str, dict[str, torch.Tensor]] | None,
    state_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Slice dataset statistics to match the sliced state/action dimensions.

    Args:
        stats: Original dataset statistics.
        state_indices: Indices to keep for state.
        action_indices: Indices to keep for action.

    Returns:
        Sliced statistics dictionary, or original if no slicing needed.
    """
    if stats is None:
        return None

    if not state_indices and not action_indices:
        return stats

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


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    state_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
    image_keys: list[str] | None = None,
    resize_size: tuple[int, int] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.
        state_indices (list[int] | None): Optional list of state dimension indices to keep.
            If None, all state dimensions are used.
        action_indices (list[int] | None): Optional list of action dimension indices to keep.
            If None, all action dimensions are used.
        image_keys (list[str] | None): Optional list of image keys to keep.
            If None, all images are used.
        resize_size (tuple[int, int] | None): Optional (height, width) to resize all images to.
            If None, no resizing is done.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    # Slice dataset stats if indices are provided
    sliced_stats = _slice_dataset_stats(dataset_stats, state_indices, action_indices)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
    ]

    # Add image selection if specified
    if image_keys:
        input_steps.append(ImageSelectProcessorStep(image_keys=image_keys))

    # Add image resize if specified
    if resize_size:
        input_steps.append(ImageCropResizeProcessorStep(resize_size=resize_size))

    # Add state/action slicing if specified
    if state_indices or action_indices:
        input_steps.append(StateActionSliceProcessorStep(
            state_indices=state_indices or [],
            action_indices=action_indices or [],
        ))

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
            features=config.output_features, norm_map=config.normalization_mapping, stats=sliced_stats
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
