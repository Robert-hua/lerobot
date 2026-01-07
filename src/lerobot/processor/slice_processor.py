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

"""
Processor steps for slicing/selecting specific dimensions from state and action tensors.

This module provides processors that allow selecting specific indices from observation.state
and action tensors, enabling training with a subset of the full state/action space.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register("state_slice_processor")
class StateSliceProcessorStep(ProcessorStep):
    """
    Selects specific dimensions from observation.state tensor.

    This processor allows you to select a subset of state dimensions by specifying
    their indices. This is useful when you want to train a policy using only
    certain joints or sensors from a robot with many degrees of freedom.

    Attributes:
        state_indices: List of indices to select from observation.state.
                       If empty, all dimensions are kept (no slicing).
        state_key: The key for the state in observations. Defaults to "observation.state".

    Example:
        # Select only the first 7 dimensions (e.g., left arm joints)
        processor = StateSliceProcessorStep(state_indices=[0, 1, 2, 3, 4, 5, 6])
    """

    state_indices: list[int] = field(default_factory=list)
    state_key: str = OBS_STATE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Slices the state tensor to keep only the specified indices.

        Args:
            transition: The input transition containing observations.

        Returns:
            A new transition with the sliced state tensor.
        """
        if not self.state_indices:
            return transition

        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION)

        if obs is not None and self.state_key in obs:
            state = obs[self.state_key]
            if isinstance(state, torch.Tensor):
                # Handle both batched (B, D) and unbatched (D,) tensors
                indices = torch.tensor(self.state_indices, device=state.device)
                sliced_state = torch.index_select(state, dim=-1, index=indices)
                obs[self.state_key] = sliced_state
                new_transition[TransitionKey.OBSERVATION] = obs

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration for serialization."""
        return {
            "state_indices": self.state_indices,
            "state_key": self.state_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the state feature shape to reflect the slicing.

        Args:
            features: The input feature dictionary.

        Returns:
            Updated feature dictionary with the new state shape.
        """
        if not self.state_indices:
            return features

        obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
        if self.state_key in obs_features:
            original_feature = obs_features[self.state_key]
            new_shape = (len(self.state_indices),) + original_feature.shape[1:]
            obs_features[self.state_key] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
            features[PipelineFeatureType.OBSERVATION] = obs_features

        return features


@dataclass
@ProcessorStepRegistry.register("action_slice_processor")
class ActionSliceProcessorStep(ProcessorStep):
    """
    Selects specific dimensions from action tensor.

    This processor allows you to select a subset of action dimensions by specifying
    their indices. This is useful when you want to train a policy that only controls
    certain joints or actuators.

    Attributes:
        action_indices: List of indices to select from action tensor.
                        If empty, all dimensions are kept (no slicing).

    Example:
        # Select only the first 7 dimensions (e.g., left arm joints)
        processor = ActionSliceProcessorStep(action_indices=[0, 1, 2, 3, 4, 5, 6])
    """

    action_indices: list[int] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Slices the action tensor to keep only the specified indices.

        Args:
            transition: The input transition containing action.

        Returns:
            A new transition with the sliced action tensor.
        """
        if not self.action_indices:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is not None and isinstance(action, torch.Tensor):
            indices = torch.tensor(self.action_indices, device=action.device)
            # Handle action tensors with shape (B, T, D) or (B, D) or (D,)
            sliced_action = torch.index_select(action, dim=-1, index=indices)
            new_transition[TransitionKey.ACTION] = sliced_action

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration for serialization."""
        return {
            "action_indices": self.action_indices,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the action feature shape to reflect the slicing.

        Args:
            features: The input feature dictionary.

        Returns:
            Updated feature dictionary with the new action shape.
        """
        if not self.action_indices:
            return features

        action_features = features.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            original_feature = action_features[ACTION]
            new_shape = (len(self.action_indices),) + original_feature.shape[1:]
            action_features[ACTION] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
            features[PipelineFeatureType.ACTION] = action_features

        return features


@dataclass
@ProcessorStepRegistry.register("state_action_slice_processor")
class StateActionSliceProcessorStep(ProcessorStep):
    """
    Selects specific dimensions from both observation.state and action tensors.

    This is a combined processor that handles both state and action slicing in a single step.
    It's useful when you want to train a policy using only a subset of the robot's
    degrees of freedom for both input (state) and output (action).

    Attributes:
        state_indices: List of indices to select from observation.state.
                       If empty, all state dimensions are kept.
        action_indices: List of indices to select from action tensor.
                        If empty, all action dimensions are kept.
        state_key: The key for the state in observations. Defaults to "observation.state".

    Example:
        # Select left arm joints (indices 0-6) for both state and action
        processor = StateActionSliceProcessorStep(
            state_indices=[0, 1, 2, 3, 4, 5, 6],
            action_indices=[0, 1, 2, 3, 4, 5, 6],
        )
    """

    state_indices: list[int] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    state_key: str = OBS_STATE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Slices both state and action tensors to keep only the specified indices.

        Args:
            transition: The input transition.

        Returns:
            A new transition with sliced state and action tensors.
        """
        new_transition = transition.copy()

        # Slice state
        if self.state_indices:
            obs = new_transition.get(TransitionKey.OBSERVATION)
            if obs is not None and self.state_key in obs:
                state = obs[self.state_key]
                if isinstance(state, torch.Tensor):
                    indices = torch.tensor(self.state_indices, device=state.device)
                    sliced_state = torch.index_select(state, dim=-1, index=indices)
                    obs[self.state_key] = sliced_state
                    new_transition[TransitionKey.OBSERVATION] = obs

        # Slice action
        if self.action_indices:
            action = new_transition.get(TransitionKey.ACTION)
            if action is not None and isinstance(action, torch.Tensor):
                indices = torch.tensor(self.action_indices, device=action.device)
                sliced_action = torch.index_select(action, dim=-1, index=indices)
                new_transition[TransitionKey.ACTION] = sliced_action

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration for serialization."""
        return {
            "state_indices": self.state_indices,
            "action_indices": self.action_indices,
            "state_key": self.state_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates both state and action feature shapes to reflect the slicing.

        Args:
            features: The input feature dictionary.

        Returns:
            Updated feature dictionary with the new shapes.
        """
        # Update state feature
        if self.state_indices:
            obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
            if self.state_key in obs_features:
                original_feature = obs_features[self.state_key]
                new_shape = (len(self.state_indices),) + original_feature.shape[1:]
                obs_features[self.state_key] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
                features[PipelineFeatureType.OBSERVATION] = obs_features

        # Update action feature
        if self.action_indices:
            action_features = features.get(PipelineFeatureType.ACTION, {})
            if ACTION in action_features:
                original_feature = action_features[ACTION]
                new_shape = (len(self.action_indices),) + original_feature.shape[1:]
                action_features[ACTION] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
                features[PipelineFeatureType.ACTION] = action_features

        return features


@dataclass
@ProcessorStepRegistry.register("image_select_processor")
class ImageSelectProcessorStep(ProcessorStep):
    """
    Selects specific image keys from observations.

    This processor allows you to select a subset of camera views by specifying
    which image keys to keep. All other image keys will be removed.

    Attributes:
        image_keys: List of image keys to keep (e.g., ["observation.images.head", 
                    "observation.images.hand_left"]).
                    If empty, all images are kept.

    Example:
        # Keep only head and hand cameras
        processor = ImageSelectProcessorStep(
            image_keys=[
                "observation.images.head",
                "observation.images.hand_left",
                "observation.images.hand_right",
            ]
        )
    """

    image_keys: list[str] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Filters observations to keep only the specified image keys.

        Args:
            transition: The input transition.

        Returns:
            A new transition with only the selected image keys.
        """
        if not self.image_keys:
            return transition

        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION)

        if obs is not None:
            # Find all image keys in observations
            keys_to_remove = []
            for key in obs.keys():
                if key.startswith("observation.images.") or key == "observation.image":
                    if key not in self.image_keys:
                        keys_to_remove.append(key)

            # Remove unwanted image keys
            for key in keys_to_remove:
                del obs[key]

            new_transition[TransitionKey.OBSERVATION] = obs

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration for serialization."""
        return {
            "image_keys": self.image_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the feature dictionary to only include selected image keys.

        Args:
            features: The input feature dictionary.

        Returns:
            Updated feature dictionary with only selected image features.
        """
        if not self.image_keys:
            return features

        obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
        keys_to_remove = []
        for key in obs_features.keys():
            if key.startswith("observation.images.") or key == "observation.image":
                if key not in self.image_keys:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del obs_features[key]

        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
