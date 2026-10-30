"""Explicit Transport data contract; no ALOHA joint-space conversion is used."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
ACTION_DIM = 14


def read_key(data: dict, key: str) -> Any:
    """Accept literal LeRobot keys or an explicitly named nested dotted path."""
    if key in data:
        return data[key]
    value = data
    for component in key.split("."):
        value = value[component]
    return value


@dataclasses.dataclass(frozen=True)
class StateField:
    key: str
    size: int
    arm: str
    meaning: str
    indices: tuple[int, ...] | None = None

    def extract(self, data: dict) -> np.ndarray:
        value = np.asarray(read_key(data, self.key), dtype=np.float32)
        if value.ndim != 1:
            raise ValueError(f"State {self.key} must be a vector, got {value.shape}")
        if self.indices is not None:
            value = value[list(self.indices)]
        if value.shape != (self.size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid state field {self.key}: shape={value.shape}")
        return value


@dataclasses.dataclass(frozen=True)
class TransportSchema:
    cameras: dict[str, str]
    state_fields: tuple[StateField, ...]
    action_key: str
    arm_order: tuple[str, str]
    action_description: str
    fps: float
    audit_source: str
    prompt_key: str = "prompt"

    @classmethod
    def load(cls, path: str | Path) -> "TransportSchema":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1 or data.get("audited") is not True:
            raise ValueError("Require schema_version=1 and audited=true after inspecting real data.")
        if data.get("action_dim") != ACTION_DIM:
            raise ValueError("This adapter is specifically for two 7-D end-effector commands.")
        if data.get("action_representation") != "relative_eef_controller_command":
            raise ValueError("Require audited relative EEF controller commands, not ALOHA joints.")
        fields = tuple(
            StateField(**{**field, "indices": tuple(field["indices"]) if field.get("indices") is not None else None})
            for field in data["state_fields"]
        )
        schema = cls(
            cameras=dict(data["cameras"]), state_fields=fields,
            action_key=data["action_key"], arm_order=tuple(data["arm_order"]),
            action_description=data["action_description"], fps=float(data["fps"]),
            audit_source=data["audit_source"], prompt_key=data.get("prompt_key", "prompt"),
        )
        schema.validate()
        return schema

    def validate(self) -> None:
        if set(self.cameras) != set(CAMERA_SLOTS) or len(set(self.cameras.values())) != 3:
            raise ValueError("Map three distinct real RGB views to the three model camera slots.")
        if len(self.arm_order) != 2 or len(set(self.arm_order)) != 2:
            raise ValueError("Explicitly identify the two arms in controller action order.")
        if {field.arm for field in self.state_fields} != set(self.arm_order):
            raise ValueError("Proprioception must include both arms, with explicit arm labels.")
        if not 0 < self.state_dim <= 32:
            raise ValueError("Select audited proprioceptive fields totalling 1..32 dimensions.")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must match the demonstration control frequency.")
        required_text = [self.action_key, self.prompt_key, self.audit_source, self.action_description,
                         *self.cameras.values(), *self.arm_order]
        for field in self.state_fields:
            required_text.extend([field.key, field.meaning])
            if field.size <= 0 or (field.indices is not None and len(field.indices) != field.size):
                raise ValueError(f"Invalid state field dimension: {field}")
            if field.indices is not None and (len(set(field.indices)) != len(field.indices) or min(field.indices) < 0):
                raise ValueError(f"Invalid state indices: {field}")
        if any(not isinstance(value, str) or not value.strip() or "TODO" in value for value in required_text):
            raise ValueError("The schema contains empty or unaudited placeholder fields.")

    @property
    def state_dim(self) -> int:
        return sum(field.size for field in self.state_fields)

    def arm_state_indices(self, arm: str) -> list[int]:
        indices, start = [], 0
        for field in self.state_fields:
            if field.arm == arm:
                indices.extend(range(start, start + field.size))
            start += field.size
        return indices

    def summary(self) -> dict:
        return {"state_dim": self.state_dim, "action_dim": ACTION_DIM,
                "action_representation": "relative_eef_controller_command",
                "cameras": self.cameras, "arm_order": self.arm_order,
                "state_fields": [dataclasses.asdict(field) for field in self.state_fields],
                "action_description": self.action_description, "fps": self.fps,
                "audit_source": self.audit_source, "discrete_state_input": True}


def parse_rgb(value: Any) -> np.ndarray:
    """LeRobot float CHW [0,1] and simulator uint8 HWC are accepted explicitly."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB frame, got {image.shape}")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
        raise ValueError(f"Expected uint8 or finite floating RGB, got {image.dtype}")
    if image.min() < -1e-6 or image.max() > 1 + 1e-6:
        raise ValueError("Floating RGB must be in [0,1]; convert raw [0,255] explicitly.")
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


@dataclasses.dataclass(frozen=True)
class TransportInputs:
    schema: TransportSchema

    def __call__(self, data: dict) -> dict:
        prompt = read_key(data, self.schema.prompt_key)
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task instruction is required.")
        result = {
            "state": np.concatenate([field.extract(data) for field in self.schema.state_fields]),
            "image": {slot: parse_rgb(read_key(data, key)) for slot, key in self.schema.cameras.items()},
            "image_mask": {slot: np.True_ for slot in CAMERA_SLOTS},
            "prompt": prompt,
        }
        try:
            actions = read_key(data, self.schema.action_key)
        except KeyError:
            return result  # Inference does not provide supervision.
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim not in (1, 2) or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite (...,{ACTION_DIM}) action, got {actions.shape}")
        result["actions"] = actions  # Already controller deltas: never subtract state again.
        return result


@dataclasses.dataclass(frozen=True)
class TransportOutputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"Invalid model action chunk {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
