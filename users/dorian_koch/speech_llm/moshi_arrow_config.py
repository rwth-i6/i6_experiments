"""Data / augmentation config for the arrow-based moshi finetune loader.

This lives on *our* side (not in the moshi-finetune fork). It is serialized as a
small JSON sidecar written next to a finetune's ``config.yaml`` and loaded by the
launcher entry point, which installs it into ``moshi_arrow_dataset`` before the
fork builds its data loader. This avoids adding fields to the fork's strict
``TrainArgs`` schema.

Keep this small and additive: new augmentation knobs go here as dataclass fields
with safe (no-op) defaults, so old runs without a sidecar keep their behaviour.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

SIDECAR_NAME = "arrow_data_config.json"


@dataclass
class ArrowDataConfig:
    # Max leading silence (seconds) prepended to each dialogue, sampled uniformly
    # in [0, jitter_max_sec] per row per epoch and clamped to the window headroom
    # so no content is truncated. 0.0 => no jitter (legacy behaviour).
    jitter_max_sec: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ArrowDataConfig":
        # Ignore unknown keys for forward/backward compatibility.
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save_beside(self, config_yaml_path: str) -> str:
        """Write the sidecar next to ``config_yaml_path``; return its path."""
        path = os.path.join(os.path.dirname(config_yaml_path), SIDECAR_NAME)
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load_beside(cls, config_yaml_path: str) -> "ArrowDataConfig":
        """Load the sidecar next to ``config_yaml_path``; defaults if absent."""
        path = os.path.join(os.path.dirname(config_yaml_path), SIDECAR_NAME)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))
