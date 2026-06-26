"""Config loading + path helpers. Single source of truth = config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path else PROJECT_ROOT / "config.yaml"
        with open(path) as f:
            return cls(yaml.safe_load(f))

    # convenience accessors -------------------------------------------------
    def __getitem__(self, key):
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Resolve a configured path (relative to project root) and ensure it exists."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def statsbomb_competitions(self) -> list[dict]:
        return self.raw["statsbomb"]["competitions"]

    @property
    def fbref_competitions(self) -> list[dict]:
        return self.raw["fbref"]["competitions"]
