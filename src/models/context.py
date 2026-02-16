"""Data model for the editing context passed to each agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "content_types.json"


@dataclass
class EditingContext:
    """All the information agents need to analyse a piece of content."""

    content: str
    content_type: str  # "article" | "social_media" | "video_script"
    rules: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rules:
            self.rules = self._load_rules()

    def _load_rules(self) -> dict[str, Any]:
        with open(CONFIG_PATH) as f:
            all_rules = json.load(f)
        return all_rules.get(self.content_type, {})

    @property
    def label(self) -> str:
        return self.rules.get("label", self.content_type)
