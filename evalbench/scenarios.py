"""Scenario definitions for model/agent evaluations.

A scenario is a pre-defined test case: the instructions given to the model
plus the checks that its output must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """A single evaluation scenario.

    Attributes:
        id: Stable identifier, e.g. "instruction-following-01".
        task: Human-readable description of what is being evaluated.
        system_prompt: System prompt sent to the model (may be empty).
        user_input: The user message for this scenario.
        checks: List of check specs (see evalbench.checks for types).
        category: Grouping used in reports, e.g. "instruction-following".
        difficulty: Optional label ("easy" / "medium" / "hard").
    """

    id: str
    task: str
    system_prompt: str
    user_input: str
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    category: str = "general"
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Build a Scenario from a parsed JSON dict, validating required fields."""
        missing = [k for k in ("id", "task", "user_input") if not data.get(k)]
        if missing:
            raise ValueError(
                f"Scenario is missing required field(s): {', '.join(missing)}"
            )
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            raise ValueError(f"Scenario {data.get('id')!r}: 'checks' must be a list")
        return cls(
            id=str(data["id"]),
            task=str(data["task"]),
            system_prompt=str(data.get("system_prompt", "")),
            user_input=str(data["user_input"]),
            checks=tuple(checks),
            category=str(data.get("category", "general")),
            difficulty=str(data.get("difficulty", "medium")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "system_prompt": self.system_prompt,
            "user_input": self.user_input,
            "checks": list(self.checks),
            "category": self.category,
            "difficulty": self.difficulty,
        }