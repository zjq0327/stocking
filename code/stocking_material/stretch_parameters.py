"""Settings for the simplified periodic quasistatic yarn-stretch solver.

Lengths use millimetres, forces use newtons, and energies use N mm.  These
demonstration defaults are not a material calibration for elastic hosiery.
"""

from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


@dataclass
class StretchParameters:
    nodes: int = 64
    axial_stiffness_N: float = 1.0
    bending_stiffness_N_mm2: float = 1e-5
    contact_stiffness_N_per_mm: float = 100.0
    lambda_x: float = 1.2
    lambda_y: float = 1.0
    load_steps: int = 8
    max_iterations: int = 2500
    gradient_tolerance: float = 1e-6
    contact_margin_ratio: float = 0.05
    max_strain: float = 0.03

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate inputs, without claiming that an equilibrium is feasible.

        The first version prescribes both periodic dimensions.  Compression
        and a force-free transverse boundary are not supported loading modes.
        """
        for name in ("nodes", "load_steps", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 32 <= self.nodes <= 256:
            raise ValueError("nodes must be between 32 and 256")

        for name in (
            "axial_stiffness_N", "bending_stiffness_N_mm2",
            "contact_stiffness_N_per_mm", "lambda_x", "lambda_y",
            "gradient_tolerance", "contact_margin_ratio", "max_strain",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive number")
            try:
                finite_value = float(value)
            except (OverflowError, ValueError):
                raise ValueError(f"{name} must be a finite positive number") from None
            if not math.isfinite(finite_value) or finite_value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("lambda_x", "lambda_y"):
            if not 1.0 <= getattr(self, name) <= 1.6:
                raise ValueError(f"{name} must be between 1.0 and 1.6")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot including every default value."""
        self.validate()
        result = asdict(self)
        integer_fields = {"nodes", "load_steps", "max_iterations"}
        for name, value in result.items():
            result[name] = int(value) if name in integer_fields else float(value)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "StretchParameters":
        """Accept known fields only; omitted fields use the defaults."""
        if not isinstance(values, Mapping):
            raise ValueError("stretch parameters must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown stretch parameters: " + ", ".join(sorted(map(str, unknown))))
        return cls(**dict(values))

    @classmethod
    def load(cls, path: str | Path) -> "StretchParameters":
        with Path(path).open("r", encoding="utf-8-sig") as source:
            return cls.from_dict(json.load(source))
