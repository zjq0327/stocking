"""Parameters for the static, single-tube plain-knit material.

The shape parameters use Keenan Crane's normalized coordinates.  ``scale_mm``
converts *all* geometric lengths to millimetres; h and d are amplitudes, not
peak-to-peak heights.  This parameterization does not model physical stretch.
"""

from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


@dataclass
class Parameters:
    a: float = 1.5
    h: float = 4.0
    d: float = 1.0
    R: float = 0.5
    rowOffset: float = 4.5
    scale_mm: float = 0.05
    nRows: int = 5
    nLoops: int = 5
    samples_per_loop: int = 192
    tube_sides: int = 24
    width: int = 256
    height: int = 256

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check scalar values; geometry generation also checks local curvature.

        Positive h and d and nonnegative a keep the analytic Frenet frame away
        from its mathematical singularities.  Global self/contact intersections
        are a separate geometry check, not a material relaxation operation.
        """
        for name in ("a", "h", "d", "R", "rowOffset", "scale_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if (name == "a" and value < 0) or (name != "a" and value <= 0):
                relation = "nonnegative" if name == "a" else "positive"
                raise ValueError(f"{name} must be {relation}")

        minimums = {
            "nRows": 1, "nLoops": 1, "samples_per_loop": 16,
            "tube_sides": 6, "width": 1, "height": 1,
        }
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")

        if not all(math.isfinite(v) and v > 0 for v in self.period_mm):
            raise ValueError("scale_mm produces an invalid physical period")
        for name in ("h", "d", "R"):
            physical_value = getattr(self, name) * self.scale_mm
            if not math.isfinite(physical_value) or physical_value <= 0.0:
                raise ValueError(f"{name} and scale_mm produce an invalid physical length")
        if not math.isfinite(self.h + self.R) or not math.isfinite(self.a + self.R):
            raise ValueError("shape bounds exceed the supported numeric range")

    @property
    def period_mm(self) -> tuple[float, float]:
        """Physical (course, wale) repeat distances in millimetres."""
        return 2.0 * math.pi * self.scale_mm, self.rowOffset * self.scale_mm

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible snapshot, including defaults."""
        self.validate()
        result = asdict(self)
        for name in ("a", "h", "d", "R", "rowOffset", "scale_mm"):
            result[name] = float(result[name])
        for name in ("nRows", "nLoops", "samples_per_loop", "tube_sides", "width", "height"):
            result[name] = int(result[name])
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Parameters":
        """Load named parameters; omitted fields use the documented defaults."""
        if not isinstance(values, Mapping):
            raise ValueError("parameters must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown parameters: " + ", ".join(sorted(map(str, unknown))))
        return cls(**dict(values))

    @classmethod
    def load(cls, path: str | Path) -> "Parameters":
        with Path(path).open("r", encoding="utf-8-sig") as source:
            return cls.from_dict(json.load(source))
