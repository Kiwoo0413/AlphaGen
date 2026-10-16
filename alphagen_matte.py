"""CPU matte refinement, informed by CompMatte's core/edge separation.

Independent implementation for an existing LTX alpha, not an optical screen keyer.
No input RGB colour keying, polarity guessing, or temporal mask propagation.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Settings:
    black_clip: float = 0.05
    white_clip: float = 0.95
    gamma: float = 1.0
    core_threshold: float = 0.6
    core_inset: int = 3
    core_strength: float = 1.0
    fill_holes: bool = False
    max_hole_area: int = 64
    safe_radius: int = 6
    edge_restore: float = 0.5
    feather: float = 0.35

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Matte settings must be finite")
        if not 0 <= self.black_clip < self.white_clip <= 1:
            raise ValueError("Require 0 <= black clip < white clip <= 1")
        if not 0.1 <= self.gamma <= 5:
            raise ValueError("Gamma must be 0.1..5")
        if not 0 < self.core_threshold <= 1:
            raise ValueError("Core threshold must be in (0, 1]")
        for name in ("core_strength", "edge_restore"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name + " must be 0..1")
        for name in ("core_inset", "safe_radius", "max_hole_area"):
            value = getattr(self, name)
            if int(value) != value or not 0 <= value <= 100000:
                raise ValueError(name + " must be a nonnegative integer")
        if self.core_inset > 100 or self.safe_radius > 100 or not 0 <= self.feather <= 10:
            raise ValueError("Morphology radius must be <= 100, feather <= 10")


def kernel(radius):
    radius = int(radius)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def fill_small_holes(binary, max_area):
    """Fill enclosed background components only; preserve border and large holes."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=8)
    border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    result = binary.copy()
    for index in range(1, count):
        if index not in border and stats[index, cv2.CC_STAT_AREA] <= max_area:
            result[labels == index] = 1
    return result


def refine(alpha, settings=None):
    s = settings or Settings()
    s.validate()
    raw = np.asarray(alpha, dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite 2D alpha")
    raw = np.clip(raw, 0, 1)
    clean = np.clip((raw - s.black_clip) / (s.white_clip - s.black_clip), 0, 1)
    clean = np.power(clean, 1.0 / s.gamma)
    binary = (raw >= s.core_threshold).astype(np.uint8)
    if s.fill_holes and s.max_hole_area:
        binary = fill_small_holes(binary, s.max_hole_area)
    core = binary
    if s.core_inset:
        # Outside-image pixels are absent, not opaque: inset also at image borders.
        core = cv2.erode(core, kernel(s.core_inset), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    core = core.astype(np.float32) * s.core_strength
    safe = (core > 0).astype(np.uint8)
    if s.safe_radius:
        safe = cv2.dilate(safe, kernel(s.safe_radius))
    edge = clean.copy()
    if s.feather > 0:
        edge = cv2.GaussianBlur(edge, (0, 0), s.feather)
    # Never re-inject raw gray below the artist's black cutoff.
    detail = np.where(raw > s.black_clip, raw, 0)
    edge += s.edge_restore * safe * np.maximum(detail - edge, 0)
    final = np.maximum(edge, core)
    return {"raw": raw, "core": core, "edge": np.clip(edge, 0, 1),
            "final": np.clip(final, 0, 1).astype(np.float32)}
