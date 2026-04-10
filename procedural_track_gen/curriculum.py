"""
Main entry point for track generation and curriculum config 

Phase 1 (difficulty < phase_boundary):  open Bezier chains
    - this phase is riddled with hardcoded hyper parameters 
    - in both the chain file and the resolve fn in this file 
    - if tracks look poor, hand tune till your fingers are sore :)
Phase 2 (difficulty geq phase_boundary):  full closed loop tracks
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bezier import build_spline, build_polylines
from .rasterise import rasterise_track
from .points import generate_points, sort_clockwise
from .features import FEATURES, FEATURE_ORDER
from .chain import build_chain


@dataclass
class CurriculumConfig:
    """
    Central knob: ``difficulty`` in [0, 1].

    Phase 1 (d < phase_boundary): open Bézier chains
        agent isolates individual features
    Phase 2 (d >= phase_boundary): closed loops
        agent handles continuous lapping

    None fields resolved via resolve()
    """
    difficulty:      float = 0.0
    phase_boundary:  float = 0.65

    # phase 1 params
    n_segments:      Optional[int]   = None
    segment_length:  Optional[float] = None
    intensity_range: Optional[tuple] = None

    # shared between
    track_width:       Optional[int] = None
    grid_size:         int           = 80
    steps_per_segment: int           = 40

    # phase 2 params
    n_control_points: Optional[int]   = None
    max_jitter:       Optional[float] = None
    min_radius:       Optional[float] = None

    def resolve(self):
        """Resolve empty fields via linear interpolation"""
        d = np.clip(self.difficulty, 0.0, 1.0)

        if self.track_width is None:
            self.track_width = int(np.interp(d, [0, 1], [8, 3]))

        if d < self.phase_boundary:
            dp = d / self.phase_boundary  # normalise within Phase 1
            if self.n_segments is None:
                self.n_segments = int(np.interp(dp, [0, 1], [4, 12]))
            if self.segment_length is None:
                self.segment_length = 20.0 
            if self.intensity_range is None:
                lo = float(np.interp(dp, [0, 1], [0.15, 0.50]))
                hi = float(np.interp(dp, [0, 1], [0.40, 1.00]))
                self.intensity_range = (lo, hi)
        else:
            dp = (d - self.phase_boundary) / (1.0 - self.phase_boundary)
            if self.n_control_points is None:
                self.n_control_points = int(np.interp(dp, [0, 1], [6, 14]))
            if self.max_jitter is None:
                self.max_jitter = float(np.interp(dp, [0, 1], [0.15, 0.40]))
            if self.min_radius is None:
                self.min_radius = float(np.interp(dp, [0, 1], [0.50, 0.30]))
        return self

    @property
    def is_chain(self) -> bool:
        return self.difficulty < self.phase_boundary

    def available_features(self) -> list[str]:
        return [f for f in FEATURE_ORDER if FEATURES[f]["unlock"] <= self.difficulty]


def generate_track(
    map_size:  tuple[int, int]     = (500, 500),
    spacing:   tuple[float, float] = (0.3, 0.3),
    config:    CurriculumConfig    = None,
) -> tuple[np.ndarray, dict]:
    """
    Single entry point for the full curriculum.

    Returns
        grid : np.ndarray[bool] rasterised traversability map
        meta : dict - phase, features, config details
    """
    if config is None:
        config = CurriculumConfig()
    config.resolve()

    num_rows, num_cols = map_size
    width  = num_cols * spacing[1]
    height = num_rows * spacing[0]

    if config.is_chain:
        return generate_chain(config, width, height)
    else:
        return generate_loop(config, width, height)


def generate_chain(config, width, height):
    """Phase 1: chain builder""" 
    start_pos = np.array([width / 2, height * 0.85])
    start_tan = np.array([0.0, -1.0]) * config.segment_length #go down

    segments, feat_log = build_chain(
        config.n_segments, config.segment_length,
        config.difficulty, config.intensity_range,
        start_pos, start_tan,
    )
    polyline = build_polylines(segments, config.steps_per_segment)

    # fit polyline into a tight bounding box for rasterisation
    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    pad = config.segment_length
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    w, h = max_x - min_x, max_y - min_y
    shifted = [(p[0] - min_x, p[1] - min_y) for p in polyline]

    grid = rasterise_track(
        shifted, config.grid_size, config.track_width, w, h, closed=False)

    return grid, {
        "phase": "chain",
        "features": feat_log,
        "n_segments": len(segments),
        "config": config,
    }


def generate_loop(config, width, height):
    """Phase 2: closed loop full track."""
    raw = generate_points(
        config.n_control_points, width, height,
        config.min_radius, margin=width * 0.05,
        max_jitter=config.max_jitter,
    )
    pts = sort_clockwise(raw)
    segments = build_spline(pts)
    polyline = build_polylines(segments, config.steps_per_segment)
    grid = rasterise_track(
        polyline, config.grid_size, config.track_width,
        width, height, closed=True)

    return grid, {
        "phase": "loop",
        "n_control_points": config.n_control_points,
        "max_jitter": config.max_jitter,
        "config": config,
    }
