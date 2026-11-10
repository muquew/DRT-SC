from dataclasses import dataclass
from fractions import Fraction
import math

from common.data.roads.schema import RoadArc, RoadNetwork, RoadNode, RoadSegment


def _finite(value, label, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or positive and value <= 0:
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))


@dataclass(frozen=True, kw_only=True)
class GridBuildConfig:


    name: str
    version: str
    crs: str
    bounds: tuple[float, float, float, float]
    spacing_m: float
    speed_m_per_s: float
    max_nodes: int = 250_000

    def __post_init__(self):
        for label in ("name", "version", "crs"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonempty string")
        if self.crs.startswith("local:"):
            if not self.crs[len("local:"):].strip():
                raise ValueError("crs must identify a local coordinate system")
        else:
            from pyproj import CRS
            from pyproj.exceptions import ProjError
            try:
                crs = CRS.from_user_input(self.crs)
            except ProjError as error:
                raise ValueError(f"invalid crs: {self.crs}") from error
            if (not crs.is_projected or len(crs.axis_info) != 2
                    or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info)):
                raise ValueError("crs must be local:* or a two-dimensional projected CRS in metres")
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 4:
            raise ValueError("bounds must be an (xmin, ymin, xmax, ymax) tuple")
        for value in self.bounds:
            _finite(value, "bounds")
        xmin, ymin, xmax, ymax = self.bounds
        if xmin > xmax or ymin > ymax:
            raise ValueError("bounds must have xmin <= xmax and ymin <= ymax")
        _finite(self.spacing_m, "spacing_m", positive=True)
        _finite(self.speed_m_per_s, "speed_m_per_s", positive=True)
        if type(self.max_nodes) is not int or self.max_nodes <= 0:
            raise ValueError("max_nodes must be a positive integer")


class GridRoadBuilder:


    def build(self, *, config: GridBuildConfig) -> RoadNetwork:
        if not isinstance(config, GridBuildConfig):
            raise TypeError("config must be a GridBuildConfig")


        step = Fraction(str(config.spacing_m))
        xmin, ymin, xmax, ymax = (Fraction(str(value)) for value in config.bounds)
        ix0, iy0 = math.floor(xmin / step) - 1, math.floor(ymin / step) - 1
        ix1, iy1 = math.ceil(xmax / step) + 1, math.ceil(ymax / step) + 1
        nx, ny = ix1 - ix0 + 1, iy1 - iy0 + 1
        if nx * ny > config.max_nodes:
            raise ValueError(f"grid requires {nx * ny} nodes, exceeding max_nodes={config.max_nodes}")

        def axis(start, stop):
            try:
                values = [float(i * step) for i in range(start, stop + 1)]
            except OverflowError as error:
                raise ValueError("grid coordinates exceed the finite numeric range") from error
            if any(not math.isfinite(v) for v in values) or any(
                    a >= b for a, b in zip(values, values[1:])):
                raise ValueError("grid spacing cannot be represented at these coordinates")
            return values

        xs, ys = axis(ix0, ix1), axis(iy0, iy1)

        def node_id(ix, iy):
            return f"grid{ix}_{iy}"

        nodes = tuple(RoadNode(id=node_id(ix0 + i, iy0 + j), x=x, y=y)
                      for i, x in enumerate(xs) for j, y in enumerate(ys))
        segments, arcs = [], []

        def connect(u, v, geometry):
            segment = RoadSegment(id=f"{u}:{v}", u=u, v=v, geometry=geometry)
            duration = segment.length_m / config.speed_m_per_s
            _finite(duration, "grid travel time", positive=True)
            segments.append(segment)
            for forward, suffix in ((True, "forward"), (False, "reverse")):
                arcs.append(RoadArc(id=f"{segment.id}:{suffix}", segment_id=segment.id,
                                    forward=forward, travel_time_s=duration))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                u = node_id(ix0 + i, iy0 + j)
                if i + 1 < nx:
                    connect(u, node_id(ix0 + i + 1, iy0 + j), ((x, y), (xs[i + 1], y)))
                if j + 1 < ny:
                    connect(u, node_id(ix0 + i, iy0 + j + 1), ((x, y), (x, ys[j + 1])))

        return RoadNetwork(
            name=config.name, version=config.version, crs=config.crs, travel_mode="synthetic",
            nodes=nodes, segments=tuple(segments), arcs=tuple(arcs),
            metadata={
                "source": {"kind": "synthetic_regular_grid"},
                "requested_bounds_m": config.bounds,
                "actual_bounds_m": (xs[0], ys[0], xs[-1], ys[-1]),
                "origin_m": (xs[0], ys[0]),
                "spacing_m": config.spacing_m,
                "alignment": "outward_to_global_spacing_multiples",
                "border_cells": 1,
                "speed_m_per_s": config.speed_m_per_s,
                "counts": {"nodes": len(nodes), "segments": len(segments), "arcs": len(arcs)},
            },
        )
