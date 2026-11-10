from dataclasses import dataclass
import math


@dataclass(frozen=True, kw_only=True)
class CoordinateTransform:


    kind: str
    max_distance_m: float
    source_crs: str | None = None
    coordinate_scale: float = 1.0
    origin_lonlat: tuple[float, float] | None = None

    def __post_init__(self):
        if self.kind not in {"crs", "anonymous_equirectangular"}:
            raise ValueError("unknown coordinate transform")
        for name in ("max_distance_m", "coordinate_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.coordinate_scale == 0:
            raise ValueError("coordinate_scale must be positive")
        if self.kind == "crs":
            if not isinstance(self.source_crs, str) or not self.source_crs.strip():
                raise ValueError("crs policy requires an explicit source_crs")
            if self.origin_lonlat is not None:
                raise ValueError("origin_lonlat is only used by anonymous_equirectangular")
        else:
            if self.source_crs is not None:
                raise ValueError("anonymous construction does not assert a source CRS")
            if (not isinstance(self.origin_lonlat, tuple) or len(self.origin_lonlat) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.origin_lonlat)
                    or not -180 <= self.origin_lonlat[0] <= 180 or not -90 < self.origin_lonlat[1] < 90):
                raise ValueError("anonymous construction requires a finite lon/lat origin in degrees")


def experimental_xy(point, policy: CoordinateTransform):

    if not isinstance(point, tuple) or len(point) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in point):
        raise ValueError("expected two finite source coordinates")
    x, y = (value * policy.coordinate_scale for value in point)
    if policy.kind == "anonymous_equirectangular":
        if not -180 <= x <= 180 or not -90 <= y <= 90:
            raise ValueError("scaled anonymous coordinates fall outside longitude/latitude bounds")
        lon0, lat0 = policy.origin_lonlat
        radius = 6371008.8
        x, y = (radius * math.radians(x - lon0) * math.cos(math.radians(lat0)),
                radius * math.radians(y - lat0))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("spatial transform produced nonfinite coordinates")
    return x, y
