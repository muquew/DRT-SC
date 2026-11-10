from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from ..schema import RoadArc, RoadNetwork, RoadNode, RoadSegment


_DRIVING = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service",
)
_CYCLING = tuple(value for value in _DRIVING if not value.startswith(("motorway", "trunk"))) + (
    "cycleway", "path", "track", "footway", "pedestrian",
)
_ALLOWED_ACCESS = {"yes", "designated", "official", "permissive", "discouraged"}
_BLOCKING_BARRIERS = {"wall", "fence", "block", "jersey_barrier"}


@dataclass(frozen=True, kw_only=True)
class OSMBuildConfig:


    name: str
    version: str
    target_crs: str
    travel_mode: str
    speed_m_per_s: float
    bounds: tuple[float, float, float, float] | None = None
    highway_types: tuple[str, ...] | None = None

    def __post_init__(self):
        for label in ("name", "version", "target_crs"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonempty string")
        if self.travel_mode not in {"driving", "cycling"}:
            raise ValueError("travel_mode must be driving or cycling")
        if (isinstance(self.speed_m_per_s, bool)
                or not isinstance(self.speed_m_per_s, (int, float))
                or not isfinite(self.speed_m_per_s) or self.speed_m_per_s <= 0):
            raise ValueError("speed_m_per_s must be finite and positive")
        if self.bounds is not None:
            if (not isinstance(self.bounds, tuple) or len(self.bounds) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not isfinite(v) for v in self.bounds)):
                raise ValueError("bounds must be a finite (west, south, east, north) tuple")
            west, south, east, north = self.bounds
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError("bounds must be ordered longitude/latitude bounds")
        if self.highway_types is not None and (
            not isinstance(self.highway_types, tuple) or not self.highway_types
            or any(not isinstance(v, str) or not v for v in self.highway_types)
            or len(set(self.highway_types)) != len(self.highway_types)
        ):
            raise ValueError("highway_types must be a nonempty tuple of distinct names")


def _access_keys(mode):
    return ("motorcar", "motor_vehicle", "vehicle", "access") if mode == "driving" else (
        "bicycle", "vehicle", "access")


def _access(tags, mode, direction=None):

    for key in _access_keys(mode):
        if direction is not None and f"{key}:{direction}" in tags:
            return tags[f"{key}:{direction}"]
        if key in tags:
            return tags[key]
    return None


def _conditional_keys(tags, mode):
    keys = set(_access_keys(mode))
    keys.update(f"{key}:{direction}" for key in _access_keys(mode)
                for direction in ("forward", "backward"))
    keys.update(f"oneway:{key}" for key in _access_keys(mode) if key != "access")
    keys.add("oneway")
    return [f"{key}:conditional" for key in sorted(keys) if f"{key}:conditional" in tags]


def _directions(tags, mode):
    keys = tuple(f"oneway:{key}" for key in _access_keys(mode) if key != "access") + ("oneway",)
    value = next((tags[key] for key in keys if key in tags), None)
    if mode == "cycling" and "oneway:bicycle" not in tags and any(
        tags.get(key) in {"opposite", "opposite_lane", "opposite_track"}
        for key in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
    ):
        value = "no"
    if value is None:
        value = "yes" if tags.get("junction") == "roundabout" or tags.get("highway") == "motorway" else "no"
    if value in {"yes", "true", "1"}:
        return (True,)
    if value == "-1":
        return (False,)
    if value in {"no", "false", "0"}:
        return (True, False)
    return ()


class OSMRoadAdapter:


    def adapt(self, source: Path, *, config: OSMBuildConfig) -> RoadNetwork:
        import osmium
        from pyproj import CRS, Transformer

        if not isinstance(config, OSMBuildConfig):
            raise TypeError("config must be an OSMBuildConfig")
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        crs = CRS.from_user_input(config.target_crs)
        if (not crs.is_projected or len(crs.axis_info) != 2
                or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info)):
            raise ValueError("target_crs must be a two-dimensional projected CRS in metres")
        transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        allowed_highways = set(config.highway_types or (
            _DRIVING if config.travel_mode == "driving" else _CYCLING))
        counts = Counter()
        conditional_tags = Counter()
        nodes, segments, arcs = {}, [], []
        blocked_nodes = set()
        seen_ways = set()
        seen_nodes = set()
        missing_node_refs = set()

        class Handler(osmium.SimpleHandler):
            def node(self, node):
                if node.id in missing_node_refs:
                    raise ValueError("OSM node appeared after a way referencing it; sort the snapshot before import")
                if node.id in seen_nodes:
                    raise ValueError("OSM snapshot contains duplicate node IDs; history files need preprocessing")
                seen_nodes.add(node.id)
                counts["source_nodes"] += 1
                if not node.tags:
                    return
                tags = dict(node.tags)
                conditional = _conditional_keys(tags, config.travel_mode)
                conditional_tags.update(conditional)
                access = _access(tags, config.travel_mode)
                barrier = tags.get("barrier")
                if conditional or (access is not None and access not in _ALLOWED_ACCESS):
                    blocked_nodes.add(node.id)
                elif access is None and (barrier in _BLOCKING_BARRIERS or (
                    config.travel_mode == "driving" and barrier in {
                        "bollard", "cycle_barrier", "stile", "turnstile", "kissing_gate"})):
                    blocked_nodes.add(node.id)
                elif barrier and access is None:
                    counts["nodes_with_unresolved_barrier_access"] += 1

            def way(self, way):
                counts["source_ways"] += 1
                if way.id in seen_ways:
                    raise ValueError("OSM snapshot contains duplicate way IDs; history files need preprocessing")
                seen_ways.add(way.id)
                tags = dict(way.tags)
                highway = tags.get("highway")
                if highway not in allowed_highways or tags.get("area") == "yes":
                    counts["ways_excluded_highway_or_area"] += 1
                    return
                conditional = _conditional_keys(tags, config.travel_mode)
                if conditional:
                    conditional_tags.update(conditional)
                    counts["ways_excluded_conditional"] += 1
                    return
                directions = _directions(tags, config.travel_mode)
                if not directions:
                    counts["ways_excluded_unknown_direction"] += 1
                    return
                if config.travel_mode == "cycling" and highway in {"footway", "pedestrian"}:
                    directions = tuple(forward for forward in directions if tags.get(
                        "bicycle:forward" if forward else "bicycle:backward", tags.get("bicycle"))
                        in _ALLOWED_ACCESS)
                    if not directions:
                        counts["ways_excluded_cycling_permission"] += 1
                        return
                directions = tuple(forward for forward in directions if (
                    _access(tags, config.travel_mode, "forward" if forward else "backward")
                    in _ALLOWED_ACCESS | {None}))
                if not directions:
                    counts["ways_excluded_access"] += 1
                    return
                if any(key.startswith("maxspeed") for key in tags):
                    counts["ways_with_speed_tags_ignored"] += 1
                way_nodes = list(way.nodes)
                for index, (first, second) in enumerate(zip(way_nodes, way_nodes[1:])):
                    counts["candidate_segments"] += 1
                    if not first.location.valid() or not second.location.valid():
                        missing_node_refs.update(n.ref for n in (first, second) if not n.location.valid())
                        counts["segments_excluded_missing_location"] += 1
                        continue
                    if first.ref in blocked_nodes or second.ref in blocked_nodes:
                        counts["segments_excluded_blocked_node"] += 1
                        continue
                    if config.bounds is not None:
                        west, south, east, north = config.bounds
                        if not all(west <= n.lon <= east and south <= n.lat <= north for n in (first, second)):
                            counts["segments_excluded_bounds"] += 1
                            continue
                    xy = tuple(transform.transform(n.lon, n.lat, errcheck=True) for n in (first, second))
                    if first.ref == second.ref or xy[0] == xy[1]:
                        counts["segments_excluded_zero_length"] += 1
                        continue
                    first_id, second_id = str(first.ref), str(second.ref)
                    segment = RoadSegment(id=f"osm:{way.id}:{index}", u=first_id, v=second_id, geometry=xy)
                    segments.append(segment)
                    for identifier, point in zip((first_id, second_id), xy):
                        nodes[identifier] = RoadNode(id=identifier, x=point[0], y=point[1])
                    arcs.extend(RoadArc(id=f"{segment.id}:{'f' if forward else 'r'}",
                                        segment_id=segment.id, forward=forward,
                                        travel_time_s=segment.length_m / config.speed_m_per_s)
                                for forward in directions)

            def relation(self, relation):
                counts["source_relations"] += 1
                if relation.tags.get("type") == "restriction":
                    counts["turn_restriction_relations_ignored"] += 1

        locations = osmium.NodeLocationsForWays(osmium.index.create_map("flex_mem"))
        locations.ignore_errors()
        osmium.apply(str(source), locations, Handler())
        if not segments:
            raise ValueError(f"No usable road segments under this configuration: {dict(counts)}")
        counts.update(nodes=len(nodes), segments=len(segments), arcs=len(arcs), blocked_nodes=len(blocked_nodes))
        return RoadNetwork(
            name=config.name, version=config.version, crs=crs.to_string(), travel_mode=config.travel_mode,
            nodes=tuple(nodes.values()), segments=tuple(segments), arcs=tuple(arcs),
            metadata={
                "source": {"path": str(source.resolve()),
                           "format": "OpenStreetMap snapshot", "coordinate_crs": "EPSG:4326"},
                "config": asdict(config), "counts": dict(counts),
                "conditional_tags_excluded": dict(conditional_tags),
                "build_rules": {
                    "speed": "constant configured metres per second; maxspeed tags are ignored",
                    "access": "most specific mode tag; only public/permissive access; directional overrides supported",
                    "conditional_access": "exclude affected ways and incident segments of affected nodes",
                    "barriers": "exclude explicit denied access and known mode-blocking barriers; unclassified barriers counted",
                    "bounds": "keep segments with both source endpoints inside bounds; no geometric clipping",
                    "connectivity": "OSM node IDs; adjacent way nodes; all connected components retained",
                    "turn_restrictions": "ignored; relation count reported",
                    "cycling_defaults": "path/track allowed by default; footway/pedestrian require explicit bicycle permission",
                    "highway_types": sorted(allowed_highways),
                },
            },
        )
