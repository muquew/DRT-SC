from collections.abc import Mapping
from contextlib import contextmanager
import gzip
import io
import json
from pathlib import Path

from common.data.roads.schema import RoadArc, RoadNetwork, RoadNode, RoadSegment


_HEADER_KEYS = {
    "type", "name", "version", "crs", "travel_mode",
    "metadata", "counts",
}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@contextmanager
def _text_file(path, mode):
    path = Path(path)
    if path.suffix == ".gz":
        with path.open(mode + "b") as raw:
            with gzip.GzipFile(fileobj=raw, mode=mode + "b", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", errors="backslashreplace" if mode == "w" else "strict",
                                      newline="\n") as stream:
                    yield stream
    else:
        with path.open(mode, encoding="utf-8", errors="backslashreplace" if mode == "w" else "strict",
                       newline="\n") as stream:
            yield stream


def save_network(roads: RoadNetwork, path: Path) -> None:

    if not isinstance(roads, RoadNetwork):
        raise TypeError("roads must be a RoadNetwork")
    with _text_file(path, "w") as stream:
        def write(record):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")) + "\n")

        write({
            "type": "header",
            "name": roads.name, "version": roads.version, "crs": roads.crs,
            "travel_mode": roads.travel_mode, "metadata": _plain(roads.metadata),
            "counts": {"nodes": len(roads.nodes), "segments": len(roads.segments),
                       "arcs": len(roads.arcs)},
        })
        for node in sorted(roads.nodes, key=lambda item: item.id):
            write({"type": "node", "id": node.id, "x": node.x, "y": node.y})
        for segment in sorted(roads.segments, key=lambda item: item.id):
            write({"type": "segment", "id": segment.id, "u": segment.u, "v": segment.v,
                   "geometry": segment.geometry})
        for arc in sorted(roads.arcs, key=lambda item: item.id):
            write({"type": "arc", "id": arc.id, "segment_id": arc.segment_id,
                   "forward": arc.forward, "travel_time_s": arc.travel_time_s,
                   "enabled": arc.enabled})


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON number {value}")


def _read_record(line, line_number):
    try:
        record = json.loads(line, parse_constant=_invalid_constant,
                            object_pairs_hook=_unique_pairs)
        if not isinstance(record, dict):
            raise ValueError("expected a JSON object")
    except ValueError as error:
        raise ValueError(f"road file line {line_number}: {error}") from error
    return record


def load_network(path: Path) -> RoadNetwork:

    with _text_file(path, "r") as stream:
        line = stream.readline()
        if not line:
            raise ValueError("road file is empty")
        header = _read_record(line, 1)
        if not _HEADER_KEYS <= set(header) or header["type"] != "header":
            raise ValueError("road file header is missing required fields")
        counts = header["counts"]
        if (not isinstance(counts, dict) or set(counts) != {"nodes", "segments", "arcs"}
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("header counts must contain nonnegative integers for nodes, segments and arcs")
        nodes, segments, arcs = [], [], []
        kinds = {"node": (RoadNode, nodes, "nodes"),
                 "segment": (RoadSegment, segments, "segments"),
                 "arc": (RoadArc, arcs, "arcs")}
        for line_number, line in enumerate(stream, start=2):
            record = _read_record(line, line_number)
            kind = record.pop("type", None)
            if not isinstance(kind, str) or kind not in kinds:
                raise ValueError(f"road file line {line_number}: unknown record type {kind!r}")
            record_class, records, count_name = kinds[kind]
            try:
                records.append(record_class(**record))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"road file line {line_number}: {error}") from error
            if len(records) > counts[count_name]:
                raise ValueError(f"road file has more {count_name} than its header count")
    if {"nodes": len(nodes), "segments": len(segments), "arcs": len(arcs)} != counts:
        raise ValueError("road file counts do not match the header; file may be incomplete")
    roads = RoadNetwork(name=header["name"], version=header["version"], crs=header["crs"],
                        travel_mode=header["travel_mode"], metadata=header["metadata"],
                        nodes=nodes, segments=segments, arcs=arcs)
    return roads
