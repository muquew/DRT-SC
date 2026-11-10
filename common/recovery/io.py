import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from common.io import _invalid_constant, _json_float, _unique_pairs
from common.data.roads.schema import RoadNetwork
from common.execution.state import ExecutionSnapshot
from common.scenarios.disruptions import DisruptionEvent, apply_disruptions
from common.recovery.inputs import RecoveryInput, RecoverySettings


def save_recovery_input(item: RecoveryInput, path: Path) -> None:

    if not isinstance(item, RecoveryInput):
        raise TypeError("item must be a RecoveryInput")
    record = {"recovery_input": item.to_dict()}
    content = json.dumps(record, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")) + "\n"
    path = Path(path)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", errors="backslashreplace",
                                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_recovery_input(path: Path, source_snapshot: ExecutionSnapshot,
                        original_roads: RoadNetwork) -> RecoveryInput:

    if not isinstance(source_snapshot, ExecutionSnapshot):
        raise TypeError("source_snapshot must be an ExecutionSnapshot")
    if not isinstance(original_roads, RoadNetwork):
        raise TypeError("original_roads must be a RoadNetwork")
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"),
                            parse_constant=_invalid_constant, parse_float=_json_float,
                            object_pairs_hook=_unique_pairs)
        if not isinstance(record, dict) or set(record) != {"recovery_input"}:
            raise ValueError("recovery input file must contain a recovery_input object")
        stored = record["recovery_input"]
        if not isinstance(stored, dict) or not isinstance(stored.get("events"), list):
            raise ValueError("recovery_input must be an object with an events array")
        settings = RecoverySettings.from_dict(stored["source_settings"])
        events = tuple(DisruptionEvent.from_dict(event) for event in stored["events"])
        restored = apply_disruptions(source_snapshot, original_roads, events, settings,
                                     instance_id=stored["instance_id"])
        return restored
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"recovery input file {path}: {error}") from error
