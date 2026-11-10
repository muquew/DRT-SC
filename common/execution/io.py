import json
from pathlib import Path

from common.initial_sequences.io import _invalid_constant, _unique_pairs
from common.execution.state import ExecutionSnapshot


def save_execution_snapshot(snapshot: ExecutionSnapshot, path: Path) -> None:
    if not isinstance(snapshot, ExecutionSnapshot):
        raise TypeError("snapshot must be an ExecutionSnapshot")
    Path(path).write_text(json.dumps(dict(execution_snapshot=snapshot.to_dict()),
        ensure_ascii=False, allow_nan=False, sort_keys=True)+"\n")


def load_execution_snapshot(path: Path, replay=None) -> ExecutionSnapshot:
    record = json.loads(Path(path).read_text(), parse_constant=_invalid_constant,
                        object_pairs_hook=_unique_pairs)
    snapshot = ExecutionSnapshot.from_dict(record.get("execution_snapshot", record))
    if replay is not None and replay.at(snapshot.time_s) != snapshot:
        raise ValueError("snapshot does not match the executed service sequence")
    return snapshot
