from collections import Counter
from dataclasses import dataclass
import math

from common.execution.replay import PlanReplay
from common.execution.state import ExecutionSnapshot


@dataclass(frozen=True, kw_only=True)
class SnapshotSelection:
    snapshot: ExecutionSnapshot
    mode: str
    target_remaining_tasks: int | None
    T_plan_s: float
    T_ref_s: float

    def to_dict(self) -> dict:

        return {
            "mode": self.mode,
            "target_remaining_tasks": self.target_remaining_tasks,
            "time_s": self.snapshot.time_s,
            "remaining_task_count": len(self.snapshot.unfinished_task_ids),
            "T_plan_s": self.T_plan_s,
            "T_ref_s": self.T_ref_s,
        }


def select_snapshot(replay: PlanReplay, *, time_s: float | None = None,
                    remaining_tasks: int | None = None) -> SnapshotSelection:


    if not isinstance(replay, PlanReplay):
        raise TypeError("replay must be a PlanReplay")
    if (time_s is None) == (remaining_tasks is None):
        raise ValueError("select exactly one of time_s or remaining_tasks")
    if remaining_tasks is None:
        snapshot = replay.at(time_s)
        mode = "time"
    else:
        if type(remaining_tasks) is not int or not 0 <= remaining_tasks <= len(replay.plan.completion_times_s):
            raise ValueError("remaining_tasks must be an integer between zero and the original served task count")
        start = replay.planning_input.planning_time_s
        snapshot = replay.at(start)
        count = len(snapshot.unfinished_task_ids)
        if count != remaining_tasks:
            completions = Counter(t for t in replay.plan.completion_times_s.values() if t > start)
            for completion_time, simultaneous in sorted(completions.items()):
                if count < remaining_tasks:
                    break
                count -= simultaneous
                if count == remaining_tasks:
                    snapshot = replay.at(completion_time)
                    break
            else:
                count = -1
            if count != remaining_tasks:
                raise ValueError(
                    f"remaining task count {remaining_tasks} is not reached by the original plan; "
                    "simultaneous completions are retained")
        mode = "remaining_tasks"
    unfinished = snapshot.unfinished_task_ids
    return SnapshotSelection(snapshot=snapshot, mode=mode,
        target_remaining_tasks=remaining_tasks,
        T_plan_s=replay.plan.total_tardiness_s,
        T_ref_s=math.fsum(replay.plan.tardiness_s[task_id] for task_id in unfinished))
