from dataclasses import dataclass
import json
from pathlib import Path

from common.execution.replay import PlanReplay
from common.execution.state import ExecutionSnapshot
from common.initial_sequences.inputs import _plain
from common.initial_sequences.io import _invalid_constant, _unique_pairs
from common.recovery.inputs import MeetingDomain, RecoveryInput, RecoverySettings
from common.recovery.io import load_recovery_input, save_recovery_input
from common.scenarios.parameters import normalize_scenario_config, scenario_event_configs
from common.scenarios.disruptions import DisruptionError, apply_disruptions
from common.scenarios.event_sampling import _count, sample_events


class ScenarioConstructionError(ValueError):
    def __init__(self, stage, reason, message, *, details=None):
        super().__init__(message)
        self.stage, self.reason = stage, reason
        self.details = {} if details is None else _plain(details)

    def to_dict(self):
        return dict(status="construction_failed", stage=self.stage, reason=self.reason,
                    message=str(self), details=self.details)


def _sampling_summary(config, batch):

    details = _plain(batch.details)
    compact = {key: value for key, value in details.items()
               if key not in {"config", "ranked_arc_ids", "unreachable_arc_ids", "remaining_portions"}}
    if "unreachable_arc_ids" in details:
        compact["unreachable_arc_count"] = len(details["unreachable_arc_ids"])
    return dict(config=config.to_dict(), status=batch.status, **compact)


@dataclass(frozen=True)
class ScenarioBuild:
    item: RecoveryInput
    sampling: tuple[dict, ...]
    config: dict
    seed: int

    def summary(self):
        return dict(time_s=self.item.time_s, input_id=self.item.instance_id,
                    seed=self.seed, config=self.config,
                    event_count=len(self.item.events), sampling=list(self.sampling))


def _settings(snapshot, config):
    recovery = config["recovery"]
    return RecoverySettings(
        handover_ranges_m={w.worker_id: recovery["handover_range_m"] for w in snapshot.workers},
        transfer_duration_s=recovery["transfer_duration_s"],
        meeting_domain=MeetingDomain(all_road_positions=True))


def build_scenario(replay, snapshot, config=None, *, seed=0, instance_id=None) -> ScenarioBuild:


    if not isinstance(replay, PlanReplay) or not isinstance(snapshot, ExecutionSnapshot):
        raise TypeError("expected a PlanReplay and ExecutionSnapshot")
    if replay.at(snapshot.time_s) != snapshot:
        raise ValueError("scenario snapshot differs from its parent replay")
    record = normalize_scenario_config(config)
    configs = scenario_event_configs(record, seed)
    events, sampling = [], []
    for event_config in configs:
        batch = sample_events(replay, snapshot, event_config)
        evidence = _sampling_summary(event_config, batch)
        sampling.append(evidence)
        if batch.status != "sampled" or len(batch.events) != batch.details["requested_count"]:
            reason = batch.details.get("reason") or "incomplete_event_batch"
            raise ScenarioConstructionError("sampling", reason,
                "Disruption sample unavailable: " + event_config.kind,
                details=dict(kind=event_config.kind, sampling=sampling))
        events.extend(batch.events)
    try:
        item = apply_disruptions(snapshot, replay.routing.roads, events, _settings(snapshot, record),
                                 instance_id=instance_id)
    except DisruptionError as error:
        raise ScenarioConstructionError("injection", error.code, str(error),
            details=dict(injection=error.to_dict(), sampling=sampling)) from error
    return ScenarioBuild(item, tuple(sampling), record, seed)


def save_scenario(result: ScenarioBuild, output_dir) -> None:

    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("scenario output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    save_recovery_input(result.item, directory / "recovery_input.json")
    (directory / "summary.json").write_text(json.dumps(result.summary(), ensure_ascii=False,
        sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def load_scenario(output_dir, replay, snapshot) -> ScenarioBuild:

    directory = Path(output_dir)
    summary = json.loads((directory / "summary.json").read_text(),
                         parse_constant=_invalid_constant, object_pairs_hook=_unique_pairs)
    record = normalize_scenario_config(summary["config"])
    item = load_recovery_input(directory / "recovery_input.json", snapshot, replay.routing.roads)
    return ScenarioBuild(item, tuple(summary.get("sampling", ())), record, summary["seed"])
