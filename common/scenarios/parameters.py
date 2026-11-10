from copy import deepcopy
from common.config import DISRUPTION_DEFAULTS
from collections.abc import Mapping
from dataclasses import fields
from fractions import Fraction

from common.initial_sequences.inputs import _nonnegative, _number
from common.scenarios.event_sampling import EventSamplingConfig
from common.scenarios.disruptions import KINDS


def _project_defaults():
    return deepcopy(DISRUPTION_DEFAULTS)


def _mix_allocation(supplied, events):

    if not isinstance(supplied, Mapping) or set(supplied) != {"count", "weights"}:
        raise ValueError("mix requires count and weights")
    count, weights = supplied["count"], supplied["weights"]
    if type(count) is not int or count < 0:
        raise ValueError("mix.count must be a nonnegative integer")
    kinds = [values.get("kind") for values, _ in events]
    if any(not isinstance(kind, str) or kind not in KINDS for kind in kinds):
        raise ValueError("mix events require known event kinds")
    if len(kinds) != len(set(kinds)):
        raise ValueError("mix requires one event recipe per kind")
    if not isinstance(weights, Mapping) or set(weights) - set(kinds):
        raise ValueError("mix.weights contains an undeclared event kind")
    normalized = {kind: _nonnegative(weights.get(kind, 0.), "mix.weights." + kind) for kind in kinds}
    exact = [Fraction(str(normalized[kind])) for kind in kinds]
    total = sum(exact, Fraction(0))
    if count and not total:
        raise ValueError("positive mix.count requires a positive event weight")
    shares = [Fraction(count) * weight / total for weight in exact] if total else [Fraction(0)] * len(kinds)
    allocated = [share.numerator // share.denominator for share in shares]
    priority = sorted(range(len(kinds)), key=lambda i: (-(shares[i] - allocated[i]), i))
    for index in priority[:count - sum(allocated)]:
        allocated[index] += 1
    return {"count": count, "weights": normalized}, allocated


def normalize_scenario_config(record=None) -> dict:


    defaults = _project_defaults()
    if record is None:
        record = defaults
    if not isinstance(record, Mapping) or set(record) - {"events", "mix", "recovery"}:
        raise ValueError("unknown scenario configuration fields")
    supplied_events = record.get("events", defaults["events"])
    if not isinstance(supplied_events, (tuple, list)):
        raise ValueError("scenario events must be a list")
    allowed = {field.name for field in fields(EventSamplingConfig)} - {"seed"}
    declared = []
    for index, supplied in enumerate(supplied_events):
        if not isinstance(supplied, Mapping) or set(supplied) - allowed - {"seed_offset"}:
            raise ValueError("unknown scenario event fields; use seed_offset rather than seed")
        values = dict(supplied)
        offset = values.pop("seed_offset", index)
        if type(offset) is not int:
            raise ValueError("event seed_offset must be an integer")
        declared.append((values, offset))
    mix = record.get("mix", defaults.get("mix") if "events" not in record else None)
    allocation = None
    if mix is not None:
        mix, allocation = _mix_allocation(mix, declared)
    events = []
    for index, (values, offset) in enumerate(declared):
        if allocation is not None:
            if values.get("fraction") is not None:
                raise ValueError("mix cannot be combined with per-kind fraction")
            if values.get("count") is not None and (
                    type(values["count"]) is not int or values["count"] < 0):
                raise ValueError("event count must be a nonnegative integer")
            values["count"] = allocation[index]
        if values.get("count") is None and values.get("fraction") is None:
            values["count"] = 1
        try:
            config = EventSamplingConfig(seed=0, **values).to_dict()
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid scenario event {index}: {error}") from error
        config.pop("seed")
        events.append({**config, "seed_offset": offset})
    result = {"events": events}
    if mix is not None:
        result["mix"] = mix
    supplied = record.get("recovery", {})
    if not isinstance(supplied, Mapping) or set(supplied) - set(defaults["recovery"]):
        raise ValueError("unknown scenario recovery fields")
    recovery = {**defaults["recovery"], **supplied}
    for name, value in recovery.items():
        recovery[name] = (_number(value, "recovery." + name, positive=True)
                          if name == "transfer_duration_s" else _nonnegative(value, "recovery." + name))
    result["recovery"] = recovery
    return result


def scenario_event_configs(config=None, seed=0) -> tuple[EventSamplingConfig, ...]:
    if type(seed) is not int:
        raise ValueError("scenario seed must be an integer")
    record = normalize_scenario_config(config)
    result = []
    for event in record["events"]:
        values = dict(event)
        offset = values.pop("seed_offset")
        result.append(EventSamplingConfig(seed=seed + offset, **values))
    return tuple(result)
