from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from common.recovery.plan import EvaluatedRecoveryPlan, RecoveryPlan, ScheduledRecoveryVisit, _total


@dataclass(frozen=True)
class _CandidateEvaluation:
    plan: RecoveryPlan
    routes: Mapping[str, tuple[ScheduledRecoveryVisit, ...]]
    tardiness_s: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, 'routes', MappingProxyType(dict(self.routes)))
        object.__setattr__(self, 'tardiness_s', MappingProxyType(dict(self.tardiness_s)))


        for name in ('total_tardiness_s', 'total_travel_time_s', 'total_wait_time_s', 'total_distance_m'):
            getattr(self, name)

    @cached_property
    def total_tardiness_s(self):
        return _total(self.tardiness_s.values(), 'evaluated.total_tardiness_s')

    @cached_property
    def total_travel_time_s(self):
        return _total((s.incoming_path.duration_s for route in self.routes.values() for s in route),
                      'evaluated.total_travel_time_s')

    @cached_property
    def total_wait_time_s(self):
        return _total((s.waiting_time_s for route in self.routes.values() for s in route),
                      'evaluated.total_wait_time_s')

    @cached_property
    def total_distance_m(self):
        return _total((s.incoming_path.length_m for route in self.routes.values() for s in route),
                      'evaluated.total_distance_m')

    @property
    def transfer_count(self):
        return len(self.plan.transfers)

    @property
    def late_task_count(self):
        return sum(t > 0 for t in self.tardiness_s.values())

    @cached_property
    def completion_times_s(self):
        return MappingProxyType({s.visit.task_id: s.completion_time_s
            for route in self.routes.values() for s in route if s.visit.operation == 'delivery'})

    @cached_property
    def transfer_start_times_s(self):
        return MappingProxyType({s.visit.transfer_id: s.start_time_s
            for route in self.routes.values() for s in route if s.visit.operation == 'transfer'})

    @cached_property
    def transfer_completion_times_s(self):
        return MappingProxyType({s.visit.transfer_id: s.completion_time_s
            for route in self.routes.values() for s in route if s.visit.operation == 'transfer'})

    @cached_property
    def _materialized(self):
        return EvaluatedRecoveryPlan(plan=self.plan, routes=self.routes, tardiness_s=self.tardiness_s)

    def materialize(self):
        return self._materialized
