from dataclasses import dataclass

from common.data.roads.routing import Routing
from common.recovery.inputs import RecoveryInput


def recovery_execution_policy() -> dict:

    return dict(all_remaining_service_confirmed=True, node_waiting_permitted=True,
        service_point_waiting_permitted=True, gate_duration_s=0., holding_duration_s=None,
        deterministic_current_arc_continuation=True,
        model_evidence="shared recovery input adapter; continue the snapshot's started actions")


class RecoveryPreparationError(ValueError):


    stage = "solver_input_adaptation"

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason

    def to_dict(self):
        return dict(status="adaptation_failed", stage=self.stage,
                    reason=self.reason, message=str(self))


@dataclass(frozen=True)
class PreparedRecovery:
    item: RecoveryInput
    projected: object
    routing: Routing
    boundary: object
    diagnostics: dict | None = None
    diagnostics_error: dict | None = None


def prepare_recovery_input(item, routing=None, *, replay=None) -> PreparedRecovery:


    if not isinstance(item, RecoveryInput):
        raise TypeError("expected a RecoveryInput")
    if getattr(item, "execution_boundary", None) is not None:
        raise ValueError("solver preparation requires the original disturbed input")
    for worker in item.source_snapshot.workers:
        if worker.available_time_s > item.time_s:
            raise RecoveryPreparationError("unsupported_execution_state",
                f"worker {worker.worker_id}: future availability is not supported by the solver adapter")
        if worker.fixed_prefix_length:
            raise RecoveryPreparationError("unsupported_execution_state",
                f"worker {worker.worker_id}: fixed remaining visits are not supported by the solver adapter")
    if routing is None:
        from common.config import routing_options
        routing = Routing(item.roads, **routing_options("formal"))
    if routing.roads != item.roads:
        raise ValueError("solver preparation routing differs from the disturbed roads")
    routing.register_query_input(item)

    from solvers.tfr.interfaces.boundary import BoundaryUncertified
    from solvers.tfr.interfaces.offline import prepare_frozen_snapshot_boundary

    try:
        projected = prepare_frozen_snapshot_boundary(item, routing,
            policy=recovery_execution_policy(),
            withdrawal_deadlines={w.id: None for w in item.workers if w.service_mode == "handover_only"})
    except BoundaryUncertified as error:
        raise RecoveryPreparationError("unsupported_execution_state", str(error)) from error
    boundary = projected.execution_boundary
    diagnostics = diagnostics_error = None
    if replay is not None:
        from common.scenarios.diagnostics import scenario_diagnostics
        try:
            diagnostics = scenario_diagnostics(replay, item, projected=projected, routing=routing)
        except MemoryError:
            raise
        except Exception as error:
            diagnostics_error = dict(status="unavailable", stage="diagnostics",
                reason=type(error).__name__, message=str(error))
    return PreparedRecovery(item, projected, routing, boundary, diagnostics, diagnostics_error)
