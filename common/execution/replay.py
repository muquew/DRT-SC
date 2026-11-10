import math

from common.data.roads.routing import Routing
from common.evaluation.feasibility import evaluate_generation
from common.initial_sequences.plan import InitialPlan
from common.initial_sequences.inputs import PlanningInput
from common.initial_sequences.inputs import _number

from common.execution.state import ExecutionSnapshot, TaskExecutionState, TravelProgress, ServiceProgress, WorkerExecutionState


class PlanReplay:


    def __init__(self, planning_input: PlanningInput, plan: InitialPlan, routing: Routing):
        if not isinstance(planning_input, PlanningInput) or not isinstance(plan, InitialPlan):
            raise TypeError("expected PlanningInput and InitialPlan")
        if not isinstance(routing, Routing):
            raise TypeError("expected Routing on the original road snapshot")
        if (plan.input_fingerprint != planning_input.fingerprint
                or plan.network_id != planning_input.network_id
                or plan.snapshot_id != planning_input.snapshot_id):
            raise ValueError("plan belongs to a different input or road snapshot")
        if plan._validated_for == (planning_input.fingerprint, routing.roads.snapshot_id):
            verified = plan
        else:
            verified = evaluate_generation(planning_input, plan.generation_result, routing)
            if verified != plan:
                raise ValueError("stored plan differs from independently evaluated routes and costs")
        self.planning_input, self.plan, self.routing = planning_input, verified, routing
        self._service = (set(planning_input.required_task_ids)
                         | set(verified.generation_result.accepted_task_ids))
        self._input_fingerprint = planning_input.fingerprint
        self._plan_fingerprint = verified.fingerprint
        self._last_snapshot = None

    def at(self, time_s: float) -> ExecutionSnapshot:

        time_s = _number(time_s, "replay.time_s")
        source = self.planning_input
        if time_s < source.planning_time_s:
            raise ValueError("replay time precedes the planning time")
        if self._last_snapshot is not None and self._last_snapshot.time_s == time_s:
            return self._last_snapshot
        pickups, deliveries, holders, workers = {}, {}, {}, []
        for worker in source.workers:
            route = self.plan.routes[worker.id]
            onboard = set(worker.carried_task_ids)
            location = self.routing.roads.normalize_position(worker.location)
            completed = 0
            for scheduled in route:
                if scheduled.completion_time_s > time_s:
                    break
                visit = scheduled.visit
                if visit.operation == "pickup":
                    onboard.add(visit.task_id)
                    pickups[visit.task_id] = scheduled.completion_time_s
                else:
                    onboard.remove(visit.task_id)
                    deliveries[visit.task_id] = scheduled.completion_time_s
                location = scheduled.position
                completed += 1

            travel = service = None
            if time_s < worker.available_time_s:
                phase = "not_started"
            elif completed == len(route):
                phase = "idle"
            else:
                scheduled = route[completed]
                if time_s < scheduled.arrival_time_s:
                    phase = "travelling"
                    departure = (worker.available_time_s if completed == 0
                                 else route[completed - 1].completion_time_s)
                    travel = TravelProgress(path=scheduled.incoming_path,
                                            departure_time_s=departure, time_s=time_s)
                    location = self.routing.position_after(travel.path, travel.elapsed_time_s)
                else:
                    phase = "servicing" if time_s >= scheduled.start_time_s else "waiting"
                    location = scheduled.position
                    if phase == "servicing":
                        service = ServiceProgress(visit=scheduled.visit, start_time_s=scheduled.start_time_s,
                            completion_time_s=scheduled.completion_time_s, time_s=time_s)
            for identifier in onboard:
                holders[identifier] = worker.id
            workers.append(WorkerExecutionState(
                worker_id=worker.id, location=location, capacity=worker.capacity, profile=worker.profile,
                available_time_s=max(time_s, worker.available_time_s), phase=phase,
                carried_task_ids=tuple(sorted(onboard)),
                load=math.fsum(source.task_map[i].load for i in sorted(onboard)),
                completed_visit_count=completed,
                remaining_visits=tuple(v.visit for v in route[completed:]),
                fixed_prefix_length=max(0, worker.fixed_prefix_length - completed), travel=travel, service=service,
            ))
        tasks = tuple(TaskExecutionState(
            task=task,
            status=("delivered" if task.id in deliveries else
                    "carried" if task.id in holders else "awaiting_pickup"),
            holder_id=holders.get(task.id), pickup_time_s=pickups.get(task.id),
            delivery_time_s=deliveries.get(task.id),
        ) for task in source.tasks if task.id in self._service)
        snapshot = ExecutionSnapshot(
            instance_id=source.instance_id, input_fingerprint=self._input_fingerprint,
            plan_fingerprint=self._plan_fingerprint, network_id=source.network_id,
            snapshot_id=source.snapshot_id, time_s=time_s, time_origin_s=source.time_origin_s,
            load_unit=source.load_unit, tasks=tasks, workers=tuple(workers),
        )
        self._last_snapshot = snapshot
        return snapshot

    def continue_original(self, snapshot: ExecutionSnapshot) -> dict[str, float]:


        if not isinstance(snapshot, ExecutionSnapshot):
            raise TypeError("expected ExecutionSnapshot")
        if snapshot is not self._last_snapshot and self.at(snapshot.time_s) != snapshot:
            raise ValueError("snapshot differs from the replayed parent plan")
        deliveries = {}
        for state in snapshot.workers:
            time_s = state.available_time_s
            remaining = self.plan.routes[state.worker_id][state.completed_visit_count:]
            for index, scheduled in enumerate(remaining):
                if index == 0 and state.phase == "servicing":
                    time_s = scheduled.completion_time_s
                    if scheduled.visit.operation == "delivery":
                        deliveries[scheduled.visit.task_id] = time_s
                    continue
                if index == 0 and state.phase == "travelling":


                    arrival = state.travel.departure_time_s + state.travel.path.duration_s
                elif index == 0 and state.phase == "waiting":
                    arrival = time_s
                else:
                    arrival = time_s + scheduled.incoming_path.duration_s
                arrival = _number(arrival, "continued arrival time")
                task = self.planning_input.task_map[scheduled.visit.task_id]
                ready = (task.pickup_ready_time_s if scheduled.visit.operation == "pickup"
                         else task.delivery_ready_time_s)
                time_s = max(arrival, ready) + task.service_duration_s(scheduled.visit.operation)
                if scheduled.visit.operation == "delivery":
                    deliveries[task.id] = time_s
        return deliveries
