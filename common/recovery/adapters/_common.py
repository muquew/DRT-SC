from collections import OrderedDict
from dataclasses import dataclass, replace
import math
import random
from time import perf_counter

from common.evaluation.recovery_plan import evaluate_recovery_plan, RecoveryPlanValidationError
from common.evaluation._recovery_cache import RecoveryEvaluationCache
from common.config import cache_limits
from common.cache_telemetry import bypass
from common.initial_sequences.inputs import _exceeds_capacity
from common.recovery.plan import RecoveryPlan, RecoveryVisit, TransferEvent
from common.recovery.observation import cpu_time


class BudgetExhausted(RuntimeError):
    pass


class ObjectiveLowerBoundReached(RuntimeError):


    pass


def score(result):
    return result.total_tardiness_s, result.total_travel_time_s, result.transfer_count


def position_key(position):
    return position.node_id or '', position.segment_id or '', position.offset_m or 0.0


@dataclass(frozen=True)
class InsertionOption:
    plan: RecoveryPlan
    evaluation: object
    score: tuple
    worker_id: str
    selection_cost: float | tuple[float, float] | None = None

    def __post_init__(self):
        if self.selection_cost is None:
            object.__setattr__(self, 'selection_cost', self.score[0])


def strip(plan, removed):

    removed = set(removed)
    events = []
    for event in plan.transfers:
        sent = tuple(t for t in event.tasks_i_to_j if t not in removed)
        returned = tuple(t for t in event.tasks_j_to_i if t not in removed)
        if sent or returned:
            events.append(replace(event, tasks_i_to_j=sent, tasks_j_to_i=returned))
    surviving = {e.id for e in events}
    sequences = {w: tuple(v for v in visits if
                         (v.operation == 'transfer' and v.transfer_id in surviving)
                         or (v.operation == 'waypoint' and v.for_task_id not in removed)
                         or (v.operation not in ('transfer','waypoint') and v.task_id not in removed))
                 for w, visits in plan.sequences.items()}
    return replace(plan, sequences=sequences, transfers=tuple(events))


class SearchEngine:
    def __init__(self, item, routing, config, *, started=None, candidate_evaluator='incremental',evaluation_cache=None,local_context=None):
        if candidate_evaluator not in ('native', 'incremental'):
            raise ValueError('candidate_evaluator must be native or incremental')
        self.candidate_evaluator = candidate_evaluator
        self.item, self.routing, self.config = item, routing, config
        routing.register_query_input(item)
        self.rng = random.Random(config.seed)
        self.started = perf_counter() if started is None else started
        self.best = self.seed = self.policy_result = None
        self.observer = None
        self._objective_zero_witness = None
        self._objective_zero_plan = None
        self.initialization_cpu_s = 0.0
        self._initialization_cpu_started = None
        self._initialization_wall_started = None
        self._phase = None
        self.phase = 'search'
        self.telemetry_started = self.started
        self.initialization_s = 0.0
        self.incumbent_trace = []
        self.stats = {'evaluations': 0, 'complete_evaluations': 0, 'rejections': {},
                      'generated_moves': 0, 'blink_skips': 0, 'iterations_started': 0, 'iterations': 0,
                      'first_complete_runtime_s': None,
                      'options_calls': 0, 'repair_rounds': 0,
                      'projection_hits': 0, 'projection_misses': 0, 'projection_evictions': 0}
        self.stats.update(meeting_cache_hits=0,meeting_cache_misses=0,meeting_cache_evictions=0,
            task_distance_cache_hits=0,task_distance_cache_misses=0,sender_position_cache_hits=0,sender_position_cache_misses=0)
        self._memo_bypasses={name:dict(bypasses=0,bypass_reasons={}) for name in ('input_projection','meeting_pair')}
        self.stats['candidate_domain'] = dict(worker_limit=None, meeting_limit=None,
            enumeration='all eligible workers; lazy insertion/transfer plans',
            meeting_scope='permitted road nodes in sender range plus declared physical anchors and path fractions',
            continuous_optimality_claim=False)
        self.holders = {t: w.id for w in item.workers for t in w.carried_task_ids}
        self.transfer_policy = getattr(config, 'effective_transfer_policy', 'candidate_points')
        self.all_tasks = frozenset(item.task_map)
        from common.evaluation._recovery_cache import shared_cache
        self.evaluation_cache = evaluation_cache or shared_cache(routing) or RecoveryEvaluationCache(
            routing,route_key_reuse=getattr(config, 'route_key_reuse', False))
        self.evaluation_cache.check_routing(routing)
        self.memo_limits = cache_limits('memo')
        self._projections = OrderedDict()
        self._positions = OrderedDict()
        self._road_node_positions = {}
        self._facility_positions = None
        self.meeting_pool = getattr(config, 'meeting_pool', 'legacy')
        if self.meeting_pool == 'road_nodes':
            self.stats['meeting_pool'] = 'road_nodes'
            self.stats['meeting_node_sources'] = {}
            self.stats['meeting_node_generation_wall_s'] = 0.0
            self.stats['meeting_node_generation_cpu_s'] = 0.0
        self._distances = {}
        self._initialized = False
        self._selected_evaluation = None
        self._base_positions = self._meeting_pool() if self.transfer_policy == 'candidate_points' else ()
        from common.recovery.local import LocalRecoveryContext
        if local_context is not None and (local_context.item is not item
                or local_context.routing is not routing.for_activity_domain(None)):
            raise ValueError('local context belongs to another input or routing session')
        self.local = local_context or LocalRecoveryContext(item, routing,evaluation_cache=self.evaluation_cache)
        self.stats['local_scope'] = self.local.stats
        self.stats['candidate_domain'] = dict(worker_limit=config.worker_candidates, joint_insertions_per_worker=8,
            generated_branches_per_worker=64, enumeration='spatial rings; expand after no feasible insertion',
            continuous_optimality_claim=False)


    def memo_cache_snapshot(self):
        from solvers.tfr.interfaces.boundary import worker_cache_snapshot
        rows={'worker_rebuild':worker_cache_snapshot()}
        for name,prefix,cache,limit in (
            ('input_projection','projection',self._projections,self.memo_limits['projection_entries']),
            ('meeting_pair','meeting_cache',self._positions,self.memo_limits['meeting_entries']),
            ('task_distance','task_distance_cache',self._distances,None),
            ('sender_position','sender_position_cache',self._road_node_positions,None)):
            row=dict(hits=self.stats[prefix+'_hits'],misses=self.stats[prefix+'_misses'],entries=len(cache),
                max_entries=limit,clears=None,reset_scope='new SearchEngine instance; no explicit clear path',scope='one immutable input/SearchEngine')
            if limit is None:
                row.update(evictions=None,bypasses=None,bypass_reasons={},
                    mechanism_reasons=dict(evictions='not_applicable: input-keyed memo has no eviction branch',
                        bypasses='not_applicable: every computed value is retained; no admission rejection',
                        max_entries='not_applicable: no configured cap; keyed by task or sender within one input'))
            else:row.update(self._memo_bypasses[name],bypass_reasons=dict(self._memo_bypasses[name]['bypass_reasons']),evictions=self.stats[prefix+'_evictions'])
            rows[name]=row
        return rows

    def metrics_snapshot(self):
        return dict(statistics=self.stats,phase=self.phase,cache=dict(evaluation=dict(self.evaluation_cache.stats),
            routing=dict(self.routing.cache_stats),memo=self.memo_cache_snapshot()))

    def objective_key(self, evaluated, *, task_ids=None):
        return evaluated.total_tardiness_s

    @staticmethod
    def objective_improves(candidate_key, incumbent_key):
        return candidate_key < incumbent_key

    def placement_score(self, evaluated, *, task_ids):
        if self.config.objective_mode == 'total_tardiness':
            return evaluated.total_tardiness_s, 0, 0
        return score(evaluated)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):


        previous = self._phase
        if value == previous:
            return
        if previous == 'initialization':
            self.initialization_cpu_s += cpu_time() - self._initialization_cpu_started
            self._initialization_cpu_started = None
        if value == 'initialization':
            self._initialization_cpu_started = cpu_time()
        self._phase = value

    def check_objective_bound(self):
        if self._objective_zero_witness is not None:
            raise ObjectiveLowerBoundReached('objective_lower_bound')

    def check_budget(self):
        if self._objective_zero_witness is not None:
            raise ObjectiveLowerBoundReached('objective_lower_bound')
        if self.config.max_evaluations is not None and self.stats['evaluations'] >= self.config.max_evaluations:
            raise BudgetExhausted('evaluation_limit')

    def empty_plan(self):
        return RecoveryPlan(input_id=self.item.instance_id, network_id=self.item.network_id,
                            snapshot_id=self.item.snapshot_id,
                            sequences={w.id: () for w in self.item.workers})

    strip = staticmethod(strip)

    def solution_key(self, plan, bank=()):

        slots = {event.id: {} for event in plan.transfers}
        for worker, visits in plan.sequences.items():
            for index, visit in enumerate(visits):
                if visit.operation == 'transfer':
                    slots[visit.transfer_id][worker] = index
        events = {}
        for event in plan.transfers:
            first, second = sorted((event.worker_i, event.worker_j))
            outgoing, incoming = ((event.tasks_i_to_j, event.tasks_j_to_i) if first == event.worker_i
                                  else (event.tasks_j_to_i, event.tasks_i_to_j))
            events[event.id] = (first, second,
                position_key(self.item.roads.normalize_position(event.location)),
                tuple(sorted(outgoing)), tuple(sorted(incoming)),
                tuple(sorted(slots[event.id].items())))
        routes = tuple((worker, tuple((v.operation, events[v.transfer_id] if v.operation == 'transfer'
                                      else (position_key(v.location), v.not_before_s, v.for_task_id) if v.operation == 'waypoint'
                                      else v.task_id) for v in visits))
                       for worker, visits in sorted(plan.sequences.items()))
        return routes, tuple(sorted(bank))

    def remember(self, result, *, source='search', publish=True):

        if any(v.operation == 'waypoint' for vs in result.plan.sequences.values() for v in vs):
            raise ValueError('native travel-control plans must be projected before recovery publication')
        if result.tardiness_s.keys() != self.all_tasks:
            raise ValueError('only complete evaluated plans can be retained')
        self.require_transfer_policy(result.plan)
        better = self.best is None or result.total_tardiness_s < self.best.total_tardiness_s
        if better:
            zero = (getattr(self.config, 'objective_zero_stopping', False)
                    and publish and result.total_tardiness_s == 0)
            validation_wall = validation_cpu = 0.0
            if zero:


                validation_started, validation_cpu_started = perf_counter(), cpu_time()
                verified = evaluate_recovery_plan(self.item, result.plan, self.routing)
                validation_wall = perf_counter() - validation_started
                validation_cpu = cpu_time() - validation_cpu_started
                if verified.total_tardiness_s != 0:
                    raise ValueError('zero objective claim differs from full physical validation')
                result = verified
            elapsed = perf_counter() - self.telemetry_started
            first = self.best is None
            if first:
                self.stats['first_complete_runtime_s'] = elapsed
            self.best = result
            self.stats['incumbent_updates'] = self.stats.get('incumbent_updates', 0) + 1
            self.incumbent_trace.append({
                'elapsed_s': elapsed, 'evaluations': self.stats['evaluations'],
                'available_perf_s': self.telemetry_started + elapsed,
                'phase': self.phase, 'source': source,
                'reason': 'first_complete' if first else 'improvement',
                'total_tardiness_s': result.total_tardiness_s,
                'total_travel_time_s': result.total_travel_time_s,
                'transfer_count': result.transfer_count,
            })
            if publish and self.observer is not None:
                self.observer.publish(self.item, self.routing, result, source=source,
                                      phase=self.phase, solver_elapsed_s=elapsed)
            if zero:


                self._objective_zero_plan = result.plan
                self._objective_zero_witness = dict(
                    objective='total_tardiness_s', lower_bound=0.0, feasible_cost=0.0,
                    scope='global lower-bound attainment for the public nonnegative floating-point objective',
                    rational_exact_optimality_certified=False, full_input_validated=True,
                    input_id=self.item.instance_id, network_id=self.item.network_id,
                    snapshot_id=self.item.snapshot_id, phase=self.phase, source=source,
                    witness_recorded_solver_elapsed_s=perf_counter()-self.telemetry_started,
                    validation_wall_s=validation_wall, validation_cpu_s=validation_cpu)

    def commit_policy(self, result, *, source='policy'):

        self.require_transfer_policy(result.plan)
        verified = evaluate_recovery_plan(self.item, result.plan, self.routing)
        self.policy_result = verified
        self.remember(verified, source=source, publish=False)
        elapsed = perf_counter() - self.telemetry_started
        self.stats['output_semantics'] = 'final_policy_state'
        self.stats.setdefault('policy_trace', []).append({
            'elapsed_s': elapsed, 'source': source, 'evaluations': self.stats['evaluations'],
            'total_tardiness_s': verified.total_tardiness_s,
            'total_travel_time_s': verified.total_travel_time_s,
            'transfer_count': verified.transfer_count})
        if self.observer is not None:
            self.observer.publish(self.item, self.routing, verified, source=source,
                                  phase='policy_decision', solver_elapsed_s=elapsed)
        return verified

    def _project(self, included):
        if included == self.all_tasks:
            bypass(self._memo_bypasses['input_projection'],'full_input_identity')
            return self.item
        if not included <= self.all_tasks:
            raise ValueError('projection contains unknown tasks')
        if included in self._projections:
            self.stats['projection_hits'] += 1
            self._projections.move_to_end(included)
            return self._projections[included]
        self.stats['projection_misses'] += 1
        if getattr(self.item, 'execution_boundary', None) is not None:
            from solvers.tfr.interfaces.boundary import project_subset
            projected = project_subset(self.item, included)
            self._projections[included] = projected
            if len(self._projections) > self.memo_limits['projection_entries']:
                self._projections.popitem(last=False)
                if self.memo_limits['projection_entries']:self.stats['projection_evictions'] += 1
                else:bypass(self._memo_bypasses['input_projection'],'disabled')
            return projected
        source = self.item.source_snapshot
        states = tuple(s for s in source.tasks if s.task.id in included)
        workers, source_workers, sequences = [], [], {}
        for worker in self.item.workers:
            cargo = tuple(t for t in worker.carried_task_ids if t in included)
            visits = tuple(v for v in self.item.reference_sequences[worker.id] if v.task_id in included)
            workers.append(replace(worker, carried_task_ids=cargo))
            source_workers.append(replace(source.worker_map[worker.id], carried_task_ids=cargo,
                load=math.fsum(self.item.task_map[t].load for t in cargo), remaining_visits=visits,
                phase='waiting' if visits else 'idle', travel=None))
            sequences[worker.id] = visits
        snapshot = replace(source, tasks=states, workers=tuple(source_workers))
        projected = replace(self.item, source_snapshot=snapshot, workers=tuple(workers),
                            tasks=tuple(t for t in self.item.tasks if t.id in included),
                            reference_sequences=sequences)
        self._projections[included] = projected
        if len(self._projections) > self.memo_limits['projection_entries']:
            self._projections.popitem(last=False)
            if self.memo_limits['projection_entries']:self.stats['projection_evictions'] += 1
            else:bypass(self._memo_bypasses['input_projection'],'disabled')
        return projected

    def evaluate(self, plan, included=None):
        self.check_budget()
        included = self.all_tasks if included is None else frozenset(included)
        self.stats['evaluations'] += 1
        try:
            result = self._evaluate_candidate(self._project(included), plan)
        except RecoveryPlanValidationError as error:
            if error.code not in {'custody', 'capacity', 'worker_withdrawal', 'meeting_domain',
                                  'handover_range', 'transfer_cycle', 'unreachable', 'remaining_custody',
                                  'service_range', 'activity_domain'}:
                raise
            rejected = self.stats['rejections']
            rejected[error.code] = rejected.get(error.code, 0) + 1
            return None
        if self.transfer_policy_violation(plan) is not None:
            rejected = self.stats['rejections']
            rejected['transfer_policy'] = rejected.get('transfer_policy', 0) + 1
            return None
        if included == self.all_tasks:
            self.stats['complete_evaluations'] += 1
        return result

    def _evaluate_candidate(self, item, plan):
        if getattr(item, 'execution_boundary', None) is not None:
            from solvers.tfr.interfaces.boundary import PartialBoundaryRecoveryInput
            if isinstance(item, PartialBoundaryRecoveryInput):
                from common.evaluation.recovery_plan import _evaluate_recovery_plan
                return _evaluate_recovery_plan(item, plan, self.routing, cache=self.evaluation_cache,
                                               allow_partial_boundary=True)
        if getattr(self.config, 'method', None) == 'bouros_sp' and any(v.operation == 'waypoint' for vs in plan.sequences.values() for v in vs):
            from common.evaluation.recovery_plan import _evaluate_recovery_plan
            return _evaluate_recovery_plan(item, plan, self.routing, cache=self.evaluation_cache,
                                           allow_native_waypoints=True)
        return evaluate_recovery_plan(item, plan, self.routing, cache=self.evaluation_cache)

    def visit_location(self, plan, visit):
        if visit.operation == 'waypoint':
            return visit.location
        if visit.operation == 'transfer':
            return plan.transfer_map[visit.transfer_id].location
        task = self.item.task_map[visit.task_id]
        return task.pickup if visit.operation == 'pickup' else task.delivery

    def transfer_policy_violation(self, plan):

        if self.transfer_policy not in ('holder_origin', 'encounter') or not plan.transfers:
            return None
        roads = self.item.roads
        outgoing = {}
        for event in plan.transfers:
            if self.transfer_policy == 'encounter':
                for worker_id in (event.worker_i, event.worker_j):
                    origin = roads.normalize_position(self.item.worker_map[worker_id].location)
                    if roads.normalize_position(event.location) != origin:
                        return 'an encounter requires both workers to be at the event location initially'
                    outgoing.setdefault(worker_id, set()).add(event.id)
            for sender, tasks in ((event.worker_i, event.tasks_i_to_j),
                                  (event.worker_j, event.tasks_j_to_i)):
                if not tasks:
                    continue
                if self.transfer_policy == 'encounter':
                    if any(task not in self.holders for task in tasks):
                        return 'current encounters may transfer only initially carried tasks'
                    continue
                if any(self.holders.get(task) != sender for task in tasks):
                    return 'only initially carried tasks may be handed over by their actual holder'
                origin = roads.normalize_position(self.item.worker_map[sender].location)
                if roads.normalize_position(event.location) != origin:
                    return 'handover location must be the initial holder position'
                outgoing.setdefault(sender, set()).add(event.id)
        for sender, events in outgoing.items():
            origin = roads.normalize_position(self.item.worker_map[sender].location)
            for visit in plan.sequences[sender]:
                if roads.normalize_position(self.visit_location(plan, visit)) != origin:
                    return 'sender must stay at its initial position until its outgoing handovers finish'
                if visit.operation == 'transfer':
                    events.discard(visit.transfer_id)
                if not events:
                    break
            if events:
                return 'an outgoing handover is missing from the sender sequence'
        return None

    def require_transfer_policy(self, plan):
        violation = self.transfer_policy_violation(plan)
        if violation is not None:
            raise ValueError(f'transfer_policy={self.transfer_policy}: {violation}')

    def _stationary_prefix_length(self, plan, worker_id):

        roads = self.item.roads
        origin = roads.normalize_position(self.item.worker_map[worker_id].location)
        for index, visit in enumerate(plan.sequences[worker_id]):
            if roads.normalize_position(self.visit_location(plan, visit)) != origin:
                return index
        return len(plan.sequences[worker_id])

    def evaluate_selected(self, plan, included=None):

        included = self.all_tasks if included is None else frozenset(included)
        chosen = self._selected_evaluation
        if chosen is not None and plan is chosen[0] and included == chosen[1]:
            return chosen[2]
        return self.evaluate(plan, included)

    def _meeting_pool(self):
        roads, domain = self.item.roads, self.item.source_settings.meeting_domain
        positions = [w.location for w in self.item.workers]
        for task in self.item.tasks:
            positions.append(task.delivery)
            if task.id not in self.holders:
                positions.append(task.pickup)
        positions.extend(roads.node_position(i) for i in domain.node_ids)
        for portion in domain.segment_portions:
            positions.extend(roads.segment_position(portion.segment_id, x) for x in
                             (portion.start_m, portion.start_m + (portion.end_m - portion.start_m) / 2, portion.end_m))
        return tuple(sorted({roads.normalize_position(p) for p in positions if domain.contains(p, roads)},
                            key=position_key))

    def _sender_road_nodes(self, sender):
        if sender.id not in self._road_node_positions:
            self.stats['sender_position_cache_misses']+=1
            started, cpu_started = perf_counter(), cpu_time()
            roads, domain = self.item.roads, self.item.source_settings.meeting_domain
            distances = self.routing.nodes_within_distance(self.local.original_anchor(sender.id), sender.handover_range_m)
            positions = tuple(roads.node_position(node) for node in distances
                              if domain.contains(roads.node_position(node), roads))
            self._road_node_positions[sender.id] = positions
            self.stats['meeting_node_sources'][sender.id] = {
                'reachable_nodes_in_range': len(distances), 'permitted_nodes_in_range': len(positions)}
            self.stats['meeting_node_generation_wall_s'] += perf_counter() - started
            self.stats['meeting_node_generation_cpu_s'] += cpu_time() - cpu_started
            self.check_budget()
        else:self.stats['sender_position_cache_hits']+=1
        return self._road_node_positions[sender.id]

    def facility_positions(self):

        if self.meeting_pool != 'road_nodes' or self.transfer_policy != 'candidate_points':
            return self._base_positions
        if self._facility_positions is None:
            additions = set()
            for worker in self.item.workers:
                self.check_budget()
                additions.update(self._sender_road_nodes(worker))
            additions.difference_update(self._base_positions)
            self._facility_positions = self._base_positions + tuple(sorted(additions, key=position_key))
            self.stats['meeting_facility_pool'] = {
                'legacy_positions': len(self._base_positions), 'added_road_nodes': len(additions),
                'total_positions': len(self._facility_positions),
                'scope': 'legacy facilities plus permitted road nodes in any sender range'}
        return self._facility_positions

    def meeting_positions(self, sender_id, receiver_id):

        key = sender_id, receiver_id
        if key in self._positions:
            self.stats['meeting_cache_hits'] += 1
            self._positions.move_to_end(key)
            return self._positions[key]
        self.stats['meeting_cache_misses'] += 1
        sender = self.item.worker_map[sender_id]
        origin = self.local.original_anchor(sender_id)
        receiver_origin = self.item.worker_map[receiver_id].location
        pool = {origin, sender.location, self.item.worker_map[receiver_id].location}
        if self.transfer_policy == 'encounter':
            pool = {origin} if origin == receiver_origin else set()
        elif self.transfer_policy != 'holder_origin':


            pool.update(self.item.roads.node_position(node) for node in
                self.routing.nodes_within_distance(origin,sender.handover_range_m))
            for event in self.local.reference_plan.transfers:
                if sender_id in (event.worker_i,event.worker_j):
                    pool.add(event.location)
        self.routing.register_query_positions(pool,metrics=('length_m',))
        result = tuple(sorted((p for p in pool if self.local.meeting_allowed(
            sender_id,receiver_id,p)), key=lambda p: (
                self.routing.shortest_distance(receiver_origin,p),position_key(p)))[:self.config.meeting_candidates])
        self.routing.register_query_positions(result,metrics=('travel_time_s',))
        self._positions[key] = result
        if len(self._positions) > self.memo_limits['meeting_entries']:
            self._positions.popitem(last=False)
            self.stats['meeting_cache_evictions'] += 1
        return result

    def task_features(self, task_id):
        task = self.item.task_map[task_id]
        owner = self.holders.get(task_id)
        origin = self.item.worker_map[owner].location if owner is not None else task.pickup
        return (*self.item.roads.coordinates(origin), *self.item.roads.coordinates(task.delivery),
                task.load, self.item.time_s if owner is not None else task.pickup_ready_time_s,
                task.delivery_ready_time_s, task.promised_delivery_time_s)

    def task_distance(self, task_id):
        if task_id not in self._distances:
            self.stats['task_distance_cache_misses']+=1
            task = self.item.task_map[task_id]
            owner = self.holders.get(task_id)
            origin = self.item.worker_map[owner].location if owner is not None else task.pickup
            distance = self.routing.travel_time(origin, task.delivery)

            self._distances[task_id] = distance
        else:self.stats['task_distance_cache_hits']+=1
        return self._distances[task_id]

    def _workers(self, plan, task_id, allowed=None):
        workers = self.local.eligible_workers(task_id,sender_id=self.holders.get(task_id))
        if allowed is not None:
            workers = tuple(w for w in workers if w in allowed)
        ranked = sorted(workers,key=lambda w:self.local.worker_candidate_priority(
            task_id,w,self.holders.get(task_id)))
        return ranked if self.config.worker_candidates is None else ranked[:self.config.worker_candidates]

    def _proposal(self, plan, sequences, events=None):
        return replace(plan, sequences={**plan.sequences, **sequences},
                       transfers=plan.transfers if events is None else tuple(events))

    def options(self, plan, task_id, included, **kwargs):
        from common.recovery.adapters._local_search import options
        return options(self, plan, task_id, included, **kwargs)

    def repair(self, plan, bank, **kwargs):
        from common.recovery.adapters._local_search import repair
        return repair(self, plan, bank, **kwargs)

    def construct_local(self, **kwargs):
        from common.recovery.adapters._local_search import construct_local
        return construct_local(self, **kwargs)

    def initialize(self, builder=None, *, source='method_constructor'):

        if self._initialized:
            return self.seed
        if builder is not None:
            self.seed = builder(self)
            self._initialized = True
            if self.seed is not None:
                self.remember(self.seed, source=source)
            return self.seed
        return self.construct_local(rule='best')
