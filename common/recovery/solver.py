from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from common.recovery.observation import RecoveryObserver

from common.data.roads.routing import Routing
from common.initial_sequences.inputs import _freeze, _mapping, _number, _plain
from common.recovery.inputs import RecoveryInput
from common.recovery.plan import EvaluatedRecoveryPlan, RecoveryPlan
from common.recovery.observation import cpu_time

METHODS = ('greedy','alns','grasp','rlns','hu_scenario','deco')
POINT_SEARCH_METHODS = METHODS
ZERO_STOP_METHODS = ('greedy', 'rlns')
ROAD_NODE_POOL_METHODS = METHODS


@dataclass(frozen=True, kw_only=True)
class SolverConfig:
    method: str
    seed: int = 0
    iterations: int | None = 100
    time_limit_s: float | None = None
    max_evaluations: int | None = None
    worker_candidates: int | None = None
    meeting_candidates: int = 8
    transfer_policy: str = 'source'
    meeting_pool: str = 'legacy'
    objective_zero_stopping: bool = False
    objective_mode: str = 'native'
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.objective_mode not in ('native', 'total_tardiness'):
            raise ValueError('objective_mode must use total tardiness')
        if self.meeting_pool not in ('legacy', 'road_nodes'):
            raise ValueError('meeting_pool must be legacy or road_nodes')
        if self.meeting_pool == 'road_nodes' and self.method not in ROAD_NODE_POOL_METHODS:
            raise ValueError('road_nodes meeting pool is unsupported for this method')
        if type(self.objective_zero_stopping) is not bool:
            raise ValueError('objective_zero_stopping must be boolean')
        if self.objective_zero_stopping and self.method not in ZERO_STOP_METHODS:
            raise ValueError('objective_zero_stopping is supported for greedy, transfer_alns, rlns and danloup_ga')
        if self.method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if self.transfer_policy not in ('source', 'holder_origin', 'candidate_points'):
            raise ValueError('transfer_policy must be source, holder_origin, or candidate_points')
        if self.transfer_policy == 'holder_origin' and self.method in POINT_SEARCH_METHODS:
            raise ValueError('holder_origin would remove the method\'s native transfer-point search')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if (self.worker_candidates is not None and
                (type(self.worker_candidates) is not int or self.worker_candidates < 1)):
            raise ValueError('worker_candidates must be None or a positive integer')
        for name in ('iterations', 'meeting_candidates'):
            value = getattr(self, name)
            if (name == 'iterations' and value is None and (self.method == 'hu_scenario' or
                    (self.method == 'rlns' and isinstance(self.parameters, Mapping)
                     and self.parameters.get('protocol') == 'paper'))):
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
            if name == 'meeting_candidates' and not 1 <= value <= 8:
                raise ValueError(f'{name} must be in [1,8] for bounded local recovery')
        if self.max_evaluations is not None and (type(self.max_evaluations) is not int or self.max_evaluations < 1):
            raise ValueError('max_evaluations must be positive or None')
        if self.time_limit_s is not None:
            raise ValueError('Algorithm wall-clock limits are not supported')
        object.__setattr__(self, 'parameters', _freeze(_mapping(self.parameters, 'solver.parameters')))

    def to_dict(self):
        values = _plain(self)
        if self.meeting_pool == 'legacy':
            values.pop('meeting_pool')
        if not self.objective_zero_stopping:
            values.pop('objective_zero_stopping')
        if self.objective_mode == 'native':
            values.pop('objective_mode')
        return values

    @property
    def objective(self):
        return 'scalar_total_tardiness'

    @property
    def effective_transfer_policy(self):
        if self.transfer_policy != 'source':
            return self.transfer_policy
        return ('candidate_points' if self.method in POINT_SEARCH_METHODS
                else 'holder_origin')


@dataclass(frozen=True, kw_only=True)
class SolveResult:
    input_id: str
    network_id: str
    snapshot_id: str
    config: SolverConfig
    status: str
    plan: EvaluatedRecoveryPlan | None
    runtime_s: float
    termination_reason: str
    statistics: Mapping[str, Any]
    timing_s: Mapping[str, float] = field(default_factory=dict)
    timing_cpu_s: Mapping[str, float] = field(default_factory=dict)
    incumbent_trace: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'timing_s', _freeze(_mapping(self.timing_s, 'timing_s')))
        object.__setattr__(self, 'timing_cpu_s', _freeze(_mapping(self.timing_cpu_s, 'timing_cpu_s')))
        object.__setattr__(self, 'incumbent_trace', tuple(_freeze(row) for row in self.incumbent_trace))

    def to_dict(self):
        metrics = None
        if self.plan is not None:
            metrics = {name: getattr(self.plan, name) for name in (
                'total_tardiness_s', 'late_task_count', 'total_travel_time_s',
                'total_wait_time_s', 'total_distance_m', 'transfer_count')}
            metrics.update({name: dict(getattr(self.plan, name)) for name in (
                'completion_times_s', 'transfer_start_times_s', 'transfer_completion_times_s')})
        return dict(input_id=self.input_id, network_id=self.network_id, snapshot_id=self.snapshot_id,
                    config=self.config.to_dict(), status=self.status, runtime_s=self.runtime_s,
                    termination_reason=self.termination_reason, statistics=_plain(self.statistics),
                    timing_s=_plain(self.timing_s), timing_cpu_s=_plain(self.timing_cpu_s),
                    incumbent_trace=_plain(self.incumbent_trace),
                    metrics=metrics, global_feasibility=True if self.plan is not None else None)


def solve_recovery(instance, config, routing=None, *, seed_plan=None,
                   observer: 'RecoveryObserver | None' = None, response_deadline_s=None, timing_hook=None):


    telemetry_started, cpu_started = perf_counter(), cpu_time()
    if response_deadline_s is not None:
        raise ValueError('Algorithm wall-clock deadlines are not supported')
    from common.evaluation.path_facts import capture_input_paths,validate_with_input_paths
    from importlib import import_module
    from common.recovery.adapters._common import SearchEngine, BudgetExhausted, ObjectiveLowerBoundReached

    supplied = seed_plan.plan if isinstance(seed_plan, EvaluatedRecoveryPlan) else seed_plan
    if isinstance(supplied, RecoveryPlan) and any(
            visit.operation == 'waypoint' for visits in supplied.sequences.values() for visit in visits):
        raise ValueError('published recovery seeds cannot contain explicit travel waypoints; use a canonical service plan')

    if not isinstance(instance, RecoveryInput) or not isinstance(config, SolverConfig):
        raise TypeError('expected RecoveryInput and SolverConfig')
    if observer is not None:
        observer.begin_solver(instance)
    routing_setup = routing_cpu = 0.0
    if routing is None:
        routing_started, routing_cpu_started = perf_counter(), cpu_time()
        with timing_hook('load_prepare') if timing_hook else nullcontext():
            routing = Routing(instance.roads, cache_profile='formal')
        routing_setup = perf_counter() - routing_started
        routing_cpu = cpu_time() - routing_cpu_started
    if not isinstance(routing, Routing):
        raise TypeError('routing must be Routing')
    if (routing.roads.network_id, routing.roads.snapshot_id) != (instance.network_id, instance.snapshot_id):
        raise ValueError('routing does not match recovery road snapshot')
    input_path_facts=capture_input_paths(routing)
    if seed_plan is not None and not isinstance(seed_plan, (RecoveryPlan, EvaluatedRecoveryPlan)):
        raise TypeError('seed_plan must be a RecoveryPlan or EvaluatedRecoveryPlan')


    started = perf_counter()
    with timing_hook('load_prepare') if timing_hook else nullcontext():
        engine = SearchEngine(instance, routing, config, started=started)
        engine.telemetry_started = telemetry_started
        engine.observer = observer
        module = ('solvers.deco.recovery' if config.method=='deco' else
                  'solvers.hu.solver' if config.method=='hu_scenario' else
                  'solvers.'+config.method)
        adapter = import_module(module).solve
    engine.timing_hook = timing_hook
    initialization_before, initialization_cpu_before = engine.initialization_s, engine.initialization_cpu_s
    setup_finished, setup_cpu_finished = perf_counter(), cpu_time()
    timing = {'routing_setup': routing_setup,
              'setup': setup_finished - telemetry_started - routing_setup,
              'initialization': 0.0, 'search': 0.0, 'final_validation': 0.0}
    timing_cpu = {'routing_setup': routing_cpu,
                  'setup': setup_cpu_finished - cpu_started - routing_cpu,
                  'initialization': 0.0, 'search': 0.0, 'final_validation': 0.0}
    termination = 'completed'
    if seed_plan is not None:
        beginning = perf_counter()
        engine.phase = 'initialization'
        try:
            proposal = seed_plan.plan if isinstance(seed_plan, EvaluatedRecoveryPlan) else seed_plan
            engine.seed = validate_with_input_paths(instance, proposal, routing, input_path_facts)
            engine.require_transfer_policy(proposal)
            engine.remember(engine.seed, source='supplied')
            engine.stats['seed_source'] = 'supplied'
            engine.stats['seed_tardiness_s'] = engine.seed.total_tardiness_s
        except BudgetExhausted as error:


            termination = str(error)
            engine.stats['termination_phase'] = 'seed_validation'
            engine.seed = None
        finally:
            engine.initialization_s += perf_counter() - beginning
            engine.stats['initialization_runtime_s'] = engine.initialization_s
            engine.phase = 'search'
    adapter_returned = False
    try:
        if termination == 'completed':
            engine.check_objective_bound()
            candidate = adapter(engine)
            adapter_returned = True
            if candidate is not None:
                engine.remember(candidate)
    except ObjectiveLowerBoundReached:
        termination = 'objective_lower_bound'
    except BudgetExhausted as error:
        termination = str(error)
        engine.stats.setdefault('termination_phase', engine.phase)
    if engine._objective_zero_witness is not None:
        termination = 'objective_lower_bound'
        engine.stats['termination_phase'] = engine._objective_zero_witness['phase']
    search_finished, search_cpu_finished = perf_counter(), cpu_time()
    timing['initialization'] = engine.initialization_s - initialization_before
    timing['search'] = search_finished - setup_finished - timing['initialization']
    timing_cpu['initialization'] = engine.initialization_cpu_s - initialization_cpu_before
    timing_cpu['search'] = search_cpu_finished - setup_cpu_finished - timing_cpu['initialization']
    verified = None
    policy_output = engine.stats.get('output_semantics') == 'final_policy_state'
    selected = engine.policy_result if policy_output else engine.best
    if selected is not None:
        if any(v.operation not in ('pickup','delivery','transfer') for vs in selected.plan.sequences.values() for v in vs):
            raise ValueError('solver outputs must be canonical service/transfer plans')

        with timing_hook('validation') if timing_hook else nullcontext():
            engine.require_transfer_policy(selected.plan)
            verified = validate_with_input_paths(instance, selected.plan, routing, input_path_facts)
    elif termination == 'completed':
        termination = 'construction_failed'
    if termination == 'objective_lower_bound':
        if (verified is None or verified.total_tardiness_s != 0
                or selected.plan != engine._objective_zero_plan):
            raise ValueError('zero-bound stopping requires its fully verified complete witness')
        engine.stats['objective_lower_bound'] = {
            **engine._objective_zero_witness,
            'final_validation_passed': True,
            'native_adapter_returned': adapter_returned,
            'native_iterations_started': engine.stats['iterations_started'],
            'native_iterations_completed': engine.stats['iterations'],
        }
    timing['final_validation'] = perf_counter() - search_finished
    timing_cpu['final_validation'] = cpu_time() - search_cpu_finished
    engine.stats['evaluation_cache'] = dict(engine.evaluation_cache.stats)
    if policy_output:
        engine.stats['best_seen_tardiness_s'] = None if engine.best is None else engine.best.total_tardiness_s
    meeting_domain = ('current colocated workers; both stay until handover'
                      if engine.transfer_policy == 'encounter' else
                      'initial holder position; sender stays until handover'
                      if engine.transfer_policy == 'holder_origin' else
                      'legacy generated positions plus all permitted road nodes within sender range'
                      if config.meeting_pool == 'road_nodes' else
                      'finite generated road positions, including segment interiors')
    if config.method == 'deco':
        meeting_domain = 'predicted encounter projected onto the road network'
    role = 'offline_scenario_online_recovery' if config.method == 'hu_scenario' else 'recovery_adapter'
    engine.stats['search_scope'] = {
        'role': role,
        'transfer_policy': engine.transfer_policy,
        'meeting_domain': meeting_domain,
        'worker_candidates': config.worker_candidates, 'meeting_candidates': config.meeting_candidates,
        'candidate_cap_usage': 'spatial worker rings;8 joint evaluations and64 generated branches per task/worker/round',
        'new_transfers_per_inserted_task': 1,
        'global_transfer_event_limit': None,
        'objective': 'total_tardiness_s',
        'candidate_tie_breakers': None if policy_output else ['total_travel_time_s', 'transfer_count'],
        'incumbent_update': ('actual selected policy states; best seen kept separately' if policy_output else
                             'strict total tardiness improvement after adapter selection'),
    }
    if config.objective_zero_stopping:
        engine.stats['search_scope']['objective_zero_stopping'] = (
            'stop after a selected complete plan attains the validated public zero objective bound; '
            'native completed iteration counters are not filled to their declared limits')
    runtime, runtime_cpu = perf_counter() - telemetry_started, cpu_time() - cpu_started

    timing['setup'] += runtime - sum(timing.values())
    timing_cpu['setup'] += runtime_cpu - sum(timing_cpu.values())
    engine.stats['cpu_time_s'] = runtime_cpu
    engine.stats['cpu_scope'] = 'current process including threads and waited-for child processes on POSIX'
    return SolveResult(input_id=instance.instance_id, network_id=instance.network_id,
                       snapshot_id=instance.snapshot_id, config=config,
                       status='feasible' if verified is not None else 'no_plan_found', plan=verified,
                       runtime_s=runtime, termination_reason=termination,
                       statistics=_freeze(engine.stats), timing_s=timing, timing_cpu_s=timing_cpu,
                       incumbent_trace=tuple(engine.incumbent_trace))
