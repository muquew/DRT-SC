from dataclasses import asdict
import math
from time import perf_counter

from common.recovery.observation import cpu_time

METHODS = ('greedy', 'alns', 'grasp', 'rlns', 'hu', 'deco', 'h', 'trail')


def solve(item, method, routing=None, *, seed=0, config=None, parameters=None,
          origin=None, cpu_origin=None):
    if method not in METHODS:
        raise ValueError('unknown method: '+str(method))
    started = perf_counter() if origin is None else origin
    cpu_started = cpu_time() if cpu_origin is None else cpu_origin
    from common.recovery.preparation import prepare_recovery_input
    from common.evaluation.path_facts import capture_input_paths, validate_with_input_paths
    prepared = prepare_recovery_input(item, routing)
    suffix, routing, boundary = prepared.projected, prepared.routing, prepared.boundary
    paths = capture_input_paths(routing)
    settings = dict(config or {})
    parameters = dict(parameters or {})
    if method in ('h', 'trail'):
        from solvers.tfr.local_solver import LocalConfig, solve_local
        native_config = LocalConfig(**dict(settings, seed=seed))
        if method == 'h':
            native = solve_local(suffix, routing, native_config, origin=started)
        else:
            from solvers.trail.solver import solve as solve_trail
            native = solve_trail(suffix, routing, parameters['checkpoint_path'],
                config=native_config, seed=seed, origin=started)
        diagnostics = native.diagnostics
        candidate = None if native.plan is None else native.plan.plan
        first_suffix_T = diagnostics.get('initial_T')
        first_time = diagnostics.get('first_solution_s')
        status = native.status
        statistics = diagnostics
        effective_config = asdict(native_config)
    else:
        from solvers.interface import solve as solve_baseline
        native = solve_baseline(suffix, routing, method=method, seed=seed,
            parameters=parameters, algorithm_config=settings)
        candidate = None if native.plan is None else native.plan.plan
        first = next(iter(native.incumbent_trace), None)
        first_suffix_T = None if first is None else first['total_tardiness_s']
        first_time = None if first is None else first['available_perf_s']-started
        status = native.termination_reason
        statistics = dict(native.statistics)
        effective_config = native.config.to_dict()
    fixed_T = boundary.observed_tardiness_s+boundary.committed_tardiness_upper_s
    result = dict(method=method, seed=seed, input_id=item.instance_id,
        task_count=len(item.tasks), worker_count=len(item.workers), time_s=item.time_s,
        status=status, config=effective_config, has_complete_solution=candidate is not None,
        first_T_s=None if first_suffix_T is None else first_suffix_T+fixed_T,
        first_solution_s=first_time, T_s=None, plan=None, completion_times_s=None,
        transfer_count=None, late_task_count=None, statistics=statistics)
    if candidate is not None:
        evaluated = validate_with_input_paths(suffix, candidate, routing, paths)
        completed = {**boundary.delivered_observed, **boundary.delivered_committed,
                     **evaluated.completion_times_s}
        if set(completed) != set(item.task_map):
            raise ValueError('solution must complete every original task')
        result.update(T_s=math.fsum(max(0., end-item.task_map[t].promised_delivery_time_s)
                                  for t, end in completed.items()),
            plan=candidate.to_dict(), completion_times_s=completed,
            transfer_count=len(candidate.transfers),
            late_task_count=sum(end > item.task_map[t].promised_delivery_time_s
                                for t, end in completed.items()))
    result.update(wall_s=perf_counter()-started, cpu_s=cpu_time()-cpu_started)
    from common.initial_sequences.inputs import _plain
    return _plain(result)
