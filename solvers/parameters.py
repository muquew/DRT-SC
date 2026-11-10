import json
from pathlib import Path

from common.recovery.solver import SolverConfig
from common.config import BASELINE_DEFAULTS
from copy import deepcopy

FIELDS = {'iterations', 'max_evaluations', 'worker_candidates', 'meeting_candidates',
          'transfer_policy', 'meeting_pool', 'objective_zero_stopping', 'parameters'}
METHODS = ('greedy', 'alns', 'grasp', 'rlns', 'hu', 'deco')


def validate_overrides(values):
    if not isinstance(values, dict) or set(values) - FIELDS:
        raise ValueError('algorithm config accepts only: ' + ', '.join(sorted(FIELDS)))
    if 'parameters' in values and not isinstance(values['parameters'], dict):
        raise ValueError('parameters must be a JSON object')
    return values


def _build_solver_config(method, *, seed=0, budget_s=None, overrides=None):
    values = dict(validate_overrides({} if overrides is None else overrides))
    parameters = dict(values.pop('parameters', {}))
    config = SolverConfig(method='hu_scenario' if method == 'hu' else method,
        seed=seed, time_limit_s=budget_s, objective_mode='total_tardiness',
        parameters=parameters, **values)
    if method == 'greedy' and config.iterations != 1:
        raise ValueError('Greedy performs one construction; iterations must be 1')
    if method == 'deco' and config.iterations not in (0, 1):
        raise ValueError('DECO iterations must be 1 for one policy pass or 0 to skip policy execution')
    if method == 'hu' and config.iterations is not None:
        raise ValueError('Hu does not use iterations; keep None and use its scenario/local-search protocol')
    if (method == 'rlns' and parameters.get('protocol', 'paper') == 'paper'
            and config.iterations is not None):
        raise ValueError("RLNS paper protocol uses restart/round/patience parameters; iterations must be None, "
                         "or select protocol='iterations' for an explicit total round budget")

    if method == 'greedy' and parameters:
        raise ValueError('greedy has no method-specific parameters')
    if method == 'alns':
        from solvers.alns import validate_parameters
        validate_parameters(config.parameters)
    elif method in ('grasp', 'rlns'):
        from importlib import import_module
        import_module('solvers.' + method).validate_parameters(config.parameters)
    elif method in ('hu', 'deco'):
        allowed = ({'library_path','online_local_passes','online_local_actions'} if method == 'hu' else
            {'history_path', 'checkpoint_path', 'python_executable', 'device', 'reward_mode',
             'encounter_radius_m', 'min_remaining_time_s', 'max_assigned_tasks'})
        if set(parameters) - allowed:
            raise ValueError('unknown ' + method + ' parameters')
        for key in set(parameters) & {'library_path', 'history_path', 'checkpoint_path', 'python_executable'}:
            if not isinstance(parameters[key], str) or not parameters[key]:
                raise ValueError(key + ' must be a nonempty string')
        if 'device' in parameters and parameters['device'] not in ('cpu', 'auto', 'gpu'):
            raise ValueError('DECO device must be cpu, auto or gpu')
        import math
        for key in set(parameters) & {'encounter_radius_m', 'min_remaining_time_s'}:
            if type(parameters[key]) not in (int, float) or not math.isfinite(parameters[key]) or parameters[key] < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        if 'max_assigned_tasks' in parameters and (type(parameters['max_assigned_tasks']) is not int or parameters['max_assigned_tasks'] < 1):
            raise ValueError('max_assigned_tasks must be a positive integer')
        for key in set(parameters) & {'online_local_passes','online_local_actions'}:
            if type(parameters[key]) is not int or parameters[key] < 1:
                raise ValueError(key + ' must be a positive integer')
    return config


def solver_config(method, *, seed=0, budget_s=None, overrides=None, parameters=None):
    values = load_algorithm_config(None, method)
    supplied = validate_overrides({} if overrides is None else overrides)
    values = {**values, **supplied, 'parameters': {
        **values.get('parameters', {}), **supplied.get('parameters', {}), **dict(parameters or {})}}
    return _build_solver_config(method, seed=seed, budget_s=budget_s, overrides=values)


def load_algorithm_config(path, method):
    if method not in METHODS:
        raise ValueError('--algorithm-config is for the six baselines only')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    document = deepcopy(BASELINE_DEFAULTS) if path is None else json.loads(Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))
    if not isinstance(document, dict) or set(document) != {'defaults', 'methods'}:
        raise ValueError('algorithm config requires defaults and methods')
    defaults = validate_overrides(document['defaults'])
    methods = document['methods']
    if not isinstance(methods, dict) or set(methods) - set(METHODS):
        raise ValueError('unknown algorithm config method')
    merged = {}
    for name in set(methods) | {method}:
        specific = validate_overrides(methods.get(name, {}))
        merged[name] = {**defaults, **specific, 'parameters': {
            **defaults.get('parameters', {}), **specific.get('parameters', {})}}
        _build_solver_config(name, overrides=merged[name])
    return merged[method]


def config_record(config):
    return {**config.to_dict(), 'meeting_pool': config.meeting_pool,
            'objective_zero_stopping': config.objective_zero_stopping}
