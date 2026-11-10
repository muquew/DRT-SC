from pathlib import Path
import os
import sys
from solvers.parameters import solver_config
from common.recovery.solver import solve_recovery

METHODS = ('greedy', 'alns', 'grasp', 'rlns', 'hu', 'deco')


def solve(item, routing, *, method, seed=0, parameters=None, algorithm_config=None, observer=None):
    if method == 'hu_scenario':
        method = 'hu'
    if method not in METHODS:
        raise ValueError('unknown baseline: ' + method)
    parameters = dict(parameters or {})
    if method in ('hu', 'deco'):
        required = ('library_path',) if method == 'hu' else ('checkpoint_path', 'history_path')
        for name in required:
            if name not in parameters:
                raise ValueError(method + ' requires ' + name)
            path = Path(parameters[name]).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            parameters[name] = str(path)
        if method == 'deco':
            parameters.setdefault('python_executable', os.environ.get('DRT_SC_DECO_PYTHON', sys.executable))
    config = solver_config(method, seed=seed, overrides=algorithm_config, parameters=parameters)
    return solve_recovery(item, config, routing, observer=observer)
