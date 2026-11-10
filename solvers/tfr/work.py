from dataclasses import dataclass, field
import warnings


def single_thread_lp(solver, *args, **kwargs):

    from scipy.optimize import OptimizeWarning
    options = dict(kwargs.get('options') or {})
    if options.get('threads', 1) != 1:
        raise ValueError('the recovery protocol uses one LP computation thread')
    options['threads'] = 1
    kwargs['options'] = options
    with warnings.catch_warnings():

        warnings.filterwarnings('ignore', category=OptimizeWarning,
            message=r"Unrecognized options detected: \{'threads': 1\}\. These will be passed to HiGHS verbatim\.")
        return solver(*args, **kwargs)


class WorkLimitReached(RuntimeError):
    pass


@dataclass
class StructureWork:
    geometry_limit: int = 64
    lp_limit: int = 16
    lp_maxiter: int = 10_000
    used: dict = field(default_factory=lambda: dict(geometry_nodes=0, lp_calls=0, lp_iterations=0))
    exhausted: str | None = None

    def __post_init__(self):
        for value in (self.geometry_limit, self.lp_limit, self.lp_maxiter):
            if type(value) is not int or value < 1:
                raise ValueError('structure work limits must be positive integers')

    def charge(self, name):
        limit = self.geometry_limit if name == 'geometry_nodes' else self.lp_limit
        if name not in ('geometry_nodes', 'lp_calls'):
            raise ValueError('unknown work counter: ' + name)
        if self.used[name] >= limit:
            self.exhausted = name
            raise WorkLimitReached(name)
        self.used[name] += 1
