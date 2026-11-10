from common.config import HU_DEFAULTS


def resolve(total_iterations=None, candidate_count=None):
    defaults = HU_DEFAULTS
    values = dict(total_iterations_per_scenario=defaults['total_iterations_per_scenario'] if total_iterations is None else total_iterations,
                  candidate_count=defaults['candidate_count'] if candidate_count is None else candidate_count)
    for name, value in values.items():
        if type(value) is not int or value < (1 if name == 'candidate_count' else 0):
            raise ValueError('invalid Hu parameter: ' + name)
    return values


class IterationBudget:
    def __init__(self, total):
        self.total, self.used = total, 0
    @property
    def remaining(self):
        return self.total - self.used
    def consume(self):
        if self.remaining <= 0:
            raise RuntimeError('Hu iteration allowance exhausted')
        self.used += 1
    def record(self, reason):
        return dict(total=self.total, used=self.used, remaining=self.remaining, terminal=reason)
