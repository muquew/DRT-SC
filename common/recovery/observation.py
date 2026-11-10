from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter, process_time

from common.initial_sequences.inputs import _plain
from common.recovery.inputs import RecoveryInput
from common.recovery.plan import EvaluatedRecoveryPlan


def children_cpu_time():

    try:
        import resource
    except ImportError:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def cpu_time():
    return process_time() + children_cpu_time()


class ObserverError(RuntimeError):


    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, kw_only=True)
class IncumbentUpdate:


    sequence: int
    plan: EvaluatedRecoveryPlan
    source: str
    phase: str
    solver_elapsed_s: float
    available_elapsed_s: float
    state_time_s: float
    available_cpu_s: float | None = None
    validation_completed_perf_s: float | None = None

    def to_dict(self):
        return dict(sequence=self.sequence, source=self.source, phase=self.phase,
                    solver_elapsed_s=self.solver_elapsed_s,
                    available_elapsed_s=self.available_elapsed_s,
                    available_cpu_s=self.available_cpu_s,
                    state_time_s=self.state_time_s, plan=self.plan.to_dict())


class RecoveryObserver:


    def __init__(self, on_incumbent=None, *, origin='solver_call', input_path_facts=None):
        if on_incumbent is not None and not callable(on_incumbent):
            raise TypeError('on_incumbent must be callable or None')
        if origin not in ('solver_call', 'event_notification'):
            raise ValueError('unknown observation origin')
        self._started, self._cpu_started = perf_counter(), cpu_time()
        self._callback, self.origin = on_incumbent, origin
        self._instance = None
        self._finished = None
        self._active_stages = 0
        self._stages, self._updates = {}, []
        self._input_path_facts = input_path_facts


    @property
    def wall_origin_s(self):

        return self._started

    def _active(self):
        if self._finished is not None:
            raise RuntimeError('observation is already finished')


    def begin_solver(self, instance):
        self._active()
        if self._instance is not None:
            raise RuntimeError('an observation belongs to one solver invocation')
        if not isinstance(instance, RecoveryInput):
            raise TypeError('expected RecoveryInput')
        self._instance = instance

    @contextmanager
    def measure(self, name, *, parent=None):

        self._active()
        if not isinstance(name, str) or not name:
            raise ValueError('a phase needs a name')
        entry = self._stages.setdefault(name, dict(calls=0, wall_s=0.0,
                                                   cpu_s=0.0, failures=0, parent=parent))
        if entry['parent'] != parent:
            raise ValueError('a phase must retain its parent')
        entry['calls'] += 1
        self._active_stages += 1
        wall, cpu = perf_counter(), cpu_time()
        try:
            yield
        except BaseException:
            entry['failures'] += 1
            raise
        finally:
            entry['wall_s'] += perf_counter()-wall
            entry['cpu_s'] += cpu_time()-cpu
            self._active_stages -= 1

    def publish(self, instance, routing, result, *, source, phase, solver_elapsed_s):
        self._active()
        if instance is not self._instance:
            raise RuntimeError('publication does not belong to this solver invocation')
        row = dict(sequence=len(self._updates)+1, source=source, phase=phase,
                   state_time_s=instance.time_s, solver_elapsed_s=solver_elapsed_s,
                   internal_elapsed_s=perf_counter()-self._started,
                   internal_cpu_s=cpu_time()-self._cpu_started,
                   available_elapsed_s=None, available_cpu_s=None,
                   callback_completed_elapsed_s=None, callback_completed_cpu_s=None,
                   status='pending', metrics=None, error=None)
        self._updates.append(row)
        stage = 'incumbent_validation'
        try:
            with self.measure(stage, parent='solve'):
                from common.evaluation.path_facts import validate_with_input_paths

                verified = validate_with_input_paths(instance, result.plan, routing, self._input_path_facts)
                if verified.to_dict() != result.to_dict():
                    raise ValueError('native candidate differs from independent full acceptance')
                validation_completed = perf_counter()
                metrics = {name: getattr(verified, name) for name in
                           ('total_tardiness_s', 'late_task_count', 'transfer_count',
                            'total_travel_time_s', 'total_wait_time_s', 'total_distance_m')}
                update = IncumbentUpdate(sequence=row['sequence'], plan=verified,
                    source=source, phase=phase, solver_elapsed_s=solver_elapsed_s,
                    available_elapsed_s=perf_counter()-self._started,
                    available_cpu_s=cpu_time()-self._cpu_started,
                    state_time_s=instance.time_s,
                    validation_completed_perf_s=validation_completed)
            row.update(status='validated', metrics=metrics,
                       available_elapsed_s=update.available_elapsed_s,
                       available_cpu_s=update.available_cpu_s)
            if self._callback is not None:
                stage = 'callback'
                with self.measure(stage, parent='solve'):
                    self._callback(update)
                row.update(status='delivered', callback_completed_elapsed_s=perf_counter()-self._started,
                           callback_completed_cpu_s=cpu_time()-self._cpu_started)
        except Exception as error:
            row.update(status='validation_failed' if stage == 'incumbent_validation' else 'callback_failed',
                       error=dict(type=type(error).__name__, message=str(error)))
            raise ObserverError(stage, f'{stage} failed for incumbent {row["sequence"]}') from error
        except BaseException as error:
            row.update(status='interrupted', error=dict(type=type(error).__name__, message=str(error)))
            raise

    def to_dict(self):
        wall, cpu = self._finished or (perf_counter(), cpu_time())
        available = [r for r in self._updates if r['available_elapsed_s'] is not None]
        delivered = [r for r in self._updates if r['callback_completed_elapsed_s'] is not None]
        item = self._instance
        return _plain(dict(
            scope='frozen_snapshot', origin=self.origin,
            status=('running' if self._finished is None else
                    'failed' if any(s['failures'] for s in self._stages.values()) else 'complete'),
            input_id=None if item is None else item.instance_id,
            network_id=None if item is None else item.network_id,
            snapshot_id=None if item is None else item.snapshot_id,
            state_time_s=None if item is None else item.time_s,
            elapsed_s=wall-self._started, cpu_time_s=cpu-self._cpu_started,
            cpu_scope='current process including threads and waited-for child processes on POSIX',
            callback_enabled=self._callback is not None,
            first_internal_elapsed_s=None if not self._updates else self._updates[0]['internal_elapsed_s'],
            first_available_elapsed_s=None if not available else available[0]['available_elapsed_s'],
            first_available_cpu_s=None if not available else available[0]['available_cpu_s'],
            first_callback_completed_elapsed_s=None if not delivered else delivered[0]['callback_completed_elapsed_s'],
            first_callback_completed_cpu_s=None if not delivered else delivered[0]['callback_completed_cpu_s'],
            last_callback_completed_elapsed_s=None if not delivered else delivered[-1]['callback_completed_elapsed_s'],
            last_callback_completed_cpu_s=None if not delivered else delivered[-1]['callback_completed_cpu_s'],
            stages=self._stages, updates=self._updates))

    def finish(self):
        if self._active_stages:
            raise RuntimeError('cannot finish an active observation phase')
        if self._finished is None:
            self._finished = perf_counter(), cpu_time()
        return self.to_dict()
