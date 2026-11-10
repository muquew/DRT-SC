from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property

from common.config import cache_limits
from common.evaluation.recovery_plan import _context, _directions, _load
from common.initial_sequences.inputs import _exceeds_capacity
from common.recovery.plan import RecoveryVisit, TransferEvent


@dataclass(frozen=True)
class ContinuationDelta:
    base_plan: object
    sender: str
    receiver: str
    position: object
    gap: int
    block: tuple
    carried_task_ids: tuple
    transfer_id: str | None = None
    serial: int = 0
    kind = 'continuation'

    @property
    def task_ids(self):
        return frozenset(v.task_id for v in self.block)

    @cached_property
    def _transfer(self):
        if self.position is None:
            return None, ()
        event = TransferEvent(id=self.transfer_id, worker_i=self.sender, worker_j=self.receiver,
            location=self.position, tasks_i_to_j=self.carried_task_ids)
        return event, (RecoveryVisit(operation='transfer', transfer_id=event.id),)

    def transfer_events(self):
        event, _ = self._transfer
        return self.base_plan.transfers if event is None else (*self.base_plan.transfers, event)

    def changed_sequences(self):
        _, visit = self._transfer
        route = self.base_plan.sequences[self.receiver]
        return {self.sender: visit,
                self.receiver: route[:self.gap] + visit + self.block + route[self.gap:]}

    def materialize(self):
        return replace(self.base_plan, sequences={**self.base_plan.sequences, **self.changed_sequences()},
                       transfers=self.transfer_events())


def iter_continuations(engine, plan, sender, block):

    serial = 0
    proxy = min(v.task_id for v in block)
    for receiver in engine._workers(plan, proxy):
        engine.check_budget()
        route = plan.sequences[receiver]
        locations = engine.meeting_positions(sender.id, receiver) if sender.carried_task_ids else (None,)
        for location in locations:
            transfer_id = None
            if location is not None:
                number = 1
                while f'transfer-{number}' in plan.transfer_map:
                    number += 1
                transfer_id = f'transfer-{number}'
            for gap in range(len(route) + 1):
                yield ContinuationDelta(plan, sender.id, receiver, location, gap,
                    block, sender.carried_task_ids, transfer_id, serial)
                serial += 1


class BulkInventoryPrefilter:


    def __init__(self, item, base_plan):
        self.item, self.base_plan = item, base_plan
        self.limit = cache_limits('memo')['meeting_entries']
        self._results = OrderedDict()

    def rejection(self, delta):
        if type(delta) is not ContinuationDelta or delta.base_plan is not self.base_plan:
            return None
        key = delta.sender, delta.receiver, delta.gap, delta.block, delta.carried_task_ids
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        result = self._check(delta)
        if self.limit:
            self._results[key] = result
            if len(self._results) > self.limit:
                self._results.popitem(last=False)
        return result

    def _check(self, delta):
        events = {e.id: e for e in delta.transfer_events()}
        remaining = False
        for worker_id, visits in delta.changed_sequences().items():
            worker = self.item.worker_map[worker_id]
            held = set(worker.carried_task_ids)
            for index, visit in enumerate(visits):
                context = _context(worker_id, index, visit)
                if visit.operation == 'transfer':
                    outgoing, incoming = map(set, _directions(events[visit.transfer_id], worker_id))
                    if (outgoing - held) or (incoming & held):
                        return 'custody'
                    held = (held - outgoing) | incoming
                elif visit.operation == 'pickup':
                    if visit.task_id in held:
                        return 'custody'
                    held.add(visit.task_id)
                elif visit.operation == 'delivery':
                    if visit.task_id not in held:
                        return 'custody'
                    held.remove(visit.task_id)
                else:
                    return None
                if _exceeds_capacity(_load(self.item, held, **context), worker.capacity):
                    return 'capacity'
            remaining |= bool(held)
        return 'remaining_custody' if remaining else None
