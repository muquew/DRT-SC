from dataclasses import dataclass, field, replace
from functools import cached_property
import math

from common.initial_sequences.inputs import _exceeds_capacity
from common.recovery.plan import RecoveryPlan, RecoveryVisit, TransferEvent


@dataclass(frozen=True)
class InsertionDelta:
    base_plan: RecoveryPlan = field(repr=False, compare=False)
    serial: int
    kind: str
    task_id: str
    receiver: str
    sender: str | None = None
    position: object = None
    pickup_index: int | None = None
    sender_transfer_index: int | None = None
    receiver_transfer_index: int | None = None
    delivery_index: int = 0
    reuse_event_id: str | None = None
    transfer_id: str | None = None

    @property
    def worker_id(self):
        return self.receiver

    def changed_sequences(self):

        pickup = RecoveryVisit(operation='pickup', task_id=self.task_id)
        delivery = RecoveryVisit(operation='delivery', task_id=self.task_id)
        if self.kind == 'direct':
            route = self.base_plan.sequences[self.receiver]
            if self.pickup_index is not None:
                i = self.pickup_index
                route = route[:i] + (pickup,) + route[i:]
            i = self.delivery_index
            return {self.receiver: route[:i] + (delivery,) + route[i:]}
        sending = self.base_plan.sequences[self.sender]
        receiving = self.base_plan.sequences[self.receiver]
        if self.pickup_index is not None:
            i = self.pickup_index
            sending = sending[:i] + (pickup,) + sending[i:]
        if self.kind == 'transfer':
            visit = RecoveryVisit(operation='transfer', transfer_id=self.transfer_id)
            i, j = self.sender_transfer_index, self.receiver_transfer_index
            sending = sending[:i] + (visit,) + sending[i:]
            receiving = receiving[:j] + (visit,) + receiving[j:]
        elif self.kind != 'reuse':
            raise ValueError('unknown insertion delta kind')
        i = self.delivery_index
        receiving = receiving[:i] + (delivery,) + receiving[i:]
        return {self.sender: sending, self.receiver: receiving}

    @cached_property
    def _events(self):
        if self.kind == 'direct':
            return self.base_plan.transfers
        if self.kind == 'reuse':
            event = self.base_plan.transfer_map[self.reuse_event_id]
            direction = 'tasks_i_to_j' if self.sender == event.worker_i else 'tasks_j_to_i'
            changed = replace(event, **{direction: (*getattr(event, direction), self.task_id)})
            return tuple(changed if e.id == event.id else e for e in self.base_plan.transfers)
        if self.kind == 'transfer':
            event = TransferEvent(id=self.transfer_id, worker_i=self.sender, worker_j=self.receiver,
                                  location=self.position, tasks_i_to_j=(self.task_id,))
            return (*self.base_plan.transfers, event)
        raise ValueError('unknown insertion delta kind')

    def transfer_events(self):
        return self._events

    events = transfer_events

    def materialize(self):
        return replace(self.base_plan,
                       sequences={**self.base_plan.sequences, **self.changed_sequences()},
                       transfers=self.transfer_events())


def iter_deltas(engine, plan, task_id, senders, receivers, *, mode='all', locations=None):


    if mode not in ('all', 'direct', 'reuse', 'transfer'):
        raise ValueError('unknown insertion mode')
    owner = engine.holders.get(task_id)
    serial = 0
    if mode in ('all', 'direct'):
        for worker_id in receivers:
            if owner is not None and worker_id != owner:
                continue
            n = len(plan.sequences[worker_id])
            for pi in ([None] if owner is not None else range(n + 1)):
                for di in range(0 if pi is None else pi + 1, n + (pi is not None) + 1):
                    engine.check_budget()
                    yield InsertionDelta(plan, serial, 'direct', task_id, worker_id,
                                         pickup_index=pi, delivery_index=di)
                    serial += 1
    if engine.transfer_policy in ('holder_origin', 'encounter') and owner is None:
        return
    if mode in ('all', 'transfer', 'reuse'):
        for event in plan.transfers:
            if locations is not None and engine.item.roads.normalize_position(event.location) not in locations:
                continue
            for sender, receiver in ((event.worker_i, event.worker_j), (event.worker_j, event.worker_i)):
                if sender not in senders or receiver not in receivers:
                    continue
                if engine.transfer_policy in ('holder_origin', 'encounter'):
                    origin = engine.item.roads.normalize_position(engine.item.worker_map[sender].location)
                    if engine.item.roads.normalize_position(event.location) != origin:
                        continue
                sr, rr = plan.sequences[sender], plan.sequences[receiver]
                si = next(i for i, visit in enumerate(sr) if visit.transfer_id == event.id)
                ri = next(i for i, visit in enumerate(rr) if visit.transfer_id == event.id)
                for pi in ([None] if owner is not None else range(si + 1)):
                    for di in range(ri + 1, len(rr) + 1):
                        engine.check_budget()
                        yield InsertionDelta(plan, serial, 'reuse', task_id, receiver, sender,
                                             event.location, pi, delivery_index=di, reuse_event_id=event.id)
                        serial += 1
    if mode not in ('all', 'transfer'):
        return
    number = 1
    while f'transfer-{number}' in plan.transfer_map:
        number += 1
    event_id = f'transfer-{number}'
    for sender in senders:
        sr = plan.sequences[sender]
        stationary_end = None
        for receiver in receivers:
            if receiver == sender:
                continue
            rr = plan.sequences[receiver]
            positions = engine.meeting_positions(sender, receiver) if locations is None else locations
            if engine.transfer_policy in ('holder_origin', 'encounter'):
                origin = engine.item.roads.normalize_position(engine.item.worker_map[sender].location)
                positions = tuple(p for p in positions if engine.item.roads.normalize_position(p) == origin)
            for position in positions:
                for pi in ([None] if owner is not None else range(len(sr) + 1)):
                    if engine.transfer_policy in ('holder_origin', 'encounter'):
                        if stationary_end is None:
                            stationary_end = engine._stationary_prefix_length(plan, sender)
                        end = stationary_end
                    else:
                        end = len(sr) + (pi is not None)
                    for si in range(0 if pi is None else pi + 1, end + 1):
                        for ri in range(len(rr) + 1):
                            for di in range(ri + 1, len(rr) + 2):
                                engine.check_budget()
                                yield InsertionDelta(plan, serial, 'transfer', task_id, receiver, sender,
                                                     position, pi, si, ri, di, transfer_id=event_id)
                                serial += 1


class InsertionDeltaCursor:


    def __init__(self, engine, plan, task_id, senders, receivers, mode='all', locations=None, state=None):
        if mode not in ('all', 'direct', 'reuse', 'transfer'):
            raise ValueError('unknown insertion mode')
        self.engine, self.plan, self.task_id = engine, plan, task_id
        self.senders, self.receivers = tuple(senders), tuple(receivers)
        self.mode, self.locations = mode, None if locations is None else tuple(locations)
        self.owner = engine.holders.get(task_id)
        self.state = {} if state is None else state
        self._positions_pair, self._positions = None, ()
        if not self.state:
            self.state.update(serial=0,
                phase='direct' if mode in ('all', 'direct') else 'reuse',
                receiver=0, pickup=0, delivery=None, event=0, direction=0,
                reuse_sender_index=None, reuse_receiver_index=None,
                sender=0, position=0, sender_transfer=None, receiver_transfer=0,
                stationary_end=None, transfer_id=None)

    def __iter__(self):
        return self

    def _emit(self, kind, receiver, **values):
        self.engine.check_budget()
        delta = InsertionDelta(self.plan, self.state['serial'], kind, self.task_id, receiver, **values)
        self.state['serial'] += 1
        self.state['delivery'] += 1
        return delta

    def _direct(self):
        state = self.state
        while state['receiver'] < len(self.receivers):
            receiver = self.receivers[state['receiver']]
            n = len(self.plan.sequences[receiver])
            if ((self.owner is not None and receiver != self.owner)
                    or state['pickup'] >= (1 if self.owner is not None else n + 1)):
                state.update(receiver=state['receiver'] + 1, pickup=0, delivery=None)
                continue
            pi = None if self.owner is not None else state['pickup']
            if state['delivery'] is None:
                state['delivery'] = 0 if pi is None else pi + 1
            if state['delivery'] > n + (pi is not None):
                state.update(pickup=state['pickup'] + 1, delivery=None)
                continue
            return self._emit('direct', receiver, pickup_index=pi, delivery_index=state['delivery'])
        state.update(phase='reuse', receiver=0, pickup=0, delivery=None)
        return None

    def _next_reuse_direction(self):
        state = self.state
        direction = state['direction'] + 1
        state.update(direction=direction % 2, event=state['event'] + direction // 2,
                     pickup=0, delivery=None, reuse_sender_index=None, reuse_receiver_index=None)

    def _reuse(self):
        state, engine = self.state, self.engine
        if (self.mode not in ('all', 'transfer', 'reuse')
                or (engine.transfer_policy in ('holder_origin', 'encounter') and self.owner is None)):
            state['phase'] = 'done'
            return None
        while state['event'] < len(self.plan.transfers):
            event = self.plan.transfers[state['event']]
            sender, receiver = ((event.worker_i, event.worker_j) if state['direction'] == 0
                                else (event.worker_j, event.worker_i))
            if ((self.locations is not None and engine.item.roads.normalize_position(event.location) not in self.locations)
                    or sender not in self.senders or receiver not in self.receivers):
                self._next_reuse_direction()
                continue
            if engine.transfer_policy in ('holder_origin', 'encounter'):
                origin = engine.item.roads.normalize_position(engine.item.worker_map[sender].location)
                if engine.item.roads.normalize_position(event.location) != origin:
                    self._next_reuse_direction()
                    continue
            if state['reuse_sender_index'] is None:
                state['reuse_sender_index'] = next(i for i, visit in enumerate(self.plan.sequences[sender])
                                                   if visit.transfer_id == event.id)
                state['reuse_receiver_index'] = next(i for i, visit in enumerate(self.plan.sequences[receiver])
                                                     if visit.transfer_id == event.id)
            if state['pickup'] >= (1 if self.owner is not None else state['reuse_sender_index'] + 1):
                self._next_reuse_direction()
                continue
            if state['delivery'] is None:
                state['delivery'] = state['reuse_receiver_index'] + 1
            if state['delivery'] > len(self.plan.sequences[receiver]):
                state.update(pickup=state['pickup'] + 1, delivery=None)
                continue
            return self._emit('reuse', receiver, sender=sender, position=event.location,
                pickup_index=None if self.owner is not None else state['pickup'],
                delivery_index=state['delivery'], reuse_event_id=event.id)
        state.update(phase='transfer' if self.mode in ('all', 'transfer') else 'done',
                     receiver=0, pickup=0, delivery=None)
        return None

    def _next_receiver(self):
        state = self.state
        state.update(receiver=state['receiver'] + 1, position=0, pickup=0,
                     sender_transfer=None, receiver_transfer=0, delivery=None)

    def _pair_positions(self, sender, receiver):
        pair = sender, receiver
        if self._positions_pair != pair:
            positions = (self.engine.meeting_positions(sender, receiver)
                         if self.locations is None else self.locations)
            if self.engine.transfer_policy in ('holder_origin', 'encounter'):
                origin = self.engine.item.roads.normalize_position(self.engine.item.worker_map[sender].location)
                positions = tuple(p for p in positions
                                  if self.engine.item.roads.normalize_position(p) == origin)
            self._positions_pair, self._positions = pair, tuple(positions)
        return self._positions

    def _transfer(self):
        state, engine = self.state, self.engine
        if state['transfer_id'] is None:
            number = 1
            while f'transfer-{number}' in self.plan.transfer_map:
                number += 1
            state['transfer_id'] = f'transfer-{number}'
        while state['sender'] < len(self.senders):
            sender = self.senders[state['sender']]
            sr = self.plan.sequences[sender]
            if state['receiver'] >= len(self.receivers):
                state.update(sender=state['sender'] + 1, receiver=0, position=0, pickup=0,
                             sender_transfer=None, receiver_transfer=0, delivery=None, stationary_end=None)
                continue
            receiver = self.receivers[state['receiver']]
            if receiver == sender:
                self._next_receiver()
                continue
            rr = self.plan.sequences[receiver]
            positions = self._pair_positions(sender, receiver)
            if state['position'] >= len(positions):
                self._next_receiver()
                continue
            if state['pickup'] >= (1 if self.owner is not None else len(sr) + 1):
                state.update(position=state['position'] + 1, pickup=0,
                             sender_transfer=None, receiver_transfer=0, delivery=None)
                continue
            pi = None if self.owner is not None else state['pickup']
            if engine.transfer_policy in ('holder_origin', 'encounter'):
                if state['stationary_end'] is None:
                    state['stationary_end'] = engine._stationary_prefix_length(self.plan, sender)
                end = state['stationary_end']
            else:
                end = len(sr) + (pi is not None)
            if state['sender_transfer'] is None:
                state['sender_transfer'] = 0 if pi is None else pi + 1
            if state['sender_transfer'] > end:
                state.update(pickup=state['pickup'] + 1, sender_transfer=None,
                             receiver_transfer=0, delivery=None)
                continue
            if state['receiver_transfer'] > len(rr):
                state.update(sender_transfer=state['sender_transfer'] + 1,
                             receiver_transfer=0, delivery=None)
                continue
            if state['delivery'] is None:
                state['delivery'] = state['receiver_transfer'] + 1
            if state['delivery'] > len(rr) + 1:
                state.update(receiver_transfer=state['receiver_transfer'] + 1, delivery=None)
                continue
            return self._emit('transfer', receiver, sender=sender, position=positions[state['position']],
                pickup_index=pi, sender_transfer_index=state['sender_transfer'],
                receiver_transfer_index=state['receiver_transfer'], delivery_index=state['delivery'],
                transfer_id=state['transfer_id'])
        state['phase'] = 'done'
        return None

    def __next__(self):
        while self.state['phase'] != 'done':
            delta = {'direct': self._direct, 'reuse': self._reuse, 'transfer': self._transfer}[self.state['phase']]()
            if delta is not None:
                return delta
        raise StopIteration


class InventoryPrefilter:


    def __init__(self, item, base_plan, task_id):
        self.base_plan, self.task_id = base_plan, task_id
        self.supported, self.fallback_reason = False, None
        self._bad, self._event_indices = {}, {}
        if type(base_plan) is not RecoveryPlan or task_id not in item.task_map:
            self.fallback_reason = 'unsupported_plan_or_task'
            return
        holders = [w.id for w in item.workers if task_id in w.carried_task_ids]
        if len(holders) > 1 or set(base_plan.sequences) != set(item.worker_map):
            self.fallback_reason = 'unsupported_base_inventory'
            return
        self.owner = holders[0] if holders else None
        for event in base_plan.transfers:
            if type(event) is not TransferEvent or task_id in (*event.tasks_i_to_j, *event.tasks_j_to_i):
                self.fallback_reason = 'task_already_in_base_transfer_or_custom_event'
                return
        try:
            for worker_id, route in base_plan.sequences.items():
                worker = item.worker_map[worker_id]
                held = set(worker.carried_task_ids) - {task_id}
                prefix = [0]

                def append_mask():


                    load = math.fsum(item.task_map[t].load for t in sorted(held | {task_id}))
                    if not math.isfinite(load):
                        raise ValueError('nonfinite inventory')
                    prefix.append(prefix[-1] + _exceeds_capacity(load, worker.capacity))

                append_mask()
                for index, visit in enumerate(route):
                    if type(visit) is not RecoveryVisit or visit.task_id == task_id:
                        raise ValueError('private visit or inserted task already in base')
                    if visit.operation == 'transfer':
                        event = base_plan.transfer_map[visit.transfer_id]
                        if worker_id not in (event.worker_i, event.worker_j):
                            raise ValueError('invalid transfer membership')
                        outgoing, incoming = ((event.tasks_i_to_j, event.tasks_j_to_i)
                                              if worker_id == event.worker_i else
                                              (event.tasks_j_to_i, event.tasks_i_to_j))
                        outgoing, incoming = set(outgoing), set(incoming)
                        if (outgoing - held) or (incoming & held):
                            raise ValueError('base custody')
                        held = (held - outgoing) | incoming
                        key = event.id, worker_id
                        if key in self._event_indices:
                            raise ValueError('duplicate transfer visit')
                        self._event_indices[key] = index
                    elif visit.operation == 'pickup':
                        if visit.task_id in held or visit.task_id not in item.task_map:
                            raise ValueError('base custody')
                        held.add(visit.task_id)
                    elif visit.operation == 'delivery':
                        if visit.task_id not in held:
                            raise ValueError('base custody')
                        held.remove(visit.task_id)
                    else:
                        raise ValueError('unsupported visit')
                    append_mask()
                if held:
                    raise ValueError('base remaining custody')
                self._bad[worker_id] = tuple(prefix)
            if any((e.id, w) not in self._event_indices for e in base_plan.transfers
                   for w in (e.worker_i, e.worker_j)):
                raise ValueError('unpaired base transfer')
        except (KeyError, ValueError, OverflowError):
            self._bad.clear()
            self._event_indices.clear()
            self.fallback_reason = 'unsupported_or_invalid_base_inventory'
            return
        self.supported = True

    def rejection(self, delta):

        if not self.supported or delta.base_plan is not self.base_plan or delta.task_id != self.task_id:
            return None
        if self.owner is not None and (delta.sender or delta.receiver) != self.owner:
            return 'custody'
        if (delta.pickup_index is None) != (self.owner is not None):
            return None
        pi, di = delta.pickup_index, delta.delivery_index
        if delta.kind == 'direct':
            ranges = ((delta.receiver, 1 if pi is None else pi, di + (pi is None)),)
        elif delta.kind == 'reuse':
            try:
                si = self._event_indices[delta.reuse_event_id, delta.sender]
                ri = self._event_indices[delta.reuse_event_id, delta.receiver]
            except KeyError:
                return None
            ranges = ((delta.sender, 1 if pi is None else pi, si + 1),
                      (delta.receiver, ri + 1, di + 1))
        elif delta.kind == 'transfer':
            ranges = ((delta.sender, 1 if pi is None else pi,
                       delta.sender_transfer_index + (pi is None)),
                      (delta.receiver, delta.receiver_transfer_index, di))
        else:
            return None
        for worker_id, lo, hi in ranges:
            prefix = self._bad.get(worker_id)
            if prefix is None or not 0 <= lo <= hi < len(prefix):
                return None
        return 'capacity' if any(self._bad[w][hi] != self._bad[w][lo] for w, lo, hi in ranges) else None
