from dataclasses import dataclass, replace
from collections import Counter
from common.cache_telemetry import observed_lru_cache as lru_cache, lru_snapshot, retention_counts, bypass
from common.config import cache_limits
from math import isfinite, fsum
import hashlib
import json
from types import MappingProxyType

from common.initial_sequences.inputs import _exceeds_capacity, _plain, _freeze
from common.recovery.plan import RecoveryVisit, RecoveryPlan, TransferEvent
from common.recovery.inputs import RecoveryInput, RecoveryWorker, checked_activity_domains


@lru_cache(maxsize=cache_limits("memo")["worker_entries"])
def _reconstructed_worker_values(values, location, cargo):


    return RecoveryWorker(id=values[0], location=location, capacity=values[1],
        profile=values[2], carried_task_ids=cargo, handover_range_m=values[3],
        service_mode=values[4])


_worker_bypass = retention_counts()
_worker_lru_clear = _reconstructed_worker_values.cache_clear
def _clear_worker_cache():
    _worker_lru_clear()
    _worker_bypass.update(retention_counts())
_reconstructed_worker_values.cache_clear = _clear_worker_cache


def worker_cache_snapshot():
    row=lru_snapshot(_reconstructed_worker_values,'process-wide worker reconstruction; native key includes all worker values')
    row['bypasses']+=_worker_bypass['bypasses']
    for reason,count in _worker_bypass['bypass_reasons'].items():
        row['bypass_reasons'][reason]=row['bypass_reasons'].get(reason,0)+count
    row['cargo_retention_items']=cache_limits('memo')['worker_cargo_items']
    row['counting_unit']='LRU lookup hits/misses; rejected eligibility is one bypass, not a lookup miss; all reset with cache_clear'
    return row


def _expected_worker(worker, *, location, cargo):

    from common.data.roads.schema import RoadPosition, MovementProfile
    if (type(worker) is not RecoveryWorker or type(location) is not RoadPosition
            or type(worker.profile) is not MovementProfile or type(cargo) is not tuple
            or len(cargo)>cache_limits("memo")["worker_cargo_items"] or any(type(t) is not str for t in cargo)
            or type(worker.id) is not str or type(worker.capacity) is not float
            or type(worker.handover_range_m) is not float or type(worker.service_mode) is not str
            or type(worker.profile.speed_factor) is not float
            or type(location.network_id) is not str
            or type(location.node_id) not in (str,type(None))
            or type(location.segment_id) not in (str,type(None))
            or type(location.offset_m) not in (float,type(None))):
        bypass(_worker_bypass,'oversized_cargo' if type(cargo) is tuple and len(cargo)>cache_limits('memo')['worker_cargo_items'] else 'unsupported_type')
        return replace(worker, location=location, carried_task_ids=cargo)

    values=(worker.id, worker.capacity, worker.profile, worker.handover_range_m,worker.service_mode)
    return _reconstructed_worker_values(values, location, cargo)


class BoundaryUncertified(ValueError):
    pass


def _finite(value, label, *, nonnegative=False):
    if not isfinite(value) or (nonnegative and value < 0):
        raise BoundaryUncertified(f'{label}: finite valid bound required')


def input_digest(item):


    if type(item) is RecoveryInput:
        return item.fingerprint
    return hashlib.sha256(json.dumps(item.to_dict(), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenSnapshotWindow:


    tau: float

    def __post_init__(self):
        _finite(self.tau, 'offline snapshot time')

    offline_frozen = True
    activation_s = property(lambda self: self.tau)
    confirmed_at_s = property(lambda self: self.tau)
    clock_margin_s = property(lambda self: 0.)
    response_budget_s = property(lambda self: None)
    tail_reserve_s = property(lambda self: None)
    response_deadline_s = property(lambda self: None)
    solve_deadline_s = property(lambda self: None)


@dataclass(frozen=True)
class ActionBlock:
    id: str
    kind: str
    participants: tuple
    origins: tuple
    destination: object
    duration_upper_s: float
    evidence: str
    task_id: str | None = None
    tasks_i_to_j: tuple = ()
    tasks_j_to_i: tuple = ()
    ready_at_s: float = 0.
    confirmed: bool = False
    safe_to_complete: bool = False
    irrevocable: bool = False
    started_at_s: float | None = None
    observed_start_s: float | None = None
    observed_end_s: float | None = None

    def __post_init__(self):
        if not self.id or self.kind not in ('travel', 'pickup', 'delivery', 'transfer', 'wait'):
            raise BoundaryUncertified('invalid action block')
        if len(self.participants) != len(set(self.participants)) or len(self.origins) != len(self.participants):
            raise BoundaryUncertified('invalid action participants/origins')
        if len(self.participants) != (2 if self.kind == 'transfer' else 1):
            raise BoundaryUncertified('action participant count')
        _finite(self.duration_upper_s, 'duration', nonnegative=True)
        _finite(self.ready_at_s, 'ready')
        if (self.observed_start_s is None) != (self.observed_end_s is None):
            raise BoundaryUncertified('observation must specify both actual clocks')
        for value in (self.started_at_s, self.observed_start_s, self.observed_end_s):
            if value is not None:
                _finite(value, 'action clock')
        if self.observed and self.observed_end_s < self.observed_start_s:
            raise BoundaryUncertified('observed completion precedes start')
        if self.kind in ('pickup', 'delivery') and self.task_id is None:
            raise BoundaryUncertified('service block requires task')
        if len(set(self.tasks_i_to_j)) != len(self.tasks_i_to_j) or len(set(self.tasks_j_to_i)) != len(self.tasks_j_to_i):
            raise BoundaryUncertified('duplicate physical item in transfer')
        if self.kind == 'transfer' and (not (self.tasks_i_to_j or self.tasks_j_to_i) or
                set(self.tasks_i_to_j) & set(self.tasks_j_to_i)):
            raise BoundaryUncertified('invalid atomic transfer inventory')

    @property
    def observed(self):
        return self.observed_end_s is not None


@dataclass(frozen=True)
class HoldingGuarantee:
    position: object
    until_s: float | None = None
    gate_duration_s: float | None = None
    evidence: str = ''

    def __post_init__(self):
        if self.until_s is not None:
            raise BoundaryUncertified('legacy finite holding expiry is unsupported; omit it or use null for continuing custody')
        if self.gate_duration_s is None:
            raise BoundaryUncertified('explicit gate duration required')
        _finite(self.gate_duration_s, 'gate duration', nonnegative=True)
        if not self.evidence:
            raise BoundaryUncertified('holding and boundary-gate evidence required')


@dataclass(frozen=True)
class PrefixExecution:
    block: ActionBlock
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ExecutionBoundary:
    original_digest: str
    original_task_ids: frozenset
    window: FrozenSnapshotWindow
    prefix: tuple
    locations: object
    release_times: object
    inventories: object
    permission_anchors: object
    delivered_observed: object
    delivered_committed: object
    remaining_task_ids: frozenset
    observed_tardiness_s: float
    committed_tardiness_upper_s: float
    holding_until: object
    original_input: object
    reference_plan: object
    activity_domains: object
    gate_duration_s: object
    completion_upper_s: object
    delivery_lower_s: object
    completion_lower_s: object
    execution_contract: object

    @property
    def content_key(self):
        return self.boundary_id

    @property
    def boundary_id(self):
        content = (self.original_digest, repr(self.window),
                   tuple(sorted((w,repr(p)) for w,p in self.permission_anchors.items())),
                   tuple(sorted((w,repr(d)) for w,d in self.activity_domains.items())),
                   tuple((x.block.id, x.start_s, x.end_s, repr(x.block)) for x in self.prefix),
                   tuple(sorted((w, repr(self.locations[w]), self.release_times[w], self.inventories[w],
                                 self.holding_until[w],self.gate_duration_s[w],self.completion_upper_s[w],self.completion_lower_s[w]) for w in self.locations)))
        return hashlib.sha256(repr(content).encode()).hexdigest()

    @property
    def H(self):
        return self.window.activation_s

    def validate_for(self, item):
        original=self.original_input
        if self.original_task_ids!=frozenset(original.task_map):
            raise BoundaryUncertified('original task set changed')
        if dict(self.permission_anchors)!={w.id:w.location for w in original.workers}:
            raise BoundaryUncertified('permission anchors must retain original anomaly locations')
        if dict(self.activity_domains)!=dict(getattr(original.source_settings,'activity_domains',{})):
            raise BoundaryUncertified('activity domains changed after prefix certification')
        for mapping in (self.locations,self.release_times,self.inventories,self.holding_until,
                        self.gate_duration_s,self.completion_upper_s,self.completion_lower_s):
            if set(mapping)!=set(original.worker_map):
                raise BoundaryUncertified('boundary maps must retain every original worker')
        for w in original.worker_map:
            release=self.release_times[w]
            _finite(self.gate_duration_s[w],'gate duration',nonnegative=True)
            _finite(self.completion_upper_s[w],'completion bound')
            _finite(self.completion_lower_s[w],'completion lower bound')
            if self.completion_lower_s[w]>self.completion_upper_s[w]:
                raise BoundaryUncertified('inconsistent prefix completion bounds')
            if self.holding_until[w] is not None:
                raise BoundaryUncertified('legacy finite holding expiry is unsupported')
            _finite(release,'release')
            if release!=max(self.H,self.completion_upper_s[w]+self.gate_duration_s[w]):
                raise BoundaryUncertified('release differs from certified completion/gate bound')
            if self.completion_upper_s[w]<self.window.confirmed_at_s:
                raise BoundaryUncertified('holding/gate boundary not realizable')
        if getattr(item, 'execution_boundary', None) is self:
            if input_digest(item.original_input) != self.original_digest:
                raise BoundaryUncertified('source input changed after prefix confirmation')
            if frozenset(item.task_map) != self.remaining_task_ids:
                raise BoundaryUncertified('suffix task partition changed')
            original=item.original_input
            if (item.roads != original.roads or item.source_settings != original.source_settings or
                    item.events != original.events or item.changes != original.changes):
                raise BoundaryUncertified('projection changes physical conditions')
            expected_reference={w:tuple((v.operation,v.task_id) for v in visits if v.operation in ('pickup','delivery'))
                                for w,visits in self.reference_plan.sequences.items()}
            actual_reference={w:tuple((v.operation,v.task_id) for v in visits) for w,visits in item.reference_sequences.items()}
            if actual_reference != expected_reference:
                raise BoundaryUncertified('projection changes the original suffix order')
            if item.source_snapshot is not item.original_input.source_snapshot:
                raise BoundaryUncertified('projection must preserve the original observed snapshot')
            if tuple(item.tasks) != tuple(t for t in item.original_input.tasks if t.id in self.remaining_task_ids):
                raise BoundaryUncertified('projection changes task attributes')
            if set(item.worker_map) != set(item.original_input.worker_map):
                raise BoundaryUncertified('projection drops a worker obligation')
            for w in item.workers:
                if w.location != self.locations[w.id] or tuple(w.carried_task_ids) != self.inventories[w.id]:
                    raise BoundaryUncertified('suffix departure inventory/location changed')
                expected = _expected_worker(item.original_input.worker_map[w.id], location=self.locations[w.id],
                                   cargo=self.inventories[w.id])
                if w != expected or self.release_times[w.id] < self.H:
                    raise BoundaryUncertified('projection changes qualification or release boundary')
        elif input_digest(item) != self.original_digest:
            raise BoundaryUncertified('boundary belongs to another input')
        if (set(self.delivered_observed) & set(self.delivered_committed) or
            (set(self.delivered_observed) | set(self.delivered_committed)) & set(self.remaining_task_ids)):
            raise BoundaryUncertified('task partition overlaps')
        if (set(self.delivered_observed) | set(self.delivered_committed) | set(self.remaining_task_ids)
                != set(self.original_task_ids)):
            raise BoundaryUncertified('original task obligations are not partitioned')
        return self

    def to_dict(self):
        result = dict(boundary_id=self.boundary_id, original_digest=self.original_digest,
            H=self.H, D_solve=self.window.solve_deadline_s, D_resp=self.window.response_deadline_s,
            t_F=self.window.confirmed_at_s, release_times=dict(self.release_times),
            locations=_plain(self.locations),permission_anchors=_plain(self.permission_anchors),
            activity_domains=_plain(self.activity_domains),holding_until=dict(self.holding_until),
            inventories={w:list(v) for w,v in self.inventories.items()},
            gate_duration_s=dict(self.gate_duration_s),completion_upper_s=dict(self.completion_upper_s),
            delivery_lower_s=dict(self.delivery_lower_s),completion_lower_s=dict(self.completion_lower_s),
            delivered_observed=dict(self.delivered_observed), delivered_committed=dict(self.delivered_committed),
            S_R=sorted(self.remaining_task_ids), K_F_observed=self.observed_tardiness_s,
            K_F_committed_upper=self.committed_tardiness_upper_s,
            K_F_is_fully_observed=not bool(self.delivered_committed),
            prefix=[dict(id=x.block.id,kind=x.block.kind,participants=x.block.participants,
                         start_s=x.start_s,end_s=x.end_s,observed=x.block.observed,evidence=x.block.evidence,block=_plain(x.block))
                    for x in self.prefix])
        if getattr(self.window, 'offline_frozen', False):
            result.update(time_semantics='offline_frozen_snapshot',
                          H_and_t_F_are_model_anchors_not_measured_confirmation=True)
        return result

    def validate_partial_for(self, item):

        if not isinstance(item, PartialBoundaryRecoveryInput):
            raise BoundaryUncertified('internal boundary subset type required')
        full = item.full_projection
        if type(full) is not BoundaryRecoveryInput or item.execution_boundary is not self:
            raise BoundaryUncertified('subset must derive from one complete boundary projection')
        self.validate_for(full)
        included = frozenset(item.task_map)
        if not included < frozenset(full.task_map):
            raise BoundaryUncertified('internal subset must be a strict subset of the complete suffix')
        if item.original_input is not full.original_input or item.source_snapshot is not full.source_snapshot:
            raise BoundaryUncertified('subset must preserve the original observed snapshot and input')
        for name in ('instance_id', 'roads', 'source_settings', 'events', 'changes'):
            if getattr(item, name) != getattr(full, name):
                raise BoundaryUncertified('subset changes physical conditions')
        if item.tasks != tuple(t for t in full.tasks if t.id in included):
            raise BoundaryUncertified('subset changes task attributes or order')
        expected_workers = tuple(_expected_worker(w, location=w.location, cargo=tuple(t for t in w.carried_task_ids if t in included))
                                 for w in full.workers)
        if item.workers != expected_workers:
            raise BoundaryUncertified('subset changes workers, custody or gate locations')
        expected_reference = {w: tuple(v for v in visits if v.task_id in included)
                              for w, visits in full.reference_sequences.items()}
        if dict(item.reference_sequences) != expected_reference:
            raise BoundaryUncertified('subset changes original suffix order')
        return self


def build_boundary(item, old_orders, blocks, window, holding_guarantees, routing, *, check_budget=lambda:None, withdrawal_deadlines=None, original_plan=None):


    if window.tau != item.time_s:
        raise BoundaryUncertified('response clock does not match anomaly snapshot')
    workers = item.worker_map
    domains=checked_activity_domains(item)


    routing=routing.for_activity_domain(None)
    withdrawal_deadlines=withdrawal_deadlines or {}
    withdrawn={w.id for w in item.workers if w.service_mode=='handover_only'}
    if withdrawn-set(withdrawal_deadlines):
        raise BoundaryUncertified('withdrawn workers need explicit cooperation declarations')
    for w,limit in withdrawal_deadlines.items():
        if w not in workers:
            raise BoundaryUncertified('unknown cooperation deadline worker')
        if limit is not None:
            _finite(limit,'cooperation deadline')
    if set(old_orders) != set(workers) or set(holding_guarantees) != set(workers):
        raise BoundaryUncertified('orders and holding declarations must cover all workers')
    blocks = tuple(blocks)
    index = {b.id:b for b in blocks}
    if len(index) != len(blocks):
        raise BoundaryUncertified('duplicate block IDs')
    occurrences = Counter((bid,w) for w,order in old_orders.items() for bid in order)
    expected = Counter((b.id,w) for b in blocks for w in b.participants)
    if occurrences != expected or any(w not in workers for b in blocks for w in b.participants):
        raise BoundaryUncertified('shared blocks must occur once in both original orders')
    declared_orders={w:tuple(('transfer',bid) if index[bid].kind=='transfer' else
                              (index[bid].kind,index[bid].task_id)
                              for bid in order if index[bid].kind in ('pickup','delivery','transfer'))
                     for w,order in old_orders.items()}
    source_orders=original_plan.sequences if original_plan is not None else item.reference_sequences
    expected_orders={w:tuple(('transfer',v.transfer_id) if v.operation=='transfer' else
                             (v.operation,v.task_id) for v in route) for w,route in source_orders.items()}
    if declared_orders!=expected_orders:
        raise BoundaryUncertified('action blocks must preserve the declared original service/shared-event order')
    if original_plan is not None:
        if (original_plan.input_id,original_plan.network_id,original_plan.snapshot_id)!=(item.instance_id,item.network_id,item.snapshot_id):
            raise BoundaryUncertified('original plan belongs to another input')
        for b in blocks:
            if b.kind=='transfer':
                e=original_plan.transfer_map.get(b.id)
                if e is None or (e.worker_i,e.worker_j)!=(b.participants[0],b.participants[1]) or (
                        e.location,e.tasks_i_to_j,e.tasks_j_to_i)!=(b.destination,b.tasks_i_to_j,b.tasks_j_to_i):
                    raise BoundaryUncertified('prefix modifies an original shared event')
    initial_held={t for w in item.workers for t in w.carried_task_ids}
    expected_services=Counter((t,'delivery') for t in item.task_map)
    expected_services.update((t,'pickup') for t in item.task_map if t not in initial_held)
    actual_services=Counter((b.task_id,b.kind) for b in blocks if b.kind in ('pickup','delivery'))
    if actual_services != expected_services:
        raise BoundaryUncertified('old action blocks must preserve all original task obligations exactly once')
    required = {b.id for b in blocks if b.observed or b.irrevocable or b.started_at_s is not None}
    changed = True
    while changed:
        check_budget()
        old = len(required)
        for w,order in old_orders.items():
            last = max((i for i,bid in enumerate(order) if bid in required), default=-1)
            required.update(order[:last+1])
        changed = len(required) != old
    positions = {w.id:w.location for w in item.workers}
    held = {w.id:set(w.carried_task_ids) for w in item.workers}
    clocks = {w:window.tau for w in workers}
    cursors = {w:0 for w in workers}
    stopped, completed = set(), set()
    observed, committed, executions = {}, {}, []
    picked = {t for inventory in held.values() for t in inventory}
    rank = {'travel':0,'pickup':1,'transfer':2,'delivery':3,'wait':4}
    while True:
        check_budget()
        candidates = []
        for b in blocks:
            check_budget()
            if b.id in completed or any(w in stopped for w in b.participants):
                continue
            if not all(cursors[w] < len(old_orders[w]) and old_orders[w][cursors[w]] == b.id for w in b.participants):
                continue
            start = max([b.ready_at_s] + [clocks[w] for w in b.participants])
            if b.kind in ('pickup','delivery'):
                task = item.task_map.get(b.task_id)
                if task is None:
                    raise BoundaryUncertified('unknown prefix task')
                if not b.observed and b.started_at_s is None:
                    ready = task.pickup_ready_time_s if b.kind == 'pickup' else task.delivery_ready_time_s
                    if ready is None:
                        raise BoundaryUncertified('pickup facts missing')
                    start = max(start, ready)
            if b.observed:
                start = b.observed_start_s


            candidates.append((start, rank[b.kind], tuple(sorted(b.participants)), b.id, b))
        if not candidates:
            break
        start,_,_,_,b = min(candidates)
        mandatory = b.id in required
        if not mandatory and (start >= window.activation_s or
                any(clocks[w] >= window.activation_s for w in b.participants) or
                not b.confirmed or not b.safe_to_complete or not b.evidence):
            stopped.update(b.participants)
            continue
        try:
            if not b.evidence or (not b.observed and (not b.confirmed or not b.safe_to_complete)):
                raise BoundaryUncertified(f'locked block {b.id} lacks safe completion evidence')
            if any(start < clocks[w] for w in b.participants):
                raise BoundaryUncertified('block starts before a required predecessor')
            if tuple(positions[w] for w in b.participants) != b.origins:
                raise BoundaryUncertified('block origin differs from confirmed physical predecessor')
            end = b.observed_end_s if b.observed else start + b.duration_upper_s
            _finite(end, 'completion upper bound')
            for w in b.participants:
                if w in domains:
                    activity=routing.for_activity_domain(domains[w]).activity_domain
                    if not (activity.contains_position(positions[w]) and activity.contains_position(b.destination)):
                        raise BoundaryUncertified('prefix exits declared activity domain')
            if b.observed:
                if end > window.confirmed_at_s:
                    raise BoundaryUncertified('future completion mislabeled as observed')


            else:
                for w in b.participants:
                    if withdrawal_deadlines.get(w) is not None and end>withdrawal_deadlines[w]:
                        raise BoundaryUncertified('prefix exceeds cooperation deadline')
                if b.started_at_s is not None and b.started_at_s > window.confirmed_at_s:
                    raise BoundaryUncertified('future action mislabeled as started')
                if b.kind == 'travel':
                    w = b.participants[0]
                    worker_routing=routing.for_activity_domain(domains.get(w))
                    duration = worker_routing.route_duration(positions[w], b.destination, workers[w].profile)
                    if not isfinite(duration) or duration > b.duration_upper_s:
                        raise BoundaryUncertified('travel completion bound is not feasible on current roads')
                elif any(positions[w] != b.destination for w in b.participants):
                    raise BoundaryUncertified('service/meeting requires explicit preceding travel blocks')
                if b.kind in ('pickup','delivery'):
                    destination = item.task_map[b.task_id].pickup if b.kind == 'pickup' else item.task_map[b.task_id].delivery
                    if b.destination != destination:
                        raise BoundaryUncertified('prefix service destination violates current task condition')
                if (b.kind in ('pickup','delivery') and b.started_at_s is None
                        and workers[b.participants[0]].service_mode != 'delivery'):
                    raise BoundaryUncertified('withdrawn worker cannot perform prefix service')
                if b.kind in ('pickup','delivery'):
                    required_duration=item.task_map[b.task_id].service_duration_s(b.kind)
                    insufficient=(end < b.started_at_s + required_duration if b.started_at_s is not None
                                  else b.duration_upper_s < required_duration)
                    if insufficient:
                        raise BoundaryUncertified('prefix service omits its declared duration')
                if b.kind == 'transfer':
                    if b.duration_upper_s + (max(0,window.tau-b.started_at_s) if b.started_at_s is not None else 0) < item.source_settings.transfer_duration_s:
                        raise BoundaryUncertified('transfer omits positive service duration')
                    if not item.source_settings.meeting_domain.contains(b.destination,item.roads):
                        raise BoundaryUncertified('prefix transfer outside meeting domain')
                    for w,outgoing,incoming in ((b.participants[0],b.tasks_i_to_j,b.tasks_j_to_i),
                                                (b.participants[1],b.tasks_j_to_i,b.tasks_i_to_j)):
                        if workers[w].service_mode=='handover_only' and incoming:
                            raise BoundaryUncertified('withdrawn worker receives prefix cargo')
                        if outgoing and routing.shortest_distance(workers[w].location,b.destination)>workers[w].handover_range_m:
                            raise BoundaryUncertified('prefix sender exceeds original anchor range')
        except BoundaryUncertified:
            if mandatory:
                raise
            stopped.update(b.participants)
            continue
        if b.kind == 'pickup':
            w=b.participants[0];t=b.task_id
            if t in picked or t in observed or t in committed:
                raise BoundaryUncertified('prefix pickup duplicates existing custody')
            held[w].add(t);picked.add(t)
        elif b.kind == 'delivery':
            w=b.participants[0];t=b.task_id
            if t not in held[w] or t in observed or t in committed:
                raise BoundaryUncertified('prefix delivery has no unique held item')
            held[w].remove(t)
            (observed if b.observed else committed)[t]=end
        elif b.kind == 'transfer':
            a,c=b.participants
            out,back=set(b.tasks_i_to_j),set(b.tasks_j_to_i)
            if out-held[a] or back-held[c] or out&held[c] or back&held[a]:
                raise BoundaryUncertified('atomic prefix transfer has inconsistent custody')
            held[a]=(held[a]-out)|back;held[c]=(held[c]-back)|out
        for w in b.participants:
            if _exceeds_capacity(fsum(item.task_map[t].load for t in held[w]),workers[w].capacity):
                raise BoundaryUncertified('prefix exceeds capacity')
            positions[w]=b.destination;clocks[w]=end;cursors[w]+=1
        completed.add(b.id);executions.append(PrefixExecution(b,start,end))
    if required-completed:
        raise BoundaryUncertified('locked/shared prefix cannot close without unresolved predecessors')
    releases,expiry,gates,completion = {},{},{},{}
    for w in workers:
        choices=[g for g in holding_guarantees[w] if g.position==positions[w]]
        if len(choices)!=1:
            raise BoundaryUncertified(f'{w}: exactly one holding/gate guarantee at selected boundary required')
        guarantee=choices[0]
        if w in domains and not domains[w].contains(positions[w],item.roads):
            raise BoundaryUncertified('release position outside activity domain')
        if withdrawal_deadlines.get(w) is not None:
            raise BoundaryUncertified('continuing custody cannot exceed a finite withdrawal cooperation deadline')
        release=max(window.activation_s,max(window.confirmed_at_s,clocks[w])+guarantee.gate_duration_s)
        releases[w]=release;expiry[w]=guarantee.until_s
        gates[w]=guarantee.gate_duration_s;completion[w]=max(window.confirmed_at_s,clocks[w])
    remaining=frozenset(item.task_map)-set(observed)-set(committed)
    inventory=Counter(t for values in held.values() for t in values)
    if any(count!=1 for count in inventory.values()) or set(inventory)-remaining:
        raise BoundaryUncertified('boundary inventory does not uniquely partition physical cargo')
    suffix_sequences={w:[] for w in workers}
    suffix_transfers=[]
    for b in blocks:
        if b.id in completed:
            continue
        if b.kind=='transfer':
            suffix_transfers.append(TransferEvent(id=b.id,worker_i=b.participants[0],worker_j=b.participants[1],
                location=b.destination,tasks_i_to_j=b.tasks_i_to_j,tasks_j_to_i=b.tasks_j_to_i))
    for w,order in old_orders.items():
        for bid in order[cursors[w]:]:
            b=index[bid]
            if b.kind in ('pickup','delivery'):
                suffix_sequences[w].append(RecoveryVisit(operation=b.kind,task_id=b.task_id))
            elif b.kind=='transfer':
                suffix_sequences[w].append(RecoveryVisit(operation='transfer',transfer_id=b.id))
    reference=RecoveryPlan(input_id=item.instance_id,network_id=item.network_id,snapshot_id=item.snapshot_id,
                          sequences={w:tuple(v) for w,v in suffix_sequences.items()},transfers=tuple(suffix_transfers))
    frozen=lambda values:MappingProxyType(dict(values))
    delivery_lower,completion_lower=prefix_physical_lower_bounds(item,executions,routing)
    contract=dict(window=_plain(window),old_orders=_plain(old_orders),blocks=_plain(blocks),
        holding_guarantees=_plain(holding_guarantees),withdrawal_deadlines=_plain(withdrawal_deadlines),
        original_plan=None if original_plan is None else original_plan.to_dict())
    if getattr(window, 'offline_frozen', False):
        contract['window_mode']='offline_frozen_snapshot'
    contract=_freeze(contract)
    return ExecutionBoundary(input_digest(item),frozenset(item.task_map),window,tuple(executions),
        frozen(positions),frozen(releases),frozen({w:tuple(sorted(v)) for w,v in held.items()}),
        frozen({w:workers[w].location for w in workers}),frozen(observed),frozen(committed),frozenset(remaining),
        fsum(max(0,t-item.task_map[s].promised_delivery_time_s) for s,t in observed.items()),
        fsum(max(0,t-item.task_map[s].promised_delivery_time_s) for s,t in committed.items()),frozen(expiry),item,reference,
        frozen(getattr(item.source_settings,'activity_domains',{})),frozen(gates),frozen(completion),
        frozen(delivery_lower),frozen(completion_lower),contract)


def prefix_physical_lower_bounds(item, prefix, routing):

    clocks={w.id:item.time_s for w in item.workers}
    domains=checked_activity_domains(item)
    delivered={}
    for execution in prefix:
        b=execution.block
        if b.observed:
            end=b.observed_end_s
        else:
            start=max([b.ready_at_s]+[clocks[w] for w in b.participants])
            if b.kind=='travel':
                wid=b.participants[0]
                worker_routing=routing.for_activity_domain(domains.get(wid))
                duration=worker_routing.route_duration(b.origins[0],b.destination,item.worker_map[wid].profile)
                end=start+duration
            elif b.kind=='transfer':
                elapsed=max(0,item.time_s-b.started_at_s) if b.started_at_s is not None else 0
                end=start+max(0,item.source_settings.transfer_duration_s-elapsed)
            elif b.kind in ('pickup','delivery'):
                task=item.task_map[b.task_id]
                ready=task.pickup_ready_time_s if b.kind=='pickup' else task.delivery_ready_time_s
                duration=task.service_duration_s(b.kind)
                if b.started_at_s is not None:
                    end=max(start,b.started_at_s+duration)
                else:
                    end=max(start,ready)+duration
            else:
                end=start
        for wid in b.participants:
            clocks[wid]=end
        if b.kind=='delivery':
            delivered[b.task_id]=end
    return delivered,clocks


@dataclass(frozen=True, kw_only=True)
class BoundaryRecoveryInput(RecoveryInput):

    original_input: RecoveryInput
    execution_boundary: ExecutionBoundary

    def validate(self):
        self.execution_boundary.validate_for(self)

    @property
    def time_s(self):
        return self.execution_boundary.H

    def to_dict(self):
        result=super().to_dict()
        result['projection_kind']='certified_committed_execution_boundary'
        result['execution_boundary']=self.execution_boundary.to_dict()
        return result


@dataclass(frozen=True, kw_only=True)
class PartialBoundaryRecoveryInput(BoundaryRecoveryInput):

    full_projection: BoundaryRecoveryInput

    def validate(self):
        self.execution_boundary.validate_partial_for(self)

    def to_dict(self):
        result = super().to_dict()
        result['projection_kind'] = 'internal_partial_boundary_search_only'
        result['complete_suffix_task_ids'] = sorted(self.full_projection.task_map)
        return result


def project_subset(full, included):

    if type(full) is not BoundaryRecoveryInput:
        raise BoundaryUncertified('complete boundary projection required for a subset')
    full.validate()
    included = frozenset(included)
    if not included < frozenset(full.task_map):
        raise BoundaryUncertified('strict subset of complete suffix tasks required')
    return PartialBoundaryRecoveryInput(
        instance_id=full.instance_id, source_snapshot=full.source_snapshot,
        source_settings=full.source_settings, roads=full.roads,
        tasks=tuple(t for t in full.tasks if t.id in included),
        workers=tuple(replace(w, carried_task_ids=tuple(t for t in w.carried_task_ids if t in included))
                      for w in full.workers),
        reference_sequences={w: tuple(v for v in visits if v.task_id in included)
                             for w, visits in full.reference_sequences.items()},
        events=full.events, changes=full.changes, original_input=full.original_input,
        execution_boundary=full.execution_boundary, full_projection=full)


def project_input(original, boundary):
    from common.initial_sequences.inputs import PlanningVisit
    boundary.validate_for(original)
    reference={w:tuple(PlanningVisit(operation=v.operation,task_id=v.task_id) for v in visits
                       if v.operation in ('pickup','delivery'))
               for w,visits in boundary.reference_plan.sequences.items()}
    return BoundaryRecoveryInput(instance_id=original.instance_id,source_snapshot=original.source_snapshot,
        source_settings=original.source_settings,roads=original.roads,
        tasks=tuple(t for t in original.tasks if t.id in boundary.remaining_task_ids),
        workers=tuple(replace(w,location=boundary.locations[w.id],carried_task_ids=boundary.inventories[w.id])
                      for w in original.workers), reference_sequences=reference,events=original.events,
        changes=original.changes,original_input=original,execution_boundary=boundary)


def boundary_from_contract(item, contract, routing, *, check_budget=lambda:None):

    from common.data.roads.schema import RoadPosition
    try:
        if contract['window_mode']!='offline_frozen_snapshot':
            raise BoundaryUncertified('invalid snapshot execution contract')
        window=FrozenSnapshotWindow(**contract['window'])
        blocks=[]
        for row in contract['blocks']:
            row=dict(row)
            row['participants']=tuple(row['participants'])
            row['origins']=tuple(RoadPosition(**p) for p in row['origins'])
            row['destination']=RoadPosition(**row['destination'])
            for field in ('tasks_i_to_j','tasks_j_to_i'):
                row[field]=tuple(row.get(field,()))
            blocks.append(ActionBlock(**row))
        holding={w:tuple(HoldingGuarantee(position=RoadPosition(**row['position']),
                          until_s=row.get('until_s'),gate_duration_s=row['gate_duration_s'],evidence=row['evidence'])
                          for row in rows) for w,rows in contract['holding_guarantees'].items()}
        original_plan=RecoveryPlan.from_dict(contract['original_plan']) if contract.get('original_plan') is not None else None
        result=build_boundary(item,{w:tuple(v) for w,v in contract['old_orders'].items()},blocks,
            window,holding,routing,check_budget=check_budget,
            withdrawal_deadlines=contract.get('withdrawal_deadlines'),original_plan=original_plan)
        return replace(result,execution_contract=_freeze(contract))
    except (KeyError,TypeError) as error:
        raise BoundaryUncertified(f'missing or invalid execution contract: {error}') from error


def boundary_to_contract(boundary):

    return _plain(boundary.execution_contract)
