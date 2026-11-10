from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
import math
from typing import Any

from common.recovery.plan import RecoveryPlan, RecoveryVisit, TransferEvent
from common.initial_sequences.inputs import _exceeds_capacity
from solvers.tfr.structure import (release_tasks, structure_key,
    check_structure, fixed_handover_bounds)

FEATURE_DIM = 16
FIELD_TYPES = ('group', 'pickup_worker', 'pickup_gap', 'receiver', 'meeting',
               'sender_gap', 'receiver_gap', 'delivery_gap', 'end')


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def source_witness_digest(item, evaluated):
    boundary = getattr(item, 'execution_boundary', None)
    return _digest(dict(input=getattr(item, 'fingerprint', item.instance_id),
        boundary=None if boundary is None else boundary.boundary_id,
        plan=evaluated.plan.to_dict(),
        timing={w: [(r.arrival_time_s, r.start_time_s, r.completion_time_s)
                    for r in rows] for w, rows in evaluated.routes.items()}))


def feedback_groups(evaluated,*,task_ids=None,limit=32,group_limit=8,analysis=None):

    from solvers.tfr.temporal import analyze
    analysis=analyze(evaluated) if analysis is None else analysis
    allowed=set(evaluated.tardiness_s if task_ids is None else task_ids)
    cache_key=frozenset(allowed),limit,group_limit
    if cache_key in analysis.group_candidates:return analysis.group_candidates[cache_key]
    single=[frozenset((t,)) for t,value in evaluated.tardiness_s.items() if value>0]
    preceding=analysis.preceding_groups()
    shared=[frozenset(event.tasks_i_to_j+event.tasks_j_to_i) for event in evaluated.plan.transfers
            if len(event.tasks_i_to_j+event.tasks_j_to_i)>=2]
    def key(group):
        late,waiting=analysis.group_score(group)
        return -late,-waiting,len(group),tuple(sorted(group))
    categories=[]
    for values in (single,preceding,shared):
        values={frozenset(group)&allowed for group in values}
        categories.append(iter(sorted((group for group in values if 0<len(group)<=group_limit),key=key)))
    result=[];seen=set()
    while categories and len(result)<limit:
        remaining=[]
        for category in categories:
            group=next(category,None)
            while group is not None and group in seen:group=next(category,None)
            if group is not None:
                seen.add(group);result.append(tuple(sorted(group)));remaining.append(category)
                if len(result)==limit:break
        categories=remaining
    result=tuple(result);analysis.group_candidates[cache_key]=result
    return result


def spatial_worker_pool(item,local,task_id,*,radius_m=None):

    return next(local.iter_worker_rings(task_id,radius_m=radius_m),(None,()))[1]


def activity_domain(item,worker):
    boundary=getattr(item,'execution_boundary',None)
    if boundary is not None and worker in boundary.activity_domains:return boundary.activity_domains[worker]
    return item.source_settings.activity_domains.get(worker)


def activity_allowed(item,worker,position):
    domain=activity_domain(item,worker)
    return domain is None or domain.contains(position,item.roads)


def handover_possible(item,sender,receiver,anchor):

    domains=[item.source_settings.meeting_domain,activity_domain(item,sender),activity_domain(item,receiver)]
    domains=[domain for domain in domains if domain is not None and not domain.all_road_positions]
    if not domains:return True
    parts=[]
    for domain in domains:
        nodes=set(domain.node_ids);segments={}
        for portion in domain.segment_portions:
            segment=item.roads.segment_map[portion.segment_id]
            segments.setdefault(segment.id,[]).append((portion.start_m,portion.end_m))
            if portion.start_m==0:nodes.add(segment.u)
            if portion.end_m==segment.length_m:nodes.add(segment.v)
        parts.append((nodes,segments))
    nodes=set.intersection(*(nodes for nodes,segments in parts))
    segments={key:list(value) for key,value in parts[0][1].items()}
    for other_nodes,other in parts[1:]:
        segments={key:[(max(a,c),min(b,d)) for a,b in intervals for c,d in other.get(key,()) if max(a,c)<=min(b,d)]
                  for key,intervals in segments.items() if key in other}
        segments={key:values for key,values in segments.items() if values}
    x,y=item.roads.coordinates(anchor);radius=item.worker_map[sender].handover_range_m
    def uncertain_near(dx,dy,coordinates):


        scale=max((abs(v) for v in (x,y,radius,*coordinates)),default=1.)
        margin=8*math.ulp(scale)
        return math.hypot(dx,dy)<=radius+margin
    for node in nodes:
        point=item.roads.node_map[node]
        if uncertain_near(x-point.x,y-point.y,(point.x,point.y)):return True
    for key,intervals in segments.items():


        geometry=item.roads.segment_map[key].geometry
        dx=max(min(p[0] for p in geometry)-x,0.,x-max(p[0] for p in geometry))
        dy=max(min(p[1] for p in geometry)-y,0.,y-max(p[1] for p in geometry))
        if uncertain_near(dx,dy,tuple(v for point in geometry for v in point)):return True
    return False


@dataclass
class RevisionObservation:
    item: Any
    evaluated: Any
    local: Any
    initial_T: float
    best_T: float
    round_index: int = 0
    stale_rounds: int = 0
    elapsed_s: float = 0.
    tried: set = field(default_factory=set)
    groups: tuple | None = None
    max_rounds: int = 50
    patience: int = 10
    revision_domain: str = 'full'
    check: Any = None
    temporal: Any = None
    witness_digest: str | None = None
    search_radii: dict = field(default_factory=dict)
    expandable: bool = True

    def __post_init__(self):
        if self.revision_domain!='full':raise ValueError('full service-structure reconstruction is required')
        self.source_witness_digest=self.witness_digest or source_witness_digest(self.item,self.evaluated)
        from solvers.tfr.temporal import analyze
        if self.temporal is None:self.temporal=analyze(self.evaluated,source_key=self.source_witness_digest)
        elif self.temporal.source_key!=self.source_witness_digest:raise ValueError('temporal analysis belongs to another source witness')
        self.groups=feedback_groups(self.evaluated,task_ids=self.item.task_map,analysis=self.temporal) if self.groups is None else tuple(self.groups)
        if len(self.groups) > 32 or any(not 0 < len(g) <= 8 for g in self.groups):
            raise ValueError('revision group domain exceeds the frozen protocol')
        boundary = getattr(self.item, 'execution_boundary', None)
        departure = None if boundary is None else boundary.release_times
        self.handover_bounds = fixed_handover_bounds(self.item, self.initial_T,
            original_plan=self.plan, departure_times=departure)
        self.worker_candidates = {}
        self.worker_pools = {}
        self.distance_cache = {}
        self.expansion_tasks=set()

    @property
    def plan(self):
        return self.evaluated.plan

    @property
    def routing(self):
        return self.local.routing

    def tick(self):
        if self.check is not None:
            self.check()

    def anchor(self, worker):
        return self.local.original_anchor(worker)

    def distance(self, a, b):
        key = a, b
        if key not in self.distance_cache:
            self.tick()
            try:
                self.distance_cache[key] = self.routing.shortest_distance(a, b)
            except (ValueError, KeyError):
                self.distance_cache[key] = math.inf
        return self.distance_cache[key]

    def workers(self, task, sender=None):
        key = task, sender, self.search_radii.get(task)
        if key not in self.worker_candidates:
            if task not in self.worker_pools:
                radius,pool=next(self.local.iter_worker_rings(task,
                    radius_m=self.search_radii.get(task)),(self.local.search_radius_m,()))
                self.search_radii[task]=radius
                self.worker_pools[task]=pool
            self.worker_candidates[key] = self.worker_pools[task]
        return self.worker_candidates[key]

    def expand_workers(self,task):

        self.workers(task)
        radius=self.local.next_search_radius(task,self.search_radii[task])
        if radius is None:return False
        previous=set(self.worker_pools[task])
        following=next((row for row in self.local.iter_worker_rings(task,radius_m=radius)
            if set(row[1])-previous),None)
        if following is None:return False
        radius,_=following
        self.search_radii[task]=radius
        self.worker_pools[task]=self.local.actor_pool(task,radius_m=radius)
        self.worker_candidates.clear()
        return True

    def progress(self):
        scale = max(self.initial_T, 1.)
        return ((self.max_rounds-self.round_index)/self.max_rounds,
            self.stale_rounds/max(1, self.patience), math.log1p(self.elapsed_s),
            self.evaluated.total_tardiness_s/scale, self.best_T/scale,
            len(self.tried)/max(1, self.max_rounds*3))


@dataclass(frozen=True)
class Candidate:
    identity: str
    value: Any
    features: tuple
    node: str | tuple[str, ...] | None = None
    legal: bool = True


@dataclass(frozen=True)
class FieldChoice:
    kind: str
    identity: str


@dataclass(frozen=True)
class RevisionScript:
    source_witness_digest: str
    choices: tuple[FieldChoice, ...]
    domains: tuple[tuple[str, ...], ...] = ()

    @property
    def group_id(self):
        return self.choices[0].identity if self.choices else ''


@dataclass
class FieldDecision:
    kind: str
    candidates: tuple[Candidate, ...]
    selected: int
    log_probability: float
    forced: bool = False


@dataclass
class ProposalSlot:
    script: RevisionScript | None
    fields: tuple[FieldDecision, ...]
    status: str
    structure: RecoveryPlan | None = None


@dataclass
class MacroAction:
    source_witness_digest: str
    slots: tuple[ProposalSlot, ...]
    graph: Any = None
    old_log_probability: float = 0.
    value: float = 0.
    update_step: int = 0
    search_radii: dict = field(default_factory=dict)
    expand_tasks: tuple = ()

    @property
    def candidates(self):
        return tuple(slot for slot in self.slots if slot.status == 'candidate')


class DecodeFailure(ValueError):
    pass


def _custody_gaps(obs, plan, worker, task, pending, *, receiving):


    route = plan.sequences[worker]
    ignored = set(pending) - {task}
    held = set(obs.item.worker_map[worker].carried_task_ids)
    states, last = [], -1
    for index, visit in enumerate(route):
        obs.tick()
        states.append(frozenset(held))
        if visit.operation == 'pickup':
            held.add(visit.task_id)
            involved = visit.task_id == task
        elif visit.operation == 'delivery':
            held.discard(visit.task_id)
            involved = visit.task_id == task
        else:
            event = plan.transfer_map[visit.transfer_id]
            outgoing, incoming = ((event.tasks_i_to_j, event.tasks_j_to_i)
                if worker == event.worker_i else (event.tasks_j_to_i, event.tasks_i_to_j))
            held.difference_update(outgoing); held.update(incoming)
            involved = task in outgoing or task in incoming
        if involved:last = index
    states.append(frozenset(held))
    capacity = obs.item.worker_map[worker].capacity
    result, exceeded = [], False
    for index in range(last+1, len(states)):
        inventory = states[index] - ignored
        if receiving:
            legal = task not in inventory and not _exceeds_capacity(
                math.fsum(obs.item.task_map[t].load for t in inventory | {task}), capacity)
        else:


            exceeded |= _exceeds_capacity(math.fsum(obs.item.task_map[t].load for t in inventory), capacity)
            legal = task in inventory and not exceeded
        if legal:result.append(index)
    return tuple(result)
