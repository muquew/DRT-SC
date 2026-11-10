from dataclasses import dataclass,replace
import heapq
import math

from common.recovery.plan import RecoveryVisit,TransferEvent
from solvers.tfr.interfaces.revisions import activity_allowed,handover_possible,activity_domain
from solvers.tfr.priority import construction_priority

from collections.abc import Mapping
from types import MappingProxyType
from common.recovery.plan import RecoveryPlan

class _RouteView(Mapping):
    __slots__ = ('base','changes')
    def __init__(self,base,changes):
        object.__setattr__(self,'base',base);object.__setattr__(self,'changes',changes)
    def __setattr__(self,name,value):raise AttributeError('route view is immutable')
    def __getitem__(self,key):
        return self.changes[key] if key in self.changes else self.base[key]
    def __iter__(self):
        yield from self.base
        yield from (key for key in self.changes if key not in self.base)
    def __len__(self):return len(self.base)+sum(key not in self.base for key in self.changes)

class _PlanOverlay:
    __slots__ = ('_base','_changed','sequences','transfers','transfer_map')
    def __init__(self,base,changed=None,transfers=None):
        if isinstance(base,_PlanOverlay):
            root=base._base;routes=dict(base._changed)
            if changed is not None:routes.update(changed)
            if transfers is None:transfers=base.transfers
        else:
            root=base;routes={}if changed is None else dict(changed)
            if transfers is None:transfers=base.transfers
        routes={worker:tuple(route) for worker,route in routes.items() if route!=root.sequences.get(worker)}
        transfers=tuple(transfers)
        values=MappingProxyType(routes)
        object.__setattr__(self,'_base',root);object.__setattr__(self,'_changed',values)
        object.__setattr__(self,'sequences',_RouteView(root.sequences,values))
        object.__setattr__(self,'transfers',tuple(transfers))
        object.__setattr__(self,'transfer_map',root.transfer_map if transfers is root.transfers else
            MappingProxyType({event.id:event for event in transfers}))
    def __setattr__(self,name,value):raise AttributeError('private plan overlay is immutable')
    def __getattr__(self,name):return getattr(self._base,name)
    def to_dict(self):raise TypeError('materialize private overlay at yield before serialization')
    def __reduce__(self):raise TypeError('private overlay is not checkpoint or IPC state')

def _materialize_plan(plan):
    if not isinstance(plan,_PlanOverlay):return plan
    return replace(plan._base,sequences=dict(plan.sequences),transfers=plan.transfers)


@dataclass(frozen=True)
class HInsertion:
    plan: object
    worker_id: str
    original: object

    def materialize(self):return self.plan
    def changed_sequences(self):
        return {worker:route for worker,route in self.plan.sequences.items()
                if route!=self.original.sequences[worker]}


def _insert(plan,worker,index,visit):
    route=plan.sequences[worker]
    return _PlanOverlay(plan,{worker:route[:index]+(visit,)+route[index:]})


def _route_capacity_risk(item,plan,worker,route,cache,ignored=frozenset()):
    manifest=tuple((v.transfer_id,plan.transfer_map[v.transfer_id].tasks_i_to_j,
            plan.transfer_map[v.transfer_id].tasks_j_to_i) for v in route if v.operation=='transfer')
    key=worker,route,manifest,ignored
    if cache is not None and key in cache:return cache[key]
    held=set(item.worker_map[worker].carried_task_ids)
    row_count,row_excess=0,0.
    def measure():
        nonlocal row_count,row_excess
        load=math.fsum(item.task_map[t].load for t in held if t not in ignored)
        capacity=item.worker_map[worker].capacity
        if load-capacity>math.ulp(max(load,capacity)):
            row_count+=1;row_excess+=load-capacity
    measure()
    for visit in route:
        if visit.operation=='pickup':held.add(visit.task_id)
        elif visit.operation=='delivery':held.discard(visit.task_id)
        elif visit.operation=='transfer':
            event=plan.transfer_map[visit.transfer_id]
            outgoing,incoming=((event.tasks_i_to_j,event.tasks_j_to_i) if worker==event.worker_i
                    else (event.tasks_j_to_i,event.tasks_i_to_j))
            held.difference_update(outgoing);held.update(incoming)
        measure()
    if cache is not None:cache[key]=row_count,row_excess
    return row_count,row_excess


def capacity_risk(item,plan,*,override=None,cache=None,ignored=frozenset()):

    count,excess=0,0.
    for worker,original in plan.sequences.items():
        route=override[1] if override is not None and override[0]==worker else original
        a,b=_route_capacity_risk(item,plan,worker,route,cache,ignored)
        count+=a;excess+=b
    return count,excess


class _CapacityPriority:

    def __init__(self,item,base,cache,ignored=frozenset()):
        self.item,self.base,self.cache=item,base,cache
        self.ignored=frozenset(ignored)
        self.order={worker:i for i,worker in enumerate(base.sequences)}
        self.nonzero={}
        for worker,route in base.sequences.items():
            value=_route_capacity_risk(item,base,worker,route,cache,self.ignored)
            if value!=(0,0.):self.nonzero[worker]=value

    def score(self,plan,changed,*,override=None):
        if plan.transfers[:len(self.base.transfers)]!=self.base.transfers:
            return capacity_risk(self.item,plan,override=override,cache=self.cache,ignored=self.ignored)
        changed=set(changed)
        if override is not None:changed.add(override[0])
        members=changed|self.nonzero.keys()
        count,excess=0,0.
        for worker in sorted(members,key=self.order.__getitem__):
            if worker not in changed:a,b=self.nonzero[worker]
            else:
                route=override[1] if override is not None and override[0]==worker else plan.sequences[worker]
                a,b=_route_capacity_risk(self.item,plan,worker,route,self.cache,self.ignored)
            count+=a;excess+=b
        return count,excess


def insertions(engine,base,task_id,ledger,*,domain='full',pending_tasks=frozenset(),ablation='full'):

    if domain not in ('full','trail'):raise ValueError('unknown H construction domain')
    if ablation not in ('full','no_transfer','sender_position','task_tardiness'):
        raise ValueError('unknown H ablation')
    if ablation=='no_transfer' and base.transfers:return
    task=engine.item.task_map[task_id]
    local=engine.local;pool=[]
    holder=engine.holders.get(task_id)
    priority=construction_priority(engine)
    boundary=getattr(engine.item,'execution_boundary',None)
    release=lambda worker:engine.item.time_s if boundary is None else boundary.release_times[worker]
    def travel(worker,a,b):
        return priority.travel(worker,a,b)
    reach_cache=getattr(engine,'_h_fixed_gap_reachability',None)
    if reach_cache is None:reach_cache=engine._h_fixed_gap_reachability={}
    reach_stats=engine.stats.setdefault('h_fixed_gap_reachability',dict(
        point_pair_queries=0,point_pair_cache_hits=0,unreachable_gaps=0))
    def gap_options(plan,worker,position,*,start=0,fixed=False):
        route=plan.sequences[worker]
        values=priority.ranked_gaps(plan,worker,route,position,start=start)
        tail=len(route)


        if tail>=start and tail not in values:values=values[:7]+(tail,)
        if not fixed:return values
        router=engine.routing.for_activity_domain(activity_domain(engine.item,worker))
        def reachable(origin,destination):
            key=id(router),origin,destination
            if key in reach_cache:
                reach_stats['point_pair_cache_hits']+=1
                return reach_cache[key]
            reach_stats['point_pair_queries']+=1
            value=router.reachable(origin,destination)
            reach_cache[key]=value
            return value
        accepted=[]
        for gap in values:
            previous=next((engine.visit_location(plan,visit) for visit in reversed(route[:gap])
                if visit.operation!='transfer'),engine.item.worker_map[worker].location)
            following=next((engine.visit_location(plan,visit)for visit in route[gap:]
                if visit.operation!='transfer'),None)
            if reachable(previous,position) and (following is None or reachable(position,following)):
                accepted.append(gap)
            else:reach_stats['unreachable_gaps']+=1
        return tuple(accepted)
    def charge(worker):
        return ledger.charge_generation(task_id,worker)
    queue=[];serial=0;risk_cache={}


    risks=_CapacityPriority(engine.item,base,risk_cache,frozenset(pending_tasks))
    delivery=RecoveryVisit(operation='delivery',task_id=task_id)
    def compact(state):
        if len(state)==7:return state
        plan,*tail=state
        routes=(plan._changed if isinstance(plan,_PlanOverlay) and plan._base is base else
            {} if plan is base else {w:route for w,route in plan.sequences.items() if route!=base.sequences[w]})
        return (dict(routes),plan.transfers,*tail)
    def materialize(state):
        routes,events,*tail=state
        return (_PlanOverlay(base,routes,events),*tail)
    def estimate(plan,changed):
        return priority.change(base,plan,changed)
    def push(kind,state,cursor=None,gaps=(),index=0):
        nonlocal serial
        state=compact(state)
        plan,worker,minimum,position,arrival,hops=materialize(state)
        optimistic=max(task.delivery_ready_time_s,arrival+travel(worker,position,task.delivery))
        if kind in ('state','expand'):


            risk=(0,0.)
        elif kind=='delivery':
            gap=gaps[index];route=plan.sequences[worker]
            risk=risks.score(plan,state[0],override=(worker,route[:gap]+(delivery,)+route[gap:]))
        else:risk=risks.score(plan,state[0])
        scored=(_insert(plan,worker,gaps[index],delivery) if kind=='delivery' else plan)
        changed=set(state[0])|({worker} if kind=='delivery' else set())
        late,distance=estimate(scored,changed)
        if kind!='delivery':
            late+=max(0.,optimistic+task.delivery_service_time_s-task.promised_delivery_time_s)
            distance+=priority.distance(position,task.delivery)
        order={'state':0,'delivery':1,'expand':2}[kind]
        deferred=False
        if kind=='delivery':
            deferred=index>0 or any(risk) or not math.isfinite(late)
        elif kind=='expand':
            optional=(engine.item.worker_map[worker].service_mode=='delivery'
                and activity_allowed(engine.item,worker,task.delivery))
            reusable=False
            if optional:
                for event in plan.transfers:
                    if worker not in (event.worker_i,event.worker_j):continue
                    if task_id in event.tasks_i_to_j+event.tasks_j_to_i:continue
                    receiver=event.worker_j if worker==event.worker_i else event.worker_i
                    if receiver not in pool or engine.item.worker_map[receiver].service_mode!='delivery':continue
                    if not handover_possible(engine.item,worker,receiver,local.original_anchor(worker)):continue
                    left=next(i for i,v in enumerate(plan.sequences[worker])
                        if v.operation=='transfer' and v.transfer_id==event.id)
                    if left>=minimum:
                        reusable=True;break
            deferred=optional and not reusable
        heapq.heappush(queue,(int(deferred),*risk,late,distance,order,hops,serial,kind,state,cursor,gaps,index));serial+=1
    def transfers(state,receivers):
        state=compact(state)
        plan,worker,minimum,position,arrival,hops=materialize(state)
        for receiver in receivers:
            if receiver==worker:continue
            if not handover_possible(engine.item,worker,receiver,local.original_anchor(worker)):continue
            if not charge(receiver):continue
            anchor=local.original_anchor(worker)


            for event in plan.transfers:
                if {event.worker_i,event.worker_j}!={worker,receiver}:continue
                if task_id in event.tasks_i_to_j+event.tasks_j_to_i:continue
                if ablation=='sender_position':
                    from solvers.tfr.ablation import sender_anchor
                    senders=(sender for sender,tasks in ((event.worker_i,event.tasks_i_to_j),
                        (event.worker_j,event.tasks_j_to_i)) if tasks)
                    if any(sender_anchor(engine.item,sender)!=sender_anchor(engine.item,worker)
                            for sender in senders):continue
                left=next(i for i,v in enumerate(plan.sequences[worker])
                    if v.operation=='transfer' and v.transfer_id==event.id)
                right=next(i for i,v in enumerate(plan.sequences[receiver])
                    if v.operation=='transfer' and v.transfer_id==event.id)
                if left<minimum or not charge(receiver):continue
                updated=(replace(event,tasks_i_to_j=event.tasks_i_to_j+(task_id,))
                    if worker==event.worker_i else replace(event,tasks_j_to_i=event.tasks_j_to_i+(task_id,)))
                candidate=_PlanOverlay(plan,transfers=tuple(updated if old.id==event.id else old for old in plan.transfers))
                now=max(arrival+travel(worker,position,event.location),release(receiver)+travel(
                    receiver,engine.item.worker_map[receiver].location,event.location))+engine.item.source_settings.transfer_duration_s
                yield compact((candidate,receiver,right+1,event.location,now,hops+1))
            for left in gap_options(plan,worker,anchor,start=minimum):
                if not charge(receiver):break
                for right in gap_options(plan,receiver,anchor):
                    if not charge(receiver):break
                    event_id=f'h-init:{task_id}:{serial}'
                    while event_id in plan.transfer_map:event_id+=':next'
                    event=TransferEvent(id=event_id,worker_i=worker,worker_j=receiver,
                        location=anchor,tasks_i_to_j=(task_id,))
                    candidate=_PlanOverlay(plan,transfers=plan.transfers+(event,))
                    visit=RecoveryVisit(operation='transfer',transfer_id=event_id)
                    candidate=_insert(_insert(candidate,worker,left,visit),receiver,right,visit)
                    now=max(arrival+travel(worker,position,anchor),release(receiver)+travel(
                        receiver,engine.item.worker_map[receiver].location,anchor))+engine.item.source_settings.transfer_duration_s
                    following=compact((candidate,receiver,right+1,anchor,now,hops+1))


                    del candidate,plan
                    yield following
                    plan=materialize(state)[0]
    from itertools import chain
    rings=iter(local.iter_worker_rings(task_id))
    if holder is not None:
        first=next(rings,None)


        rings=chain((first if first is not None else (local.search_radius_m,()),),rings)
    expanded=[]
    for radius,new_workers in rings:


        previous=iter(tuple(expanded))
        pool.extend(new_workers)
        if holder is not None and not expanded:
            push('state',(base,holder,0,engine.item.worker_map[holder].location,release(holder),0))
        if holder is None:
            for worker in new_workers:
                if not activity_allowed(engine.item,worker,task.pickup) or not charge(worker):continue
                for gap in gap_options(base,worker,task.pickup,fixed=True):
                    if not charge(worker):break
                    arrival=max(release(worker)+travel(worker,engine.item.worker_map[worker].location,task.pickup),
                        task.pickup_ready_time_s,task.release_time_s or engine.item.time_s)
                    plan=_insert(base,worker,gap,RecoveryVisit(operation='pickup',task_id=task_id))
                    push('state',(plan,worker,gap+1,task.pickup,arrival,0))
        while True:
            if not queue:
                saved=next(previous,None)
                if saved is None:break
                push('expand',saved,cursor=transfers(saved,new_workers))
            engine.check_budget()
            *_,kind,state,cursor,gaps,index=heapq.heappop(queue)
            plan,worker,minimum,position,arrival,hops=materialize(state)
            if kind=='delivery':
                if ledger.can_emit(task_id,worker) and charge(worker):
                    candidate=_insert(plan,worker,gaps[index],delivery)
                    ledger.emit(task_id,worker)
                    yield HInsertion(_materialize_plan(candidate),worker,base)
                if index+1<len(gaps):push('delivery',state,gaps=gaps,index=index+1)
                continue
            if kind=='expand':
                following=next(cursor,None)
                if following is not None:
                    push('state',following);push('expand',state,cursor=cursor)
                continue
            if ablation!='no_transfer' and (domain=='full' or hops<2):
                expanded.append(state)
            allowed=(engine.item.worker_map[worker].service_mode=='delivery'
                and activity_allowed(engine.item,worker,task.delivery))
            if allowed and ledger.can_emit(task_id,worker):
                values=gap_options(plan,worker,task.delivery,start=minimum,fixed=True)
                def gap_order(gap):
                    route=plan.sequences[worker]
                    candidate=_insert(plan,worker,gap,delivery)
                    return (*risks.score(plan,state[0],
                        override=(worker,route[:gap]+(delivery,)+route[gap:])),
                        *estimate(candidate,set(state[0])|{worker}),values.index(gap))
                values=tuple(sorted(values,key=gap_order))
                if values:push('delivery',state,gaps=values)
            if ablation!='no_transfer' and (domain=='full' or hops<2):
                push('expand',state,cursor=transfers(state,tuple(pool)))
