from collections import defaultdict
import json
import math
from types import SimpleNamespace

from solvers.tfr.temporal import analyze


class FeedbackPriority:


    def __init__(self, obs, group):
        from solvers.tfr.priority import ConstructionPriority
        from solvers.tfr.structure import release_tasks
        self.obs, self.item = obs, obs.item
        self.group = frozenset(group)
        self.plan = release_tasks(obs.plan, group)
        engine = SimpleNamespace(item=obs.item, stats={}, check_budget=obs.tick,
            visit_location=self._location)
        self.priority = ConstructionPriority(engine)
        self.stats = self.priority.stats
        self.stats.update(forecasts=0, forecast_hits=0)
        self._routes, self._base, self._scores = {}, {}, {}
        self._transfer_ready = {row.visit.transfer_id: row.start_time_s
            for rows in obs.evaluated.routes.values() for row in rows
            if row.visit.operation == 'transfer'}

    def _location(self, plan, visit):
        if visit.operation == 'transfer':return plan.transfer_map[visit.transfer_id].location
        task = self.item.task_map[visit.task_id]
        return task.pickup if visit.operation == 'pickup' else task.delivery

    def _release(self, worker):
        boundary = getattr(self.item, 'execution_boundary', None)
        return self.item.time_s if boundary is None else boundary.release_times[worker]

    def _service(self, task, operation):
        data = self.item.task_map[task]
        if operation == 'pickup':
            return (data.pickup, max(data.pickup_ready_time_s,
                data.release_time_s if data.release_time_s is not None else self.item.time_s),
                data.pickup_service_time_s, None, task)
        return data.delivery, data.delivery_ready_time_s, data.delivery_service_time_s, data.promised_delivery_time_s, task

    def _route(self, worker):
        if worker not in self._routes:
            self._routes[worker] = tuple(
                (self._location(self.plan, visit), self._transfer_ready.get(visit.transfer_id, self.item.time_s),
                 self.item.source_settings.transfer_duration_s, None, None)
                if visit.operation == 'transfer' else self._service(visit.task_id, visit.operation)
                for visit in self.plan.sequences[worker])
        return self._routes[worker]

    def _schedule(self, worker, route, task):
        now, position = self._release(worker), self.item.worker_map[worker].location
        late, travel, completion = [], [], math.inf
        for destination, ready, service, promise, identifier in route:
            self.obs.tick()
            duration = self.priority.travel(worker, position, destination)
            travel.append(duration)
            now = max(now + duration, ready) + service
            if promise is not None:
                late.append(max(0., now-promise))
                if identifier == task:completion = now
            position = destination
        return math.fsum(late), completion, math.fsum(travel)

    def _rank(self, task, worker, packet, key, *, receiving):
        if key in self._scores:
            self.stats['forecast_hits'] += 1
            return self._scores[key]
        self.stats['forecasts'] += 1
        route = self._route(worker)
        if worker not in self._base:self._base[worker] = self._schedule(worker, route, None)
        before, _, old_travel = self._base[worker]
        from solvers.tfr.interfaces.revisions import _custody_gaps
        allowed = set(_custody_gaps(self.obs, self.plan, worker, task, self.group, receiving=receiving))
        ranked = self.priority.ranked_gaps(self.plan, worker, self.plan.sequences[worker], packet[0][0],
            limit=len(route)+1)
        gaps = tuple(gap for gap in ranked if gap in allowed)[:8]
        if len(route) in allowed and len(route) not in gaps:gaps = gaps[:7] + (len(route),)
        values = []
        for gap in gaps:
            late, finish, travel = self._schedule(worker, route[:gap]+packet+route[gap:], task)
            rank = late-before, finish, travel-old_travel
            values.append(rank if all(map(math.isfinite, rank)) else (math.inf,)*3)
        result = min(values, default=(math.inf,)*3)
        self._scores[key] = result
        return result

    def pickup(self, task, worker):

        return self._rank(task, worker, (self._service(task, 'pickup'), self._service(task, 'delivery')),
            ('pickup', task, worker), receiving=True)

    def receiver(self, task, sender, receiver):

        delivery = self._service(task, 'delivery')
        if receiver is None:return self._rank(task, sender, (delivery,), ('direct', task, sender), receiving=False)
        anchor = self.obs.anchor(sender)
        arrival = self._release(sender) + self.priority.travel(sender, self.item.worker_map[sender].location, anchor)
        packet = ((anchor, arrival, self.item.source_settings.transfer_duration_s, None, None), delivery)
        return self._rank(task, receiver, packet, ('receiver', task, sender, receiver), receiving=True)


def select_groups(groups, global_attempts, slots=3):

    if type(slots)is not int or slots<0:
        raise ValueError('slots must be a nonnegative integer')
    unique=tuple(dict.fromkeys(tuple(sorted(group))for group in groups))
    def attempts(group):
        key=json.dumps(group,separators=(',',':'))
        return global_attempts.get(key,global_attempts.get(group,0))
    return tuple(group for _,group in sorted(enumerate(unique),
        key=lambda row:(attempts(row[1]),row[0]))[:slots])


def preferred_exchange(obs, group):

    tasks=tuple(sorted(set(group)))
    if len(tasks)!=2:return {}
    holders={task:worker.id for worker in obs.item.workers for task in worker.carried_task_ids}
    if any(task not in holders or task not in obs.evaluated.tardiness_s for task in tasks):return {}
    first,second=(holders[task]for task in tasks)
    if first==second:return {}
    workers=obs.item.worker_map
    if any(workers[worker].service_mode!='delivery'for worker in (first,second)):return {}
    try:
        current=math.fsum(obs.evaluated.tardiness_s[task]for task in tasks)
        delta=obs.item.source_settings.transfer_duration_s
        anchor=obs.anchor(first)
        boundary=getattr(obs.item,'execution_boundary',None)
        release=({worker:obs.item.time_s for worker in (first,second)}
            if boundary is None else boundary.release_times)
        def travel(worker,origin,destination):
            return 0. if origin==destination else obs.routing.travel_time(origin,destination,workers[worker].profile)
        arrivals=[release[worker]+travel(worker,workers[worker].location,anchor)
            for worker in (first,second)]
        if not all(math.isfinite(value)for value in (current,delta,*arrivals)):return {}
        end=max(arrivals)+delta
        tardiness=[]
        for task,receiver in zip(tasks,(second,first)):
            data=obs.item.task_map[task]
            arrival=end+travel(receiver,anchor,data.delivery)
            if not all(math.isfinite(value)for value in (arrival,data.delivery_ready_time_s,
                    data.delivery_service_time_s,data.promised_delivery_time_s)):return {}
            completion=max(arrival,data.delivery_ready_time_s)+data.delivery_service_time_s
            tardiness.append(max(0.,completion-data.promised_delivery_time_s))
        estimate=math.fsum(tardiness)
    except (ValueError,KeyError,OverflowError):return {}
    return {tasks[0]:second,tasks[1]:first}if math.isfinite(estimate)and estimate<current else {}


def own_tardiness_groups(evaluated, limit=32, group_limit=8):


    if type(limit)is not int or not 1<=limit<=32:
        raise ValueError('limit must be an integer in 1..32')
    if type(group_limit)is not int or not 1<=group_limit<=8:
        raise ValueError('group_limit must be an integer in 1..8')
    tasks=set(evaluated.tardiness_s)
    parents={task:task for task in tasks}
    def root(task):
        while parents[task]!=task:
            parents[task]=parents[parents[task]];task=parents[task]
        return task
    for event in evaluated.plan.transfers:
        cargo=tuple(event.tasks_i_to_j+event.tasks_j_to_i)
        for task in cargo[1:]:
            left,right=root(cargo[0]),root(task)
            if left!=right:parents[max(left,right)]=min(left,right)
    components=defaultdict(set)
    for task in sorted(tasks):components[root(task)].add(task)
    groups=[tuple(sorted(group))for group in components.values()if len(group)<=group_limit]
    return tuple(sorted(groups,key=lambda group:(
        -math.fsum(evaluated.tardiness_s[task]for task in group),len(group),group))[:limit])


def joint_feedback_groups(evaluated, existing, limit=32, group_limit=8, *, analysis=None):


    if type(limit)is not int or not 1<=limit<=32:
        raise ValueError('limit must be an integer in 1..32')
    if type(group_limit)is not int or not 1<=group_limit<=8:
        raise ValueError('group_limit must be an integer in 1..8')
    tasks=set(evaluated.tardiness_s)
    parents={task:task for task in tasks}
    def root(task):
        while parents[task]!=task:
            parents[task]=parents[parents[task]];task=parents[task]
        return task
    for event in evaluated.plan.transfers:
        cargo=tuple(event.tasks_i_to_j+event.tasks_j_to_i)
        for task in cargo[1:]:
            left,right=root(cargo[0]),root(task)
            if left!=right:parents[max(left,right)]=min(left,right)
    components=defaultdict(set)
    for task in sorted(tasks):components[root(task)].add(task)
    atoms={task:frozenset(components[root(task)])for task in tasks}
    def close(values):
        values=frozenset(values)
        if not values or not values<=tasks:return None
        group=frozenset().union(*(atoms[task]for task in values))
        return group if len(group)<=group_limit else None
    def visit_group(row):
        visit=row.visit
        if visit.operation=='transfer':
            event=evaluated.plan.transfer_map[visit.transfer_id]
            return close(event.tasks_i_to_j+event.tasks_j_to_i)
        return close((visit.task_id,))if visit.task_id is not None else None
    base=[];seen=set()
    for group in existing:
        group=close(group)
        if group is not None and group not in seen:
            base.append(group);seen.add(group)

    for task in sorted(tasks,key=lambda task:(-evaluated.tardiness_s[task],task)):
        group=atoms[task]
        if len(group)==1 and group not in seen:base.append(group);seen.add(group)
    joint=set()
    def add(values):
        group=close(values)
        if group is not None and len(group)>1:joint.add(group)
    def windows(tokens):
        for end in range(len(tokens)):
            group=frozenset()
            for token in reversed(tokens[max(0,end-group_limit+1):end+1]):
                group=group|token
                if len(group)>group_limit:break
                if len(group)>1:joint.add(group)
    initial=defaultdict(list)
    for worker,route in sorted(evaluated.routes.items()):
        chain=[]
        for row in route:
            group=visit_group(row)
            if group is None:
                windows(chain);chain=[]
            elif not chain or group!=chain[-1]:chain.append(group)
        windows(chain)
        if route and route[0].visit.operation=='delivery':
            group=visit_group(route[0])
            if group is not None:initial[route[0].incoming_path.origin].append(group)


    for tokens in initial.values():windows(tokens)
    temporal=analysis if analysis is not None else analyze(evaluated)
    node_groups={node:close(task for _,row in rows for task in (visit_group(row)or()))
        for node,rows in temporal.node_rows.items()}
    for node,predecessors in temporal.tight_predecessors.items():
        if node_groups[node]is None:continue
        add(set(node_groups[node]).union(*(node_groups[previous]or set()for previous in predecessors)))
    def priority(group):
        late,waiting=temporal.group_score(group)
        return -late,-waiting,len(group),tuple(sorted(group))
    ranked=iter(sorted(joint,key=priority));base=iter(base)
    result=[];used=set();streams=[base,ranked]
    while streams and len(result)<limit:
        following=[]
        for stream in streams:
            group=next(stream,None)
            while group is not None and group in used:group=next(stream,None)
            if group is not None:
                used.add(group);result.append(tuple(sorted(group)));following.append(stream)
                if len(result)==limit:break
        streams=following
    return tuple(result)
