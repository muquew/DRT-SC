from collections import OrderedDict
from heapq import nsmallest,heapify,heappop,heappush
import math
from statistics import median

from common.config import cache_limits


class ConstructionPriority:

    def __init__(self,engine):
        self.engine,self.item=engine,engine.item
        roads=self.item.roads
        sample=sorted({round((len(roads.arcs)-1)*i/127)for i in range(128)})if roads.arcs else ()
        ratios=[roads.arcs[i].travel_time_s/roads.segment_map[roads.arcs[i].segment_id].length_m
            for i in sample if roads.arcs[i].enabled and roads.segment_map[roads.arcs[i].segment_id].length_m>0]
        self.seconds_per_m=median(ratios)if ratios else 1.
        limits=cache_limits('h_structure')
        self.point_limit=limits['potential_entries'];self.route_limit=limits['estimate_entries']
        self.points,self.routes,self.joint=OrderedDict(),OrderedDict(),OrderedDict()
        self.stats=engine.stats.setdefault('h_priority',dict(coordinate_distances=0,route_scores=0,
            route_score_hits=0,gap_scores=0,calibration_samples=len(ratios),seconds_per_m=self.seconds_per_m,
            role='candidate_ordering',physical_acceptance=False))
        for name in ('joint_scores','joint_score_hits','joint_cycles','joint_workers_peak'):
            self.stats.setdefault(name,0)

    def point(self,position):
        if position in self.points:
            self.points.move_to_end(position);return self.points[position]
        value=self.item.roads.coordinates(position)
        if self.point_limit:
            self.points[position]=value
            if len(self.points)>self.point_limit:self.points.popitem(last=False)
        return value

    def distance(self,origin,destination):
        self.engine.check_budget();self.stats['coordinate_distances']+=1
        return math.dist(self.point(origin),self.point(destination))

    def travel(self,worker,origin,destination):
        return self.distance(origin,destination)*self.seconds_per_m/self.item.worker_map[worker].profile.speed_factor

    def ranked_gaps(self,plan,worker,route,position,*,start=0,stop=None,limit=8):
        stop=len(route)+1 if stop is None else min(stop,len(route)+1)
        origin=self.item.worker_map[worker].location
        def rank(index):
            self.stats['gap_scores']+=1
            a=origin if index==0 else self.engine.visit_location(plan,route[index-1])
            b=None if index==len(route)else self.engine.visit_location(plan,route[index])
            cost=self.distance(a,position)
            if b is not None:cost+=self.distance(position,b)-self.distance(a,b)
            return cost,index
        return tuple(index for _,index in nsmallest(limit,(rank(i)for i in range(start,stop))))

    def route(self,plan,worker):
        route=plan.sequences[worker]
        key=worker,route,tuple(plan.transfer_map[v.transfer_id].location for v in route if v.operation=='transfer')
        if key in self.routes:
            self.stats['route_score_hits']+=1;self.routes.move_to_end(key)
            return self.routes[key]
        self.stats['route_scores']+=1
        boundary=getattr(self.item,'execution_boundary',None)
        now=self.item.time_s if boundary is None else boundary.release_times[worker]
        position=self.item.worker_map[worker].location
        late,distance=0.,0.
        for visit in route:
            destination=self.engine.visit_location(plan,visit)
            length=self.distance(position,destination)
            distance+=length
            now+=length*self.seconds_per_m/self.item.worker_map[worker].profile.speed_factor
            if visit.operation=='pickup':
                task=self.item.task_map[visit.task_id]
                now=max(now,task.pickup_ready_time_s,task.release_time_s or self.item.time_s)+task.pickup_service_time_s
            elif visit.operation=='delivery':
                task=self.item.task_map[visit.task_id]
                now=max(now,task.delivery_ready_time_s)+task.delivery_service_time_s
                late+=max(0.,now-task.promised_delivery_time_s)
            else:now+=self.item.source_settings.transfer_duration_s
            position=destination
        value=late,distance
        if self.route_limit:
            self.routes[key]=value
            if len(self.routes)>self.route_limit:self.routes.popitem(last=False)
        return value

    def change(self,before,after,workers):


        affected=set(workers)
        if not affected:return 0.,0.
        neighbors={}
        for plan in (before,after):
            for event in plan.transfers:
                neighbors.setdefault(event.worker_i,set()).add(event.worker_j)
                neighbors.setdefault(event.worker_j,set()).add(event.worker_i)
        queue=list(affected)
        while queue:
            worker=queue.pop()
            for neighbor in neighbors.get(worker,()):
                if neighbor not in affected:
                    affected.add(neighbor);queue.append(neighbor)
        members=tuple(sorted(affected))
        self.stats['joint_workers_peak']=max(self.stats['joint_workers_peak'],len(members))
        previous=self._joint(before,members)
        following=self._joint(after,members)
        late=(math.inf if not math.isfinite(following[0]) else following[0]-previous[0])
        distance=following[1]-previous[1]
        return late,math.inf if math.isnan(distance) else distance

    def _joint(self,plan,workers):
        from common.evaluation.recovery_plan import _service_duration
        routes=tuple((worker,plan.sequences[worker])for worker in workers)
        transfers=tuple(plan.transfer_map[name]for name in sorted({visit.transfer_id
            for _,route in routes for visit in route if visit.operation=='transfer'}))
        key=routes,transfers
        if key in self.joint:
            self.stats['joint_score_hits']+=1;self.joint.move_to_end(key)
            return self.joint[key]
        self.stats['joint_scores']+=1
        boundary=getattr(self.item,'execution_boundary',None)
        times,degree,outgoing,deliveries={},{},{},{}
        distance=0.
        for worker,route in routes:
            self.engine.check_budget()
            previous=('origin',worker)
            times[previous]=self.item.time_s if boundary is None else boundary.release_times[worker]
            degree[previous]=0;outgoing[previous]=[]
            position=self.item.worker_map[worker].location
            speed=self.item.worker_map[worker].profile.speed_factor
            for index,visit in enumerate(route):
                destination=self.engine.visit_location(plan,visit)
                length=self.distance(position,destination);distance+=length
                duration=_service_duration(self.item,worker,index,visit)
                if visit.operation=='transfer':
                    node=('transfer',visit.transfer_id)
                    ready=self.item.time_s
                else:
                    node=('service',worker,index)
                    if visit.operation=='pickup':
                        task=self.item.task_map[visit.task_id]
                        release=self.item.time_s if task.release_time_s is None else task.release_time_s
                        ready=max(task.pickup_ready_time_s,release)
                    elif visit.operation=='delivery':
                        task=self.item.task_map[visit.task_id]
                        ready=task.delivery_ready_time_s;deliveries[visit.task_id]=(node,task.promised_delivery_time_s)
                    else:ready=max(self.item.time_s,getattr(visit,'not_before_s',None) or self.item.time_s)
                times[node]=max(times.get(node,-math.inf),ready+duration)
                degree[node]=degree.get(node,0)+1;outgoing.setdefault(node,[])
                outgoing[previous].append((node,length*self.seconds_per_m/speed+duration))
                previous,position=node,destination
        available=[node for node,count in degree.items()if not count];heapify(available)
        closed=0
        while available:
            node=heappop(available);closed+=1
            for child,lag in outgoing[node]:
                times[child]=max(times[child],times[node]+lag)
                degree[child]-=1
                if not degree[child]:heappush(available,child)
        if closed!=len(times):
            self.stats['joint_cycles']+=1;late=math.inf
        else:late=math.fsum(max(0.,times[node]-promise)for node,promise in deliveries.values())
        value=late,distance
        if self.route_limit:
            self.joint[key]=value
            if len(self.joint)>self.route_limit:self.joint.popitem(last=False)
        return value


def construction_priority(engine):
    value=getattr(engine,'h_construction_priority',None)
    if value is None:value=engine.h_construction_priority=ConstructionPriority(engine)
    return value
