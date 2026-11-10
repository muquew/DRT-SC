from collections import defaultdict,deque
import heapq
import math


class TemporalAnalysis:
    def __init__(self,value,*,source_key=None,tolerance=1e-7):
        self.source_key=source_key
        self.plan=value.plan
        self.visit_nodes={}
        self.rows={}
        self.node_rows=defaultdict(list)
        self.successors=defaultdict(set)
        self.tight_predecessors=defaultdict(set)
        self.tight_visit_edges=set()
        self.task_nodes=defaultdict(set)
        self.task_events=defaultdict(set)
        self.meeting_visits=defaultdict(dict)
        self.delivery_nodes={}
        self._preceding=None
        self._group_scores={}
        self.group_candidates={}
        nodes=set()
        for worker,route in value.routes.items():
            previous=None
            for index,row in enumerate(route):
                key=(worker,index);visit=row.visit
                node=('meeting',visit.transfer_id) if visit.operation=='transfer' else ('visit',worker,index)
                self.visit_nodes[key]=node;self.rows[key]=row;self.node_rows[node].append((key,row));nodes.add(node)
                if previous is not None:
                    old_key,old_node,old_row=previous
                    self.successors[old_node].add(node)
                    if (abs(row.arrival_time_s-row.start_time_s)<=tolerance and
                            abs(old_row.completion_time_s+row.incoming_path.duration_s-row.start_time_s)<=tolerance):
                        self.tight_predecessors[node].add(old_node)
                        self.tight_visit_edges.add((old_key,key))
                previous=key,node,row
                if visit.operation=='transfer':
                    event=value.plan.transfer_map[visit.transfer_id]
                    self.meeting_visits[event.id][worker]=key
                    for task in event.tasks_i_to_j+event.tasks_j_to_i:
                        self.task_nodes[task].add(node);self.task_events[task].add(event.id)
                elif visit.task_id is not None:
                    self.task_nodes[visit.task_id].add(node)
                    if visit.operation=='delivery':self.delivery_nodes[visit.task_id]=node
        indegree=dict.fromkeys(nodes,0)
        for previous,following in self.successors.items():
            for node in following:indegree[node]+=1
        queue=[node for node in nodes if indegree[node]==0];heapq.heapify(queue);order=[]
        while queue:
            node=heapq.heappop(queue);order.append(node)
            for following in sorted(self.successors[node]):
                indegree[following]-=1
                if indegree[following]==0:heapq.heappush(queue,following)
        if len(order)!=len(nodes):raise ValueError('certified source has a cyclic temporal dependency graph')
        self.late_tasks=tuple(sorted(t for t,late in value.tardiness_s.items() if late>0))
        self.weights=tuple(value.tardiness_s[t] for t in self.late_tasks)
        bits={self.delivery_nodes[t]:1<<index for index,t in enumerate(self.late_tasks)}
        self.downstream_bits={};self._weighted={0:0.}
        for node in reversed(order):
            mask=bits.get(node,0);total=self.weight(mask)
            for following in sorted(self.successors[node]):
                child=self.downstream_bits[following]
                addition=child & ~mask
                if not addition:continue
                overlap=child & mask
                if not overlap:total=math.fsum((total,self.weight(child)))
                elif addition.bit_count()<=overlap.bit_count():total=math.fsum((total,self.weight(addition)))
                else:total=math.fsum((total,self.weight(child),-self.weight(overlap)))
                mask |= child
            self.downstream_bits[node]=mask
            self._weighted.setdefault(mask,total)
        self.event_wait={identifier:max(value.routes[w][index].waiting_time_s for w,index in participants.values())
                         for identifier,participants in self.meeting_visits.items()}
        self.task_downstream_bits={task:self.union(self.downstream_bits[node] for node in related)
                                   for task,related in self.task_nodes.items()}

    @staticmethod
    def union(values):
        result=0
        for value in values:result |= value
        return result

    def weight(self,mask):
        if mask not in self._weighted:
            remaining=mask;values=[]
            while remaining:
                bit=remaining & -remaining;values.append(self.weights[bit.bit_length()-1]);remaining-=bit
            self._weighted[mask]=math.fsum(values)
        return self._weighted[mask]

    def visit_downstream(self,key):return self.weight(self.downstream_bits[self.visit_nodes[key]])
    def event_downstream(self,identifier):return self.weight(self.downstream_bits[('meeting',identifier)])
    def task_downstream(self,task):return self.weight(self.task_downstream_bits.get(task,0))
    def task_wait(self,task):return math.fsum(self.event_wait[event] for event in self.task_events.get(task,()))

    def group_score(self,group):
        group=frozenset(group)
        if group in self._group_scores:return self._group_scores[group]
        mask=self.union(self.task_downstream_bits.get(task,0) for task in group)
        events=set().union(*(self.task_events.get(task,set()) for task in group))
        result=self.weight(mask),math.fsum(self.event_wait[event] for event in events)
        self._group_scores[group]=result
        return result

    def preceding_groups(self):

        if self._preceding is not None:return self._preceding
        groups=set();pending=deque(self.delivery_nodes[task] for task in self.late_tasks);seen=set()
        while pending:
            node=pending.popleft()
            if node in seen:continue
            seen.add(node)
            if node[0]=='meeting':
                event=self.plan.transfer_map[node[1]]
                group=frozenset(event.tasks_i_to_j+event.tasks_j_to_i)
            else:
                row=self.rows[node[1],node[2]]
                group=frozenset((row.visit.task_id,)) if row.visit.task_id is not None else frozenset()
            if group:groups.add(group)
            pending.extend(sorted(self.tight_predecessors[node]))
        self._preceding=tuple(sorted(groups,key=lambda group:(len(group),tuple(sorted(group)))))
        return self._preceding


def analyze(value,*,source_key=None):return TemporalAnalysis(value,source_key=source_key)
