from contextlib import nullcontext
from collections import OrderedDict
from common.config import cache_limits
from dataclasses import dataclass, replace
from fractions import Fraction as F
import heapq
from solvers.tfr.geometry.features import DECISION_WINDOW; from solvers.tfr.geometry.features import GeometryNodeCandidate; from solvers.tfr.geometry.features import GeometryNodeDecision
from solvers.tfr.geometry.features import summarize_node
import time
import math
from types import SimpleNamespace

from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from common.data.roads.schema import RoadPosition
from common.evaluation.recovery_plan import evaluate_recovery_plan, RecoveryPlanValidationError, _advance_time, _service_duration
from common.recovery.methods.roads_exact import ExactRoads
from common.recovery.methods.numerics import public_cost_underestimate_bound
from common.recovery.inputs import checked_activity_domains
from solvers.tfr.geometry.bounds import affine_cell_minimum; from solvers.tfr.geometry.bounds import domain_travel_minimum; from solvers.tfr.geometry.bounds import domain_travel_minimum_steps; from solvers.tfr.geometry.bounds import complete_steps
from solvers.tfr.geometry.initialization import cost_steps; from solvers.tfr.geometry.initialization import meeting_cells_steps


_IN_PROGRESS = object()


@dataclass(frozen=True)
class Cell:

    segment: str | None = None
    low: F = F(0)
    high: F = F(0)
    point: object = None


@dataclass
class TravelBranch:
    constant: F
    coefficients: dict

    conditions: tuple = ()


@dataclass(frozen=True)
class GeometryNode:

    domains: tuple
    modes: tuple | None = None
    depth: int = 0
    label: tuple = ()


def _intersection(roads, left, right):
    lp, li = left
    rp, ri = right
    lp, rp = set(lp), set(rp)
    def contains(p, points, intervals):
        return p in points or (p.segment_id is not None and any(
            a <= p.offset_m <= b for a, b in intervals.get(p.segment_id, ())))
    points = {p for p in lp | rp if contains(p, lp, li) and contains(p, rp, ri)}
    intervals = {}
    for segment in li.keys() & ri.keys():
        for a, b in li[segment]:
            for c, d in ri[segment]:
                roads.check_budget()
                low, high = max(a, c), min(b, d)
                if low <= high:
                    points.update((roads.position(segment, low), roads.position(segment, high)))
                    if low < high:
                        intervals.setdefault(segment, []).append((low, high))
    return tuple(points), intervals


def meeting_cells(item, event, roads, activity_domains=None, permission_anchors=None, resources=None):

    if permission_anchors is None:
        original = getattr(item, 'original_input', item)
        permission_anchors = {w.id: w.location for w in original.workers}
    expand = roads.domain_parts if resources is None else lambda d: resources.domain_parts(roads, d)
    points, parts = expand(item.source_settings.meeting_domain)
    allowed = tuple(roads.normalize(p) for p in points), parts
    for wid, sent in ((event.worker_i, event.tasks_i_to_j),
                      (event.worker_j, event.tasks_j_to_i)):
        worker = item.worker_map[wid]
        if sent:
            allowed = _intersection(roads, allowed, roads.reachable_parts(
                permission_anchors[wid], worker.handover_range_m))
        if activity_domains and wid in activity_domains:
            points, parts = expand(activity_domains[wid])
            allowed = _intersection(roads, allowed,
                                    (tuple(roads.normalize(p) for p in points), parts))
    points, parts = allowed
    cells = [Cell(segment=s, low=F(a), high=F(b))
             for s in sorted(parts) for a, b in sorted(set(parts[s]))]


    cells.extend(Cell(point=p) for p in sorted(points, key=lambda p:
                 (p.node_id or '', p.segment_id or '', p.offset_m or 0)))
    return tuple(cells)


def _add(a, b):
    result = dict(a)
    for key, value in b.items():
        result[key] = result.get(key, F(0)) + value
    return {k: v for k, v in result.items() if v}


def travel_branches(roads, origin, destination, cells, speed):


    speed = F(speed)
    def fixed(value):
        return not isinstance(value, str) or cells[value].point is not None
    def position(value):
        return roads.normalize(cells[value].point if isinstance(value, str) else value)
    if fixed(origin) and fixed(destination):
        value = roads.cost(position(origin), position(destination), speed=speed)
        return [] if value is None else [TravelBranch(value, {})]
    if fixed(origin) or fixed(destination):
        incoming = fixed(origin)
        variable, anchor = (destination, origin) if incoming else (origin, destination)
        cell = cells[variable]
        result = []
        for branch in roads.affine_branches(position(anchor), cell.segment, incoming, speed=speed):
            if max(cell.low, branch.lo) <= min(cell.high, branch.hi):
                result.append(TravelBranch(branch.intercept, {variable: branch.slope},
                    (({variable: -F(1)}, -branch.lo), ({variable: F(1)}, branch.hi))))


        if not roads._arcs[cell.segment]:
            anchor_position = position(anchor)
            segment = roads.roads.segment_map[cell.segment]
            offsets = ([anchor_position.offset_m]
                       if anchor_position.segment_id == cell.segment else [])
            if anchor_position.node_id == segment.u:
                offsets.append(F(0))
            if anchor_position.node_id == segment.v:
                offsets.append(roads._lengths[cell.segment])
            for offset in offsets:
                if cell.low <= offset <= cell.high and roads.activity_contains(anchor_position):
                    result.append(TravelBranch(F(0), {},
                        (({variable: -F(1)}, -offset), ({variable: F(1)}, offset))))
        return result
    a, b = cells[origin], cells[destination]
    sa, sb = roads.roads.segment_map[a.segment], roads.roads.segment_map[b.segment]
    la, lb = roads._lengths[a.segment], roads._lengths[b.segment]
    result = []
    for first in roads._arcs[a.segment]:
        for last in roads._arcs[b.segment]:
            roads.check_budget()
            start = sa.v if first.forward else sa.u
            end = sb.u if last.forward else sb.v
            origin_interval = roads.endpoint_interval(a.segment, la if first.forward else F(0))
            destination_interval = roads.endpoint_interval(b.segment, F(0) if last.forward else lb)
            if (origin_interval is None or destination_interval is None
                    or max(a.low, origin_interval[0]) > min(a.high, origin_interval[1])
                    or max(b.low, destination_interval[0]) > min(b.high, destination_interval[1])):
                continue
            middle = roads._node_cost(start, end, 'time')
            if middle is None:
                continue
            ua, ub = roads._unit(first, 'time') / speed, roads._unit(last, 'time') / speed
            ca, xa = (ua * la, -ua) if first.forward else (F(0), ua)
            cb, xb = (F(0), ub) if last.forward else (ub * lb, -ub)
            conditions = ()
            if roads.activity is not None:
                conditions = (({origin: -F(1)}, -origin_interval[0]),
                              ({origin: F(1)}, origin_interval[1]),
                              ({destination: -F(1)}, -destination_interval[0]),
                              ({destination: F(1)}, destination_interval[1]))
            result.append(TravelBranch(ca + middle / speed + cb,
                                       _add({origin: xa}, {destination: xb}), conditions))
    if a.segment == b.segment:
        if not roads._arcs[a.segment]:


            equality = _add({origin: F(1)}, {destination: -F(1)})
            for low, high in roads.allowed_intervals(a.segment):
                low, high = max(low, a.low, b.low), min(high, a.high, b.high)
                if low <= high:
                    result.append(TravelBranch(F(0), {},
                        ((equality, F(0)), ({k: -v for k, v in equality.items()}, F(0)),
                         ({origin: -F(1)}, -low), ({origin: F(1)}, high))))
        for arc in roads._arcs[a.segment]:
            unit = roads._unit(arc, 'time') / speed
            sign = F(1 if arc.forward else -1)
            coeff = _add({origin: -sign * unit}, {destination: sign * unit})
            order = _add({origin: sign}, {destination: -sign})
            for low, high in roads.allowed_intervals(a.segment):
                if max(a.low, low) > min(a.high, high) or max(b.low, low) > min(b.high, high):
                    continue
                conditions = ((order, F(0)),)
                if roads.activity is not None:
                    conditions += (({origin: -F(1)}, -low), ({origin: F(1)}, high),
                                   ({destination: -F(1)}, -low), ({destination: F(1)}, high))
                result.append(TravelBranch(F(0), coeff, conditions))
    return result


class JointGeometryLP:


    def _compile(self):
        item = self.instance
        for wid, visits in sorted(self.plan.sequences.items()):
            previous = ('origin', wid)
            self._nodes[previous] = (F(self.departure_times[wid]), F(0), item.worker_map[wid].location)
            for k, visit in enumerate(visits):
                self.check_budget()
                if visit.operation == 'transfer':
                    node = ('transfer', visit.transfer_id)
                    self._nodes[node] = (F(item.time_s), F(item.source_settings.transfer_duration_s), visit.transfer_id)
                elif visit.operation in ('pickup', 'delivery'):
                    task = item.task_map[visit.task_id]
                    node = ('service', wid, k)
                    ready = task.pickup_ready_time_s if visit.operation == 'pickup' else task.delivery_ready_time_s
                    location = task.pickup if visit.operation == 'pickup' else task.delivery
                    self._nodes[node] = (F(ready), F(_service_duration(item,wid,k,visit)), location)
                    if visit.operation == 'delivery':
                        self._deliveries[task.id] = node
                else:
                    raise ValueError('joint geometry supports only pickup, delivery and transfer events')
                self._edges.append((previous, node, wid))
                previous = node
        if set(self._deliveries) != set(item.task_map):
            raise ValueError('joint geometry requires every remaining delivery')

    def _partition_lp_components(self):
        workers = {wid for event in self.plan.transfers for wid in (event.worker_i,event.worker_j)}
        if not self.plan.transfers:workers = set(self.plan.sequences)
        self._component_nodes = {node:values for node,values in self._nodes.items()
            if node[0]=='transfer' or node[1] in workers}
        self._background_nodes = {node:values for node,values in self._nodes.items()
            if node not in self._component_nodes}
        self._component_edges = tuple(edge for edge in self._edges if edge[2] in workers)
        self._component_edge_indices = tuple(index for index,edge in enumerate(self._edges) if edge[2] in workers)
        self._background_edges = tuple(edge for edge in self._edges if edge[2] not in workers)
        self._component_deliveries = {tid:node for tid,node in self._deliveries.items()
            if node in self._component_nodes}
        self._background_deliveries = {tid:node for tid,node in self._deliveries.items()
            if node not in self._component_nodes}
        self._background_objective = None
        return workers

    def _fixed_lp_objective(self, branches):


        times = {node:values[0] for node,values in self._background_nodes.items()}
        for (a,b,_),branch in zip(self._edges,branches):
            if a not in times:continue
            self.check_budget()
            if branch.coefficients or branch.conditions:
                raise ValueError('an independent worker has variable travel')
            times[b] = max(times[b],times[a]+self._nodes[a][1]+branch.constant)
        return sum((max(F(0),times[node]+self._nodes[node][1]
            -F(self.instance.task_map[tid].promised_delivery_time_s))
            for tid,node in self._background_deliveries.items()),F(0))


    def _build_lp(self, cells, branches):
        if set(cells) != set(self.plan.transfer_map):
            raise ValueError('LP cells must cover every transfer exactly')
        if len(branches) != len(self._edges):
            raise ValueError('LP travel branches must align with every route edge')
        variable_ids = {key for key, cell in cells.items() if cell.point is None}
        if any((set(branch.coefficients) | {key for row, _ in branch.conditions for key in row})
               - variable_ids for branch in branches):
            raise ValueError('LP branch references a missing or fixed coordinate')
        coordinates = {key: index for index, (key, cell) in enumerate(
            (pair for pair in cells.items() if pair[1].point is None))}


        nodes = getattr(self, '_component_nodes', self._nodes)
        deliveries = getattr(self, '_component_deliveries', self._deliveries)
        if hasattr(self,'_background_nodes') and self._background_objective is None:
            self._background_objective = self._fixed_lp_objective(branches)
        if hasattr(self,'objective_constant_s'):
            self.objective_constant_s = getattr(self,'_background_objective',None) or F(0)
        times = {node: len(coordinates) + k for k, node in enumerate(nodes)}
        tardiness = {tid: len(coordinates) + len(times) + k for k, tid in enumerate(deliveries)}
        size = len(coordinates) + len(times) + len(tardiness)
        objective = [0.] * size
        bounds = [(cells[key].low, cells[key].high) for key in coordinates]
        for node, (ready, duration, _) in nodes.items():
            high = None
            if node[0] == 'origin':
                high = ready
            elif node[0] == 'transfer':
                event = self.plan.transfer_map[node[1]]
                limits = [self.withdrawal_deadlines[w]
                          for w in (event.worker_i, event.worker_j) if self.withdrawal_deadlines.get(w) is not None]
                completion_limit = min(limits) if limits else None
                high = F(completion_limit) - duration if limits else None
            if high is not None and high < ready:


                public_completion = None
                if node[0] == 'transfer':
                    try:
                        public_completion = _advance_time(
                            float(ready), float(duration), 'transfer duration')
                    except RecoveryPlanValidationError:
                        pass
                if (node[0] == 'transfer' and (public_completion is None
                        or public_completion <= completion_limit)):
                    self._failure_reason = 'uncertified_time_bound_rounding'
                    warning = 'rational_time_bounds_without_public_clock_certificate'
                    if warning not in self.numerical_warnings:
                        self.numerical_warnings.append(warning)
                else:
                    self._failure_reason = 'exact_inconsistent_time_bounds'
                return None
            bounds.append((ready, high))
        bounds.extend((0., None) for _ in tardiness)


        origin = F(self.instance.time_s)
        characteristic = [abs(ready-origin) for ready, _, _ in self._nodes.values()]
        characteristic += [duration for _, duration, _ in self._nodes.values()]
        characteristic += [abs(F(t.promised_delivery_time_s)-origin) for t in self.instance.tasks]
        for branch in branches:
            characteristic.append(abs(branch.constant))
            characteristic.extend(abs(v)*max(abs(cells[k].low), abs(cells[k].high))
                                  for k, v in branch.coefficients.items())
        time_scale = max(characteristic, default=F(1)) or F(1)
        shifts = [cells[key].low for key in coordinates] + [origin]*len(times) + [F(0)]*len(tardiness)
        scales = [(cells[key].high-cells[key].low) or F(1) for key in coordinates] + [time_scale]*(len(times)+len(tardiness))
        bounds = [(None if lo is None else float((F(lo)-shift)/scale),
                   None if hi is None else float((F(hi)-shift)/scale))
                  for (lo, hi), shift, scale in zip(bounds, shifts, scales)]
        rows, cols, data, rhs = [], [], [], []
        def add(coeff, upper):
            index = len(rhs)
            upper = F(upper) - sum((value*shifts[key] for key, value in coeff.items()), F(0))
            transformed = {key: value*scales[key] for key, value in coeff.items() if value}
            row_scale = max((abs(value) for value in transformed.values()), default=F(1))
            for key, value in transformed.items():
                rows.append(index); cols.append(key); data.append(float(value/row_scale))
            rhs.append(float(upper/row_scale))
        for (a, b, _), branch in zip(self._edges, branches):
            if a not in times:
                if branch.coefficients or branch.conditions:
                    raise ValueError('an independent worker has variable travel')
                continue
            coeff = {times[a]: F(1), times[b]: -F(1)}
            coeff.update({coordinates[k]: v for k, v in branch.coefficients.items()})
            add(coeff, -branch.constant - self._nodes[a][1])
            for order, upper in branch.conditions:
                add({coordinates[k]: v for k, v in order.items()}, upper)
        for tid, node in deliveries.items():
            objective[tardiness[tid]] = 1.
            add({times[node]: F(1), tardiness[tid]: -F(1)},
                F(self.instance.task_map[tid].promised_delivery_time_s) - self._nodes[node][1])
        matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), size)).tocsr()
        self.stats.update(lp_time_nodes=len(nodes), lp_background_time_nodes=len(self._nodes)-len(nodes),
            lp_tardiness_variables=len(deliveries), lp_background_deliveries=len(self._deliveries)-len(deliveries))
        return dict(coordinates=coordinates, times=times, tardiness=tardiness,
                    objective=objective, bounds=bounds, matrix=matrix, rhs=rhs,
                    shifts=shifts, scales=scales, time_scale=time_scale,
                    canonical_order=tuple(sorted(coordinates)), canonical_index=-1,
                    completed=False, primary_value=None, result=None,
                    objective_constant_exact=getattr(self, '_background_objective', None))


    def _lp(self, cells, branches):


        if self._lp_state is None:
            self._lp_state = self._build_lp(cells, branches)
            resume = self._retry_states.pop(id(self._pending), None)
            if self._lp_state is not None and resume is not None:
                self._lp_state['canonical_index'] = resume['canonical_index']
                self._lp_state['primary_value'] = resume['primary_value']
                size = len(resume['coordinate_bounds'])
                self._lp_state['bounds'][:size] = resume['coordinate_bounds']
        state = self._lp_state
        if state is None:
            self._leaf_finished = True
            return None
        primary = state['canonical_index'] == -1
        objective = state['objective'] if primary else [0.]*len(state['objective'])
        equality, equality_rhs = None, None
        if not primary:
            key = state['canonical_order'][state['canonical_index']]
            objective[state['coordinates'][key]] = 1.
            equality = csr_matrix([state['objective']])
            equality_rhs = [state['primary_value']]
        options = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
        if not self.numerical_warnings:
            self.numerical_warnings.append('floating_lp_without_exact_optimality_or_infeasibility_certificate')
        self.check_budget()
        if getattr(self, 'work_budget', None) is not None:
            self.work_budget.charge('lp_calls')
            options['maxiter'] = self.work_budget.lp_maxiter
        self.stats['lp_calls'] += 1
        self.stats['primary_lp_calls' if primary else 'canonical_lp_calls'] += 1
        with self.measure('lp'):
            from solvers.tfr.work import single_thread_lp
            result = single_thread_lp(linprog, objective, A_ub=state['matrix'], b_ub=state['rhs'],
                             A_eq=equality, b_eq=equality_rhs, bounds=state['bounds'],
                             method='highs', options=options)
        self.stats['primary_lp_completed'] += int(primary)
        if getattr(self, 'work_budget', None) is not None:
            self.work_budget.used['lp_iterations'] += int(result.nit or 0)
        diagnostic = dict(time_scale=float(state['time_scale']), status=result.status,
                          phase='primary' if primary else 'canonical_coordinate',
                          coordinate=None if primary else key,
                          exact_optimality_certified=False, exact_infeasibility_certified=False)
        self.lp_diagnostics.append(diagnostic)
        if result.status == 1:
            if getattr(self, 'work_budget', None) is not None:
                from solvers.tfr.work import WorkLimitReached
                self.work_budget.exhausted = 'lp_iterations'
                raise WorkLimitReached('lp_iterations')
            raise TimeoutError('joint LP interrupted by native solver; branch retained')
        if result.status == 2:
            self._failure_reason = 'uncertified_lp_infeasibility'
            self.numerical_failures.append(dict(status=2, message='HiGHS infeasibility has no exact certificate; leaf remains unresolved'))
            self._leaf_finished = True
            return None
        if not result.success:
            self._failure_reason = 'lp_numerical_failure'
            self.numerical_failures.append(dict(status=result.status, message=result.message))
            self._leaf_finished = True
            return None
        if not all(math.isfinite(float(x)) for x in result.x):
            self._failure_reason = 'nonfinite_solution'
            self.numerical_failures.append(dict(status='nonfinite_solution', message='LP returned nonfinite variables'))
            self._leaf_finished = True
            return None
        residual = state['matrix'] @ result.x
        max_violation = max((max(0., float(a)-b) for a, b in zip(residual, state['rhs'])), default=0.)
        bound_violation = max((max(0., lo-x if lo is not None else 0., x-hi if hi is not None else 0.)
                               for x, (lo, hi) in zip(result.x, state['bounds'])), default=0.)
        objective_residual = 0. if primary else abs(sum(float(a)*float(x) for a, x in
            zip(state['objective'], result.x))-state['primary_value'])
        diagnostic.update(max_scaled_row_violation=max_violation,
                          max_scaled_bound_violation=bound_violation,
                          canonical_objective_face_residual=objective_residual)
        if max(max_violation, bound_violation, objective_residual) > 1e-8:
            self._failure_reason = 'scaled_residual'
            self.numerical_failures.append(dict(status='scaled_residual', message='LP residual exceeds diagnostic tolerance'))
            self._leaf_finished = True
            return None
        state['result'] = result
        if primary:
            state['primary_value'] = sum(float(a)*float(x) for a, x in zip(state['objective'], result.x))
            state['canonical_index'] = 0
        else:
            index = state['coordinates'][key]
            value = min(state['bounds'][index][1], max(state['bounds'][index][0], float(result.x[index])))
            state['bounds'][index] = (value, value)
            state['canonical_index'] += 1
        self._leaf_finished = state['canonical_index'] == len(state['canonical_order'])
        state['completed'] = self._leaf_finished

        if not primary and not self._leaf_finished:
            return _IN_PROGRESS
        coordinates, shifts, scales = state['coordinates'], state['shifts'], state['scales']
        locations = {}
        for key, cell in cells.items():
            if cell.point is not None:
                p = cell.point
                locations[key] = RoadPosition(network_id=p.network_id, node_id=p.node_id,
                    segment_id=p.segment_id, offset_m=None if p.offset_m is None else float(p.offset_m))
            else:
                offset = min(float(cell.high), max(float(cell.low), float(shifts[coordinates[key]] + scales[coordinates[key]]*F(float(result.x[coordinates[key]])))))
                locations[key] = self.instance.roads.segment_position(cell.segment, offset)
        return replace(self.plan, transfers=tuple(replace(e, location=locations[e.id]) for e in self.plan.transfers))


class GeometryArchive(JointGeometryLP):


    def __init__(self, instance, routing, plan, check_budget=lambda: None, *,
                 validate=None, measure=None, deadline_monotonic=None,
                 withdrawal_deadlines=None, activity_domains=None,
                 departure_times=None, exact_roads=None, execution_boundary=None, resources=None,
                 first_solution_priority=False, initialization_quantum=256,
                 decision_runtime=None, work_budget=None, fixed_meetings=None,
                 global_upper=None,pause_after_domains=False):
        if type(initialization_quantum) is not int or initialization_quantum < 1:
            raise ValueError('initialization_quantum must be a positive integer')
        self.initialization_quantum = initialization_quantum
        self.work_budget = work_budget
        if global_upper is not None and not callable(global_upper):
            raise ValueError('global upper bound must be a certified-incumbent callback')
        self.global_upper = global_upper
        self.pause_after_domains = pause_after_domains
        self.fixed_meetings=dict(fixed_meetings or {})
        if set(self.fixed_meetings)-set(plan.transfer_map):
            raise ValueError('unknown fixed background meeting')
        self.fixed_meetings={name:instance.roads.normalize_position(position)
            for name,position in self.fixed_meetings.items()}
        self._initialization_cursor = None
        self._expansion_cursor = None
        self._initial_domains = None
        self.initialization_phase = 'not_started'
        self.cooperative_progress = {}
        self.instance, self.routing, self.plan = instance, routing, plan
        self.first_solution_priority = first_solution_priority
        self.decisions = decision_runtime
        self._policy_node_turns = 0
        self.check_budget = check_budget
        self.resources = resources
        if resources is not None:
            resources.check_context(instance)
            if exact_roads is not None:
                raise ValueError('supply either exact_roads or per-solve geometry resources')
        attached = getattr(instance, 'execution_boundary', None)
        if execution_boundary is not None and execution_boundary is not attached:
            raise ValueError('execution boundary must be attached to its certified projection')
        self.execution_boundary = attached
        self.permission_anchors = {w.id: w.location for w in instance.workers}
        if attached is not None:
            attached.validate_for(instance)
            self.permission_anchors = dict(attached.permission_anchors)
        self.validate = validate or (lambda p: evaluate_recovery_plan(instance, p, routing))
        self.measure = measure or (lambda name: nullcontext())
        if deadline_monotonic is not None:
            raise ValueError("Algorithm wall-clock limits are not supported")
        self.deadline = None
        self.withdrawal_deadlines = withdrawal_deadlines or {}
        for wid, limit in self.withdrawal_deadlines.items():
            if wid not in instance.worker_map:
                raise ValueError(f'unknown cooperation deadline worker: {wid}')
            if limit is not None and (not math.isfinite(limit) or limit < instance.time_s):
                raise ValueError('cooperation deadline must be finite and no earlier than snapshot; explicit None means unlimited')
        domains = dict(instance.source_settings.activity_domains)
        for declared in ((attached.activity_domains if attached else {}), activity_domains or {}):
            for wid, domain in declared.items():
                if wid in domains and domains[wid] != domain:
                    raise ValueError('activity domain may not be reset at the execution boundary')
                domains[wid] = domain
        self.activity_domains = checked_activity_domains(instance, domains)
        missing_deadlines = {w.id for w in instance.workers if w.service_mode == 'handover_only'} - self.withdrawal_deadlines.keys()
        if missing_deadlines:
            raise ValueError(f'withdrawn workers require explicit cooperation deadlines: {sorted(missing_deadlines)}')
        self.departure_times = {w.id: instance.time_s for w in instance.workers}
        if attached is not None:
            self.departure_times = dict(attached.release_times)
            if departure_times is not None and dict(departure_times) != self.departure_times:
                raise ValueError('departure times disagree with certified execution boundary')
        elif departure_times and any(t != instance.time_s for t in departure_times.values()):
            raise ValueError('future departure times require an explicit certified execution boundary')
        else:
            self.departure_times.update(departure_times or {})
        self.roads = (resources.router(check_budget=check_budget) if resources is not None else
                      exact_roads or ExactRoads(instance.roads, check_budget))
        if self.roads.activity is not None:
            raise ValueError('the permission-range router must use the original full road network')
        self._activity_routers = {}
        self.best = None
        self.done = False
        self.numerical_failures = []
        self.numerical_warnings = []
        self.lp_diagnostics = []
        self.stats = dict(branches=0, lp_calls=0, pruned=0, infeasible=0, rejected=0,
            frontier_nodes_created=0, frontier_nodes_expanded=0, frontier_peak=0,
            selected_depth_peak=0, depth_first_pops=0,
            fixed_mode_minima_reused=0, fixed_mode_minima_computed=0, variable_mode_builds=0,
            bound_calls=0, road_splits=0, mode_splits=0, primary_lp_calls=0,
            canonical_lp_calls=0, candidates_validated=0, non_improving=0,
            unresolved_leaves=0, exact_objective_audits=0, exact_objective_interruptions=0,
            initialization_work_units=0, initialization_yields=0, initialization_restarts=0,
            initialization_wall_s=0., initialization_cpu_s=0., lp_wall_s=0., lp_cpu_s=0.,
            initialization_domains_completed=0, initialization_edges_completed=0,
            bound_work_units=0,bound_yields=0,primary_lp_completed=0,
            global_dominated=0,global_upper_prune_s=None,
            fixed_meeting_count=len(self.fixed_meetings),movable_meeting_count=len(plan.transfers)-len(self.fixed_meetings))
        self._pending = None
        self._pending_lower = None
        self._pending_plan = None
        self._pending_value = None
        self._pending_record = None
        self._best_exact = None
        self._lp_state = None
        self._leaf_finished = False
        self._failure_reason = None
        self._frontier = []
        self._initialized = False
        self._serial = 0
        self._events = tuple(sorted(plan.transfers, key=lambda e: e.id))
        self.cache_limits = cache_limits("h_archive")
        self._travel_cache = OrderedDict()
        self._fixed_minima = {}
        self._audit_travel_cache = OrderedDict()
        self.stats['cache_retention'] = {
            name: dict(hits=0, misses=0, evictions=0, bypasses=0, bypass_reasons={}, entries=0,
                       max_entries=self.cache_limits[limit])
            for name,limit in (('archive_travel','travel_entries'),('archive_audit','audit_entries'))}
        self._retry_states = {}
        self.unresolved = []
        self.candidate_objectives = []
        self._nodes, self._edges, self._deliveries = {}, [], {}
        self._compile()
        self._partition_components()

    def _partition_components(self):


        workers = self._partition_lp_components()
        rows = {wid:({},{},[]) for wid in sorted(set(self.plan.sequences)-workers)}
        for node,values in self._background_nodes.items():rows[node[1]][0][node] = values
        for tid,node in self._background_deliveries.items():rows[node[1]][1][tid] = node
        for edge in self._background_edges:rows[edge[2]][2].append(edge)
        self._background_workers = {wid:(nodes,tuple(edges),deliveries)
            for wid,(nodes,deliveries,edges) in rows.items()}
        self._background_objective = None
        if self.resources is not None:
            self.resources.stats.setdefault('independent_timing_computations',0)
            self.resources.stats.setdefault('independent_timing_hits',0)
        self.stats.update(background_constant_computations=0,
            component_workers=len(workers),background_workers=len(self._background_workers),
            component_edges=len(self._component_edges),background_edges=len(self._background_edges))

    def _roads_for(self, worker_id):

        domain = self.activity_domains.get(worker_id)
        if domain is None or domain.all_road_positions:
            return self.roads
        if domain not in self._activity_routers:
            self._activity_routers[domain] = (self.resources.router(domain, check_budget=self.check_budget) if self.resources is not None else
                ExactRoads(self.instance.roads, self.check_budget, activity_domain=domain))
        return self._activity_routers[domain]


    def _propagate(self, minima):
        return complete_steps(self._propagate_steps(minima))

    def _propagate_steps(self, minima):
        return (yield from self._propagate_graph_steps(
            self._nodes,self._edges,self._deliveries,minima))

    def _propagate_graph_steps(self, nodes, edges, deliveries, minima):
        times = {n: values[0] for n, values in nodes.items()}
        incoming = {n: [] for n in times}
        outgoing = {n: [] for n in times}
        for (a, b, _), value in zip(edges, minima):
            self.check_budget()
            yield
            if value is None:
                return None
            incoming[b].append(a)
            outgoing[a].append((b, nodes[a][1] + max(F(0), value)))
        degree = {n: len(v) for n, v in incoming.items()}
        ready = [n for n, d in degree.items() if not d]
        count = 0
        while ready:
            self.check_budget()
            yield
            node = ready.pop()
            count += 1
            for child, lag in outgoing[node]:
                yield
                times[child] = max(times[child], times[node] + lag)
                degree[child] -= 1
                if not degree[child]:
                    ready.append(child)
        if count != len(times):
            return None


        return sum((max(F(0), times[node] + nodes[node][1]
                        - F(self.instance.task_map[tid].promised_delivery_time_s))
                    for tid, node in deliveries.items()), F(0))

    def _lower_bound(self, cells, branches):

        return self._propagate([affine_cell_minimum(branch, cells) for branch in branches])

    def _node_bound(self, node):
        return complete_steps(self._node_bound_steps(node))

    def _node_bound_steps(self, node):
        self.stats['bound_calls'] += 1
        if any(not domain for domain in node.domains):
            return None
        if getattr(self, '_background_nodes', None) and self._background_objective is None:
            background_value = yield from self._background_bound_steps()
            if self.stats.get('background_partial_prune_pending'):


                return F(self.stats['background_prune_lower_exact'])
            self._background_objective = background_value
            self.stats['background_constant_computations'] += 1
            if self._background_objective is None:return None
        background = getattr(self, '_background_objective', None) or F(0)
        if self._can_prune(background):


            self.stats['background_prune_pending'] = True
            self.stats['background_prune_lower_exact'] = str(background)
            self.stats['background_prune_kind'] = 'complete_background'
            return background
        if node.modes is not None:
            cells = {event.id: domain[0] for event, domain in zip(self._events, node.domains)}
            minima = []
            edges = getattr(self, '_component_edges', self._edges)
            indices = getattr(self, '_component_edge_indices', range(len(self._edges)))
            for index in indices:
                modes = node.modes[index]
                values = []
                for mode in modes:
                    self.check_budget()
                    yield
                    value = affine_cell_minimum(mode, cells)
                    if value is not None:
                        values.append(value)
                minima.append(min(values) if values else None)
            value = yield from self._component_propagate_steps(edges,minima)
            return None if value is None else background+value
        domains = {event.id: domain for event, domain in zip(self._events, node.domains)}
        edges = getattr(self, '_component_edges', self._edges)
        minima = yield from self._edge_minima_steps(edges, domains)
        value = yield from self._component_propagate_steps(edges,minima)
        return None if value is None else background+value

    def _component_propagate_steps(self, edges, minima):
        if not hasattr(self,'_component_nodes'):
            return (yield from self._propagate_steps(minima))
        return (yield from self._propagate_graph_steps(self._component_nodes,edges,
            self._component_deliveries,minima))

    def _background_bound_steps(self):
        total = F(0)
        completed_workers = 0
        for wid in self._background_workers:
            self.check_budget()
            yield
            if self.resources is None:
                value = yield from self._background_worker_steps(wid)
            else:
                roads = self._roads_for(wid)
                key = ('independent_worker_timing',self.resources._router_key(roads),wid,
                    F(self.departure_times[wid]),tuple(self.plan.sequences[wid]))
                value = yield from self.resources.scalar_steps(key,
                    lambda: self._background_worker_steps(wid),'independent_timing_computations')
            if value is None:return None
            total += value
            completed_workers += 1
            if completed_workers < len(self._background_workers) and total and self._can_prune(total):


                self.stats.update(background_prune_pending=True,
                    background_partial_prune_pending=True,
                    background_prune_lower_exact=str(total),
                    background_prune_kind='partial_nonnegative_worker_sum',
                    background_partial_workers_completed=completed_workers,
                    background_partial_workers_skipped=len(self._background_workers)-completed_workers)
                return None
        return total

    def _background_worker_steps(self, wid):
        nodes,edges,deliveries = self._background_workers[wid]
        minima = []
        roads = self._roads_for(wid)
        speed = self.instance.worker_map[wid].profile.speed_factor
        for a,b,_ in edges:
            self.check_budget()
            yield
            origin,destination = nodes[a][2],nodes[b][2]
            value = (yield from cost_steps(roads,origin,destination,speed,progress=self.cooperative_progress)) if self.resources is None else (
                yield from self.resources.fixed_cost_steps(roads,origin,destination,speed,progress=self.cooperative_progress))
            minima.append(value)
            if not self._initialized:self.stats['initialization_edges_completed'] += 1
        return (yield from self._propagate_graph_steps(nodes,edges,deliveries,minima))

    def _edge_minima_steps(self, edges, domains):
        minima = []
        for a, b, wid in edges:
            self.check_budget()
            yield
            origin, destination = self._nodes[a][2], self._nodes[b][2]
            speed = self.instance.worker_map[wid].profile.speed_factor
            roads = self._roads_for(wid)
            if not isinstance(origin, str) and not isinstance(destination, str):
                key = roads, origin, destination, speed
                if key not in self._fixed_minima:
                    value = (yield from cost_steps(roads, origin, destination, speed, progress=self.cooperative_progress)) if self.resources is None else (
                        yield from self.resources.fixed_cost_steps(roads, origin, destination, speed, progress=self.cooperative_progress))
                    self._fixed_minima[key] = value
                minima.append(self._fixed_minima[key])
            else:
                left = domains[origin] if isinstance(origin, str) else (Cell(point=origin),)
                right = domains[destination] if isinstance(destination, str) else (Cell(point=destination),)
                key = roads, left, right, speed
                if key not in self._travel_cache:
                    self.stats['cache_retention']['archive_travel']['misses'] += 1
                    value = (yield from domain_travel_minimum_steps(roads, left, right, speed, progress=self.cooperative_progress)) if self.resources is None else (
                        yield from self.resources.domain_minimum_steps(roads, left, right, speed, progress=self.cooperative_progress))
                    self._travel_cache[key] = value
                    if len(self._travel_cache) > self.cache_limits["travel_entries"]:
                        self._travel_cache.popitem(last=False)
                        counter=self.stats['cache_retention']['archive_travel']
                        if self.cache_limits['travel_entries']:counter['evictions'] += 1
                        else:
                            from common.cache_telemetry import bypass
                            bypass(counter,'disabled')
                    self.stats['cache_retention']['archive_travel']['entries'] = len(self._travel_cache)
                else:
                    self.stats['cache_retention']['archive_travel']['hits'] += 1
                    self._travel_cache.move_to_end(key)
                    value = self._travel_cache[key]
                minima.append(value)
            if not self._initialized:
                self.stats['initialization_edges_completed'] += 1
        return minima

    def _fixed_travel_cost(self, roads, origin, destination, speed, *, retain=True):


        self.check_budget()
        if not retain:
            return (roads.cost(origin,destination,speed=speed) if self.resources is None else
                self.resources.fixed_cost(roads,origin,destination,speed))
        key = (roads, origin, destination, speed)
        if key not in self._fixed_minima:
            value = (roads.cost(origin, destination, speed=speed)
                if self.resources is None else
                self.resources.fixed_cost(roads, origin, destination, speed))

            self._fixed_minima[key] = value
        return self._fixed_minima[key]

    def _travel_modes(self, cells):
        modes = []
        for a, b, wid in self._edges:
            self.check_budget()
            origin, destination = self._nodes[a][2], self._nodes[b][2]
            speed = self.instance.worker_map[wid].profile.speed_factor
            roads = self._roads_for(wid)
            if not isinstance(origin, str) and not isinstance(destination, str):
                reused = (roads, origin, destination, speed) in self._fixed_minima
                retained = not hasattr(self,'_component_nodes') or a in self._component_nodes
                value = self._fixed_travel_cost(roads, origin, destination, speed,retain=retained)
                modes.append(() if value is None else (TravelBranch(value, {}),))
                self.stats['fixed_mode_minima_reused' if reused else 'fixed_mode_minima_computed'] += 1
            else:


                modes.append(tuple(travel_branches(roads, origin, destination, cells, speed)))
                self.stats['variable_mode_builds'] += 1
        return tuple(modes)

    def _push(self, node, lower):
        if lower is None:
            self.stats['pruned'] += 1
            self.stats['infeasible'] += 1
            return
        self._serial += 1
        heapq.heappush(self._frontier, (lower, -node.depth, node.label, self._serial, node))
        self.stats['frontier_nodes_created'] += 1
        self.stats['frontier_peak'] = max(self.stats['frontier_peak'], len(self._frontier))

    def _push_children(self, children):


        staged, serial, rejected = list(self._frontier), self._serial, 0
        for child, lower in children:
            if lower is None:
                rejected += 1
            else:
                serial += 1
                heapq.heappush(staged, (lower, -child.depth, child.label, serial, child))
        self._frontier = staged
        self.stats['frontier_nodes_created'] += serial-self._serial
        self._serial = serial
        self.stats['pruned'] += rejected
        self.stats['infeasible'] += rejected
        self.stats['frontier_peak'] = max(self.stats['frontier_peak'], len(staged))

    def _initialize(self):
        from solvers.tfr.geometry.bounds import NativeQueryPaused
        started, cpu = time.perf_counter(), time.process_time()
        if self._initialization_cursor is None:
            self._initialization_cursor = self._initialize_steps()
        used = 0
        try:
            domains_ready = self._initial_domains is not None
            for _ in range(self.initialization_quantum):
                if used >= self.initialization_quantum:break
                self.cooperative_progress['_domain_native_remaining'] = self.initialization_quantum-used
                before = sum(self.cooperative_progress.get(name,0) for name in ('domain_native_units','point_native_units'))
                advanced = False
                try:
                    try:self.check_budget()
                    except TimeoutError as error:raise NativeQueryPaused(str(error)) from error
                    next(self._initialization_cursor)
                    advanced = True
                finally:
                    charge = max(int(advanced),sum(self.cooperative_progress.get(name,0) for name in ('domain_native_units','point_native_units'))-before)
                    self.stats['initialization_work_units'] += charge;used += charge
                if self.pause_after_domains and not domains_ready and self._initial_domains is not None:
                    break
        except StopIteration:
            self._initialization_cursor = None
        except NativeQueryPaused:
            self.stats['native_query_pauses'] = self.stats.get('native_query_pauses',0)+1
            raise
        except BaseException:
            self._initialization_cursor.close()
            self._initialization_cursor = None
            self.stats['initialization_restarts'] += 1
            raise
        else:
            self.stats['initialization_yields'] += 1
        finally:
            self.cooperative_progress.pop('_domain_native_remaining',None)
            self.stats['initialization_wall_s'] += time.perf_counter()-started
            self.stats['initialization_cpu_s'] += time.process_time()-cpu

    def _lp(self, cells, branches):
        started, cpu = time.perf_counter(), time.process_time()
        try:
            with self.measure('lp'):
                return super()._lp(cells, branches)
        finally:
            self.stats['lp_wall_s'] += time.perf_counter()-started
            self.stats['lp_cpu_s'] += time.process_time()-cpu

    def _initialize_steps(self):
        self.initialization_phase = 'meeting_domains'
        domains = []
        for event in self._events:
            domain = yield from meeting_cells_steps(self.instance, event, self.roads,
                self.activity_domains, self.permission_anchors, self.resources, progress=self.cooperative_progress)
            if event.id in self.fixed_meetings:
                point=self.roads.normalize(self.fixed_meetings[event.id])
                legal=False
                for cell in domain:
                    if cell.point is not None and self.roads.normalize(cell.point)==point:
                        legal=True;break
                    if cell.segment is not None:
                        segment=self.instance.roads.segment_map[cell.segment]
                        offset=(point.offset_m if point.segment_id==cell.segment else F(0) if point.node_id==segment.u
                            else F(segment.length_m) if point.node_id==segment.v else None)
                        if offset is not None and cell.low<=offset<=cell.high:legal=True;break
                domain=(Cell(point=point),) if legal else ()
            domains.append(domain)
            self.stats['initialization_domains_completed'] += 1
        root = GeometryNode(tuple(domains))
        self._initial_domains = root.domains
        self.initialization_phase = 'root_bound'
        if self.pause_after_domains:yield
        lower = yield from self._node_bound_steps(root)
        self.check_budget()
        self._push(root, lower)
        self._initialized = True
        self.initialization_phase = 'completed'

    def _expand(self, node):
        from solvers.tfr.geometry.bounds import NativeQueryPaused
        if self._expansion_cursor is None:
            self._expansion_cursor = self._expand_steps(node)
        used = 0
        try:
            for _ in range(self.initialization_quantum):
                if used >= self.initialization_quantum:break
                self.cooperative_progress['_domain_native_remaining'] = self.initialization_quantum-used
                before = sum(self.cooperative_progress.get(name,0) for name in ('domain_native_units','point_native_units'))
                advanced = False
                try:
                    try:self.check_budget()
                    except TimeoutError as error:raise NativeQueryPaused(str(error)) from error
                    next(self._expansion_cursor)
                    advanced = True
                finally:
                    charge = max(int(advanced),sum(self.cooperative_progress.get(name,0) for name in ('domain_native_units','point_native_units'))-before)
                    self.stats['bound_work_units'] += charge;used += charge
        except StopIteration as done:
            self._expansion_cursor = None
            return done.value
        except NativeQueryPaused:
            self.stats['native_query_pauses'] = self.stats.get('native_query_pauses',0)+1
            raise
        except BaseException:
            self._expansion_cursor.close()
            self._expansion_cursor = None
            raise
        finally:
            self.cooperative_progress.pop('_domain_native_remaining',None)
        self.stats['bound_yields'] += 1
        return _IN_PROGRESS

    def _expand_steps(self, node):

        collections = node.domains if node.modes is None else node.modes
        for index, choices in enumerate(collections):
            if len(choices) > 1:
                middle = len(choices)//2
                children = []
                for side, subset in enumerate((choices[:middle], choices[middle:])):
                    updated = collections[:index] + (subset,) + collections[index+1:]
                    child = GeometryNode(updated if node.modes is None else node.domains,
                        None if node.modes is None else updated, node.depth+1, node.label+(side,))
                    lower = yield from self._node_bound_steps(child)
                    children.append((child, lower))

                self._push_children(children)
                self.stats['road_splits' if node.modes is None else 'mode_splits'] += 1
                return True
        if node.modes is None:
            cells = {event.id: domain[0] for event, domain in zip(self._events, node.domains)}
            modes = self._travel_modes(cells)
            child = GeometryNode(node.domains, modes, node.depth+1, node.label+(0,))
            lower = yield from self._node_bound_steps(child)
            self._push(child, lower)
            return True
        return False


    def _validate(self, plan):
        with self.measure('validation'):
            result = self.validate(plan)
            if set(result.tardiness_s) != set(self.instance.task_map):
                raise ValueError('validator did not score all remaining tasks')
            for wid, route in result.routes.items():
                for visit in route:
                    if visit.visit.operation == 'transfer' and self.withdrawal_deadlines.get(wid) is not None:
                        if visit.completion_time_s > self.withdrawal_deadlines[wid]:
                            raise RecoveryPlanValidationError('withdrawal_deadline', 'handover completes after cooperation deadline')
                    if visit.visit.operation == 'transfer' and wid in self.activity_domains:
                        if not self.activity_domains[wid].contains(visit.position, self.instance.roads):
                            raise RecoveryPlanValidationError('activity_domain', 'meeting outside activity restriction')
            return result

    def _can_prune(self, lower):
        upper = None if self.best is None else self.best.total_tardiness_s
        global_reference = False
        if self.global_upper is not None:
            candidate = self.global_upper()
            if candidate is not None:
                if (type(candidate)not in(int,float) or not math.isfinite(candidate) or candidate<0):
                    raise ValueError('global upper bound is not a finite certified tardiness')
                if upper is None or candidate<upper:
                    upper,global_reference = candidate,True
        if upper is None:return False
        cache = getattr(self, '_prune_certificate_cache', None)
        if (cache is None or cache['item'] is not self.instance or cache['plan'] is not self.plan
                or cache['routing'] is not self.routing or cache['routing_type'] is not type(self.routing)):


            cache = dict(item=self.instance, plan=self.plan, routing=self.routing,
                routing_type=type(self.routing), visits=sum(map(len,self.plan.sequences.values())),
                upper=None, error=None, computations=0, hits=0)
            self._prune_certificate_cache = cache
        if cache['upper'] != upper:
            cache['error'] = public_cost_underestimate_bound(
                SimpleNamespace(item=self.instance, routing=self.routing),cache['visits'],upper)
            cache['upper'] = upper
            cache['computations'] += 1
        else:
            cache['hits'] += 1

        error = cache['error']
        dominated = error is not None and lower >= F(upper) + error
        if dominated and global_reference:
            self.stats['global_dominated'] += 1
            self.stats['global_upper_prune_s'] = upper
        return dominated

    def _exact_objective(self, value):


        with self.measure('exact_objective_audit'):
            locations = {event.id: event.location for event in value.plan.transfers}
            minima = []
            component = hasattr(self, '_component_edges') and self._background_objective is not None
            edges = self._component_edges if component else self._edges
            for a, b, wid in edges:
                self.check_budget()
                origin, destination = self._nodes[a][2], self._nodes[b][2]
                fixed = not isinstance(origin, str) and not isinstance(destination, str)
                origin = locations[origin] if isinstance(origin, str) else origin
                destination = locations[destination] if isinstance(destination, str) else destination
                speed = self.instance.worker_map[wid].profile.speed_factor
                roads = self._roads_for(wid)
                key = (roads, origin, destination, speed)
                cache = self._fixed_minima if fixed else self._audit_travel_cache
                if key not in cache:
                    if not fixed:self.stats['cache_retention']['archive_audit']['misses'] += 1
                    value = roads.cost(origin, destination, speed=speed)
                    cache[key] = value
                    if not fixed and len(cache) > self.cache_limits["audit_entries"]:
                        cache.popitem(last=False)
                        counter=self.stats['cache_retention']['archive_audit']
                        if self.cache_limits['audit_entries']:counter['evictions'] += 1
                        else:
                            from common.cache_telemetry import bypass
                            bypass(counter,'disabled')
                    if not fixed:self.stats['cache_retention']['archive_audit']['entries'] = len(cache)
                else:
                    if not fixed:
                        self.stats['cache_retention']['archive_audit']['hits'] += 1
                        cache.move_to_end(key)
                    value = cache[key]
                minima.append(value)
            if component:
                result = complete_steps(self._propagate_graph_steps(self._component_nodes,
                    edges,self._component_deliveries,minima))
                if result is not None:result += self._background_objective
            else:result = self._propagate(minima)
            if result is None:
                raise ValueError('publicly validated candidate has no exact rational event schedule')
            self.stats['exact_objective_audits'] += 1
            return result

    def _finish_leaf(self):
        self.stats['branches'] += 1
        if self._failure_reason == 'exact_inconsistent_time_bounds':
            self.stats['infeasible'] += 1
        else:


            resume = None
            if self._lp_state is not None and not self._lp_state['completed']:
                state = self._lp_state
                resume = dict(canonical_index=state['canonical_index'],
                              primary_value=state['primary_value'],
                              coordinate_bounds=tuple(state['bounds'][:len(state['coordinates'])]))
            self.unresolved.append(dict(node=self._pending, lower=self._pending_lower,
                reason=self._failure_reason or 'uncertified_lp_optimality', resume=resume))
            self.stats['unresolved_leaves'] = len(self.unresolved)
        self._pending = self._pending_lower = self._pending_plan = self._lp_state = None
        self._pending_value = self._pending_record = None
        self._leaf_finished = False
        self._failure_reason = None

    def retry_unresolved(self):


        if self._pending is not None:
            raise ValueError('finish or resume the active geometry step before retrying')
        count = len(self.unresolved)
        for record in self.unresolved:
            self._push(record['node'], record['lower'])
            if record['resume'] is not None:
                self._retry_states[id(record['node'])] = record['resume']
        self.unresolved.clear()
        self.stats['unresolved_leaves'] = 0
        self.done = False
        return count

    def frontier_summary(self):
        lowers = [entry[0] for entry in self._frontier]
        lowers += [entry['lower'] for entry in self.unresolved]
        if self._pending_lower is not None:
            lowers.append(self._pending_lower)
        minimum = min(lowers) if lowers else None
        return dict(runnable_nodes=len(self._frontier), pending=self._pending is not None,
            initialized=self._initialized, initialization_phase=self.initialization_phase,
            initialization_in_progress=self._initialization_cursor is not None,
            child_bounds_in_progress=self._expansion_cursor is not None,
            cooperative_work=dict(self.cooperative_progress),
            unresolved_leaves=len(self.unresolved),
            unresolved_reasons={reason: sum(r['reason'] == reason for r in self.unresolved)
                                for reason in sorted({r['reason'] for r in self.unresolved})},
            minimum_unresolved_lower_exact=None if minimum is None else str(minimum),
            minimum_unresolved_lower_approx=None if minimum is None else float(minimum),
            mathematical_optimality_certified=False,
            global_dominated_nodes=self.stats['global_dominated'],
            global_upper_prune_s=self.stats['global_upper_prune_s'],
            best_exact_tardiness=None if self._best_exact is None else str(self._best_exact),
            exact_objective_pending=self._pending_value is not None,
            canonical_in_progress=self._lp_state is not None and not self._lp_state['completed'],
            candidate_objectives=list(self.candidate_objectives))


    def _pop_frontier(self):

        depth_first = self.first_solution_priority and self.best is None
        index = 0
        if depth_first:
            index = min(range(len(self._frontier)), key=lambda i: (
                -self._frontier[i][-1].depth, self._frontier[i][0],
                self._frontier[i][2], self._frontier[i][3]))
        heuristic_index = index
        if self.decisions is not None and self.decisions.strategies.geometry_nodes is not None:
            if self._policy_node_turns % DECISION_WINDOW == DECISION_WINDOW - 1:
                self.decisions.stats['geometry_nodes']['forced_turns'] += 1
            else:


                indices = [index]
                indices.extend(i for i in range(min(DECISION_WINDOW, len(self._frontier))) if i != index)
                indices = indices[:DECISION_WINDOW]
                summaries = []
                for i in indices:
                    lower, _, label, serial, node = self._frontier[i]
                    road_sizes = tuple(len(domain) for domain in node.domains)
                    mode_sizes = () if node.modes is None else tuple(len(modes) for modes in node.modes)
                    road_unfixed = sum(size > 1 for size in road_sizes)
                    mode_unfixed = sum(size > 1 for size in mode_sizes)
                    phase = ('road_domains' if road_unfixed else 'travel_mode_init') if node.modes is None else (
                        'travel_modes' if mode_unfixed else 'leaf')
                    sizes = road_sizes if node.modes is None else mode_sizes
                    summaries.append(GeometryNodeCandidate(lower.numerator, lower.denominator,
                        node.depth, label, serial, phase, road_unfixed, mode_unfixed,
                        sum(math.log(size) for size in sizes if size > 1), summarize_node(self, node, Cell) if self.decisions.strategies.geometry_semantics else None))
                decision = GeometryNodeDecision(tuple(summaries),
                    local_incumbent_tardiness=None if self.best is None else self.best.total_tardiness_s,
                    remaining_budget_s=None,
                    transfer_count=len(self._events), runnable_node_count=len(self._frontier),
                    primary_lp_completed=self.stats['primary_lp_completed'])
                index = indices[self.decisions.index('geometry_nodes', decision)]
            self._policy_node_turns += 1
        if index == 0 and not depth_first:
            entry = heapq.heappop(self._frontier)
        else:
            entry = self._frontier[index]
            self._frontier[index] = self._frontier[-1]
            self._frontier.pop()
            if index < len(self._frontier):
                heapq.heapify(self._frontier)
        self.stats['depth_first_pops'] += int(depth_first and index == heuristic_index)
        return entry

    def step(self):

        self.check_budget()
        if self.done:
            return None
        background = getattr(self, '_background_objective', None)
        if self.stats.get('background_prune_pending') or (background is not None and self._can_prune(background)):


            for cursor in (self._initialization_cursor, self._expansion_cursor):
                if cursor is not None:
                    cursor.close()
            self._initialization_cursor = self._expansion_cursor = None
            self._frontier.clear()
            self._pending = self._pending_lower = self._pending_plan = None
            self._pending_value = self._pending_record = self._lp_state = None
            self.stats['background_prune_pending'] = False
            if self.stats.get('background_partial_prune_pending'):
                self.stats['background_partial_early_prunes'] = self.stats.get('background_partial_early_prunes',0)+1
                self.stats['background_partial_prune_pending'] = False
            else:
                self.stats['background_prune_lower_exact'] = str(background)
                self.stats['background_prune_kind'] = 'complete_background'
            self.stats['background_early_prunes'] = self.stats.get('background_early_prunes', 0) + 1
            self.stats['pruned'] += 1
            self.initialization_phase = 'background_dominated'
            self.done = True
            return None
        if self.best is not None and self.best.total_tardiness_s == 0:
            self.done = True
            return None
        if not self.plan.transfers:
            try:
                value = self._validate(self.plan)
            except RecoveryPlanValidationError:
                self.stats['rejected'] += 1
                value = None
            if value is not None:
                self.stats['candidates_validated'] += 1
                self.candidate_objectives.append(dict(total_tardiness_s=value.total_tardiness_s,
                    phase='fixed_positions', improved=True))
            self.best, self.done = value, True
            return value
        with self.measure('geometry'):
            if not self._initialized:
                self._initialize()
                return None
            if self._pending is None:
                if not self._frontier:
                    self.done = True
                    return None
                if self.work_budget is not None:
                    self.work_budget.charge('geometry_nodes')
                entry = self._pop_frontier()
                lower, _, _, _, self._pending = entry
                self.stats['selected_depth_peak'] = max(
                    self.stats['selected_depth_peak'], self._pending.depth)
                self._pending_lower = lower
            node = self._pending
            if self._lp_state is None and self._pending_plan is None:
                if self._can_prune(self._pending_lower):
                    if self._expansion_cursor is not None:
                        self._expansion_cursor.close()
                        self._expansion_cursor = None
                    self.stats['pruned'] += 1
                    self._pending = self._pending_lower = None
                    return None
                expanded = self._expand(node)
                if expanded is _IN_PROGRESS:
                    return None
                if expanded:
                    self.stats['frontier_nodes_expanded'] += 1
                    self._pending = self._pending_lower = None
                    return None
            if self._pending_plan is None:
                cells = {event.id: domain[0] for event, domain in zip(self._events, node.domains)}
                branches = tuple(modes[0] for modes in node.modes)
                plan = self._lp(cells, branches)
                if plan is _IN_PROGRESS:
                    return None
                if plan is None:
                    self._finish_leaf()
                    return None
                self._pending_plan = plan


        self.check_budget()
        value = self._pending_value
        if value is None:
            try:
                value = self._validate(self._pending_plan)
                self._pending_value = value
            except RecoveryPlanValidationError as error:
                self.stats['rejected'] += 1
                self.numerical_failures.append(dict(status='validation_rejected', message=str(error)))
                self._failure_reason = 'validation_rejected'
        improved = False
        if value is not None:
            if self._pending_record is None:
                self.stats['candidates_validated'] += 1
                self._pending_record = dict(total_tardiness_s=value.total_tardiness_s,
                    phase='canonical' if self._leaf_finished else 'primary', improved=False,
                    exact_tardiness=None, exact_comparison_complete=False)
                self.candidate_objectives.append(self._pending_record)
            try:
                exact = self._exact_objective(value)
            except TimeoutError:


                self.stats['exact_objective_interruptions'] += 1
                improved = self.best is None or value.total_tardiness_s < self.best.total_tardiness_s
                self._pending_record.update(improved=improved, comparison='native_pending_exact_audit')
                if improved:
                    self.best, self._best_exact = value, None
                return value if improved else None
            if self.best is value and self._best_exact is None:


                self._best_exact = exact
            elif self.best is None:
                improved = True
            elif self._best_exact is not None:
                improved = exact < self._best_exact
            else:


                improved = value.total_tardiness_s < self.best.total_tardiness_s
            self._pending_record.update(exact_tardiness=str(exact), exact_comparison_complete=True,
                comparison='exact_same_archive', improved=improved or self._pending_record['improved'])
            self.stats['non_improving'] += int(not self._pending_record['improved'])
            if improved:
                self.best, self._best_exact = value, exact
            elif self._leaf_finished and exact == self._best_exact:


                self.best = value
        self._pending_plan = None
        self._pending_value = self._pending_record = None
        if self._leaf_finished:
            self._finish_leaf()
        return value if improved else None

    @property
    def exhausted_without_numerical_failure(self):
        return (self.done and not self.unresolved and not self._frontier
                and self._pending is None and not self.numerical_failures
                and not self.numerical_warnings and not self.stats['global_dominated'])
