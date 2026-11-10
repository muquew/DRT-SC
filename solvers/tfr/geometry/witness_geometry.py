from collections import deque
from contextlib import nullcontext
from fractions import Fraction as F
import math
from types import MappingProxyType

from common.recovery.inputs import checked_activity_domains
from solvers.tfr.geometry.search import Cell; from solvers.tfr.geometry.search import JointGeometryLP; from solvers.tfr.geometry.search import TravelBranch


def witnessed_path_branch(roads, path, origin_id, destination_id, cells):


    if (path.network_id != roads.network_id or path.snapshot_id != roads.snapshot_id):
        raise ValueError('witness path belongs to a different road snapshot')
    variable_origin = origin_id is not None and cells[origin_id].point is None
    variable_destination = destination_id is not None and cells[destination_id].point is None
    if not variable_origin and not variable_destination:
        return TravelBranch(F(path.duration_s), {})
    if not path.legs:
        raise ValueError('a stationary path has no witnessed moving endpoint mode')

    coefficients, anchors, conditions = {}, {}, []

    def endpoint(identifier, position, leg, incoming):
        cell = cells[identifier]
        arc = roads.arc_map[leg.arc_id]
        if (not arc.enabled or arc.segment_id != leg.segment_id
                or cell.segment != leg.segment_id or position.segment_id != cell.segment):
            raise ValueError('moving endpoint does not match its witnessed road segment')
        coordinate = F(position.offset_m)
        if not cell.low <= coordinate <= cell.high:
            raise ValueError('witness anchor lies outside its local cell')
        if not 0 <= cell.low <= cell.high <= F(roads.segment_map[cell.segment].length_m):
            raise ValueError('local cell lies outside its road segment')
        sign = F(1 if arc.forward else -1)
        if sign * (F(leg.end_offset_m) - F(leg.start_offset_m)) <= 0:
            raise ValueError('witness leg direction disagrees with the current arc')
        expected_offset = leg.end_offset_m if incoming else leg.start_offset_m
        if coordinate != F(expected_offset):
            raise ValueError('witness endpoint offset disagrees with its road leg')
        unit = (F(arc.travel_time_s) / F(roads.segment_map[cell.segment].length_m)
                / F(path.profile.speed_factor))
        coefficient = sign * unit * (1 if incoming else -1)
        coefficients[identifier] = coefficients.get(identifier, F(0)) + coefficient
        anchors[identifier] = coordinate
        return sign

    first, last = path.legs[0], path.legs[-1]
    origin_sign = endpoint(origin_id, path.origin, first, False) if variable_origin else None
    destination_sign = endpoint(destination_id, path.destination, last, True) if variable_destination else None
    if len(path.legs) == 1:
        sign = origin_sign if origin_sign is not None else destination_sign
        order, upper = {}, F(0)
        if variable_origin:
            order[origin_id] = sign
        else:
            upper -= sign * F(first.start_offset_m)
        if variable_destination:
            order[destination_id] = order.get(destination_id, F(0)) - sign
        else:
            upper += sign * F(last.end_offset_m)
        conditions.append(({key: value for key, value in order.items() if value}, upper))
    else:
        if variable_origin:
            conditions.append(({origin_id: origin_sign}, origin_sign * F(first.end_offset_m)))
        if variable_destination:
            conditions.append(({destination_id: -destination_sign}, -destination_sign * F(last.start_offset_m)))
    coefficients = {key: value for key, value in coefficients.items() if value}
    constant = F(path.duration_s) - sum((value * anchors[key] for key, value in coefficients.items()), F(0))
    return TravelBranch(constant, coefficients, tuple(conditions))


def _arc_index(roads, segments, check_budget=lambda: None):
    result = {segment: {} for segment in segments}
    if not result:
        return result
    for arc in roads.arcs:
        check_budget()
        if arc.enabled and arc.segment_id in result:
            sign = 1 if arc.forward else -1
            previous = result[arc.segment_id].get(sign)
            if previous is None or (arc.travel_time_s, arc.id) < (previous.travel_time_s, previous.id):
                result[arc.segment_id][sign] = arc
    return result


def _zero_path_branch(roads, path, origin_id, destination_id, cells, expanded, arc_index, sides, orders):
    variables = [identifier for identifier in (origin_id, destination_id)
                 if identifier is not None and cells[identifier].point is None]
    if not variables:
        return TravelBranch(F(0), {})
    if len(variables) == 2 and cells[origin_id].segment == cells[destination_id].segment:
        direction = orders.get(origin_id)
        if direction is not None and orders.get(destination_id) == direction:
            segment = roads.segment_map[cells[origin_id].segment]
            arc = arc_index[segment.id].get(direction)
            if arc is not None:
                unit = F(arc.travel_time_s) / F(segment.length_m) / F(path.profile.speed_factor)
                coefficient = {origin_id: -F(direction)*unit}
                coefficient[destination_id] = coefficient.get(destination_id, F(0)) + F(direction)*unit
                return TravelBranch(F(0), coefficient,
                    (({origin_id: F(direction), destination_id: -F(direction)}, F(0)),))


        equality = {origin_id: F(1)}
        equality[destination_id] = equality.get(destination_id, F(0)) - F(1)
        equality = {key: value for key, value in equality.items() if value}
        return TravelBranch(F(0), {}, ((equality, F(0)),
            ({key: -value for key, value in equality.items()}, F(0))))
    if len(variables) == 2:
        return _connector_branch(roads, path, origin_id, destination_id, cells, expanded, arc_index)
    identifier = variables[0]
    incoming = identifier == destination_id
    position = path.destination if incoming else path.origin
    cell = cells[identifier]
    segment = roads.segment_map[cell.segment]
    anchor = (F(position.offset_m) if position.segment_id == cell.segment else
              F(0) if position.node_id == segment.u else
              F(segment.length_m) if position.node_id == segment.v else None)
    if anchor is None:
        raise ValueError('zero-leg mode is not incident to its actual fixed anchor')
    if not cell.low <= anchor <= cell.high:
        raise ValueError('zero-leg anchor lies outside its local cell')
    side = sides.get(identifier, 1 if position.node_id != segment.v else -1)
    travel_direction = side if incoming else -side
    arc = arc_index[cell.segment].get(travel_direction)
    if arc is None:


        return TravelBranch(F(0), {}, (({identifier: F(1)}, anchor), ({identifier: F(-1)}, -anchor)))
    slope = F(side) * F(arc.travel_time_s) / F(segment.length_m) / F(path.profile.speed_factor)
    return TravelBranch(-slope*anchor, {identifier: slope},
                       (({identifier: F(-side)}, F(-side)*anchor),))


def _connector_branch(roads, path, origin_id, destination_id, cells, expanded, arc_index=None):


    variables = [(identifier, position, incoming) for identifier, position, incoming in
                 ((origin_id, path.origin, False), (destination_id, path.destination, True))
                 if identifier is not None and cells[identifier].point is None]
    if not any(identifier in expanded for identifier, _, _ in variables):
        return witnessed_path_branch(roads, path, origin_id, destination_id, cells)
    coefficients, anchors, endpoint_modes = {}, {}, {}
    for identifier, position, incoming in variables:
        cell = cells[identifier]
        segment = roads.segment_map[cell.segment]
        anchor = (F(position.offset_m) if position.segment_id == cell.segment else
                  F(0) if position.node_id == segment.u else
                  F(segment.length_m) if position.node_id == segment.v else None)
        if anchor is None or not cell.low <= anchor <= cell.high:
            raise ValueError('expanded segment is not incident to its witness anchor')
        leg = (path.legs[-1] if incoming else path.legs[0]) if path.legs else None
        original_mode = leg is not None and leg.segment_id == cell.segment
        if original_mode:
            arc = roads.arc_map[leg.arc_id]
            sign = F(1 if arc.forward else -1)
            fixed_offset = F(leg.start_offset_m if incoming else leg.end_offset_m)
        else:
            if identifier not in expanded or position.node_id is None:
                raise ValueError('no known connector for the moving endpoint')
            away_sign = 1 if position.node_id == segment.u else -1
            wanted = away_sign if incoming else -away_sign
            options = arc_index if arc_index is not None else _arc_index(roads, (cell.segment,))
            arc = options[cell.segment].get(wanted)
            if arc is None:
                raise ValueError('expanded endpoint has no enabled directed connector')
            sign, fixed_offset = F(wanted), anchor
        if not arc.enabled:
            raise ValueError('witness arc no longer enabled')
        unit = F(arc.travel_time_s) / F(segment.length_m) / F(path.profile.speed_factor)
        coefficient = sign * unit * (1 if incoming else -1)
        coefficients[identifier] = coefficients.get(identifier, F(0)) + coefficient
        anchors[identifier] = anchor
        endpoint_modes[incoming] = (identifier, sign, fixed_offset, original_mode)
    conditions = []
    if len(path.legs) == 1 and all(mode[3] for mode in endpoint_modes.values()):
        sign = next(iter(endpoint_modes.values()))[1]
        order, upper = {}, F(0)
        if False in endpoint_modes:
            order[origin_id] = sign
        else:
            upper -= sign * F(path.legs[0].start_offset_m)
        if True in endpoint_modes:
            order[destination_id] = order.get(destination_id, F(0)) - sign
        else:
            upper += sign * F(path.legs[-1].end_offset_m)
        conditions.append(({key: coefficient for key, coefficient in order.items() if coefficient}, upper))
    else:
        for incoming, (identifier, sign, fixed, _) in endpoint_modes.items():
            coefficient = -sign if incoming else sign
            conditions.append(({identifier: coefficient}, coefficient * fixed))
    coefficients = {key: coefficient for key, coefficient in coefficients.items() if coefficient}
    return TravelBranch(F(path.duration_s) - sum((coefficient * anchors[key] for key, coefficient in coefficients.items()), F(0)),
                        coefficients, tuple(conditions))


class WitnessLP(JointGeometryLP):


    def __init__(self, instance, validated_value, cells, *,
                 withdrawal_deadlines=None, check_budget=lambda: None,
                 measure=None, deadline_monotonic=None):
        self.instance, self.plan = instance, validated_value.plan
        self.check_budget = check_budget
        self.measure = measure or (lambda name: nullcontext())
        if deadline_monotonic is not None:
            raise ValueError("Algorithm wall-clock limits are not supported")
        self.deadline = None
        self.withdrawal_deadlines = dict(withdrawal_deadlines or {})
        for wid, limit in self.withdrawal_deadlines.items():
            if wid not in instance.worker_map:
                raise ValueError('unknown cooperation deadline worker')
            if limit is not None and (not math.isfinite(limit) or limit < instance.time_s):
                raise ValueError('cooperation deadline precedes snapshot or is nonfinite')
        missing = {worker.id for worker in instance.workers if worker.service_mode == 'handover_only'} - self.withdrawal_deadlines.keys()
        if missing:
            raise ValueError('withdrawn workers require explicit cooperation deadlines')
        boundary = getattr(instance, 'execution_boundary', None)
        if boundary is not None:
            boundary.validate_for(instance)
        self.departure_times = (dict(boundary.release_times) if boundary is not None
                                else {worker.id: instance.time_s for worker in instance.workers})
        self._nodes, self._edges, self._deliveries = {}, [], {}
        self._compile()
        self._partition_lp_components()
        self.cells = MappingProxyType(dict(cells))
        self.branches = ()
        self._lp_state = None
        self._retry_states = {}
        self._pending = None
        self._leaf_finished = False
        self._failure_reason = None
        self.stats = dict(lp_calls=0, primary_lp_calls=0, canonical_lp_calls=0,
                          primary_lp_completed=0)
        self.numerical_warnings, self.numerical_failures, self.lp_diagnostics = [], [], []
        self.affected_nodes = frozenset()
        self.affected_task_ids = frozenset()
        self.unaffected_tardiness_s = F(0)
        self.objective_constant_s = F(0)
        self.frozen_meeting_ids = frozenset()

    def primary_candidate(self):
        if getattr(self, 'supported', True) is False:
            raise ValueError('unsupported witness neighborhood')
        if self.stats['primary_lp_calls']:
            raise ValueError('primary LP already started; resume the same model explicitly')
        return self._lp(self.cells, self.branches)


def build_witness_lp(instance, validated_value, movable_meeting_ids, *,
                     legal_cells=None, withdrawal_deadlines=None,
                     check_budget=lambda: None, measure=None, deadline_monotonic=None,
                     connector_meetings=(), zero_sides=None, zero_orders=None, arc_index=None):


    plan = validated_value.plan
    if (plan.input_id != instance.instance_id or plan.network_id != instance.network_id
            or plan.snapshot_id != instance.snapshot_id):
        raise ValueError('witness plan belongs to a different recovery input')
    if set(plan.sequences) != set(instance.worker_map):
        raise ValueError('witness plan must cover every worker')
    requested = frozenset(movable_meeting_ids)
    if requested - plan.transfer_map.keys() or set(legal_cells or {}) - plan.transfer_map.keys():
        raise ValueError('unknown movable meeting or local cell')
    cells = {event.id: (legal_cells[event.id] if event.id in requested and event.id in (legal_cells or {})
                       else Cell(point=event.location)) for event in plan.transfers}
    arc_index = (_arc_index(instance.roads, {cell.segment for cell in cells.values() if cell.point is None}, check_budget)
                 if arc_index is None else arc_index)
    for identifier, cell in cells.items():
        if cell.point is not None and instance.roads.normalize_position(cell.point) != instance.roads.normalize_position(plan.transfer_map[identifier].location):
            raise ValueError('a fixed local cell must retain its witnessed location')
    model = WitnessLP(instance, validated_value, cells,
        withdrawal_deadlines=withdrawal_deadlines, check_budget=check_budget,
        measure=measure, deadline_monotonic=deadline_monotonic)
    paths, frozen = [], set()
    for wid, visits in sorted(plan.sequences.items()):
        rows = validated_value.routes[wid]
        if tuple(row.visit for row in rows) != visits:
            raise ValueError('witness routes disagree with the complete plan')
        for row in rows:
            check_budget()
            if row.incoming_path.profile != instance.worker_map[wid].profile:
                raise ValueError('witness path profile changed')
            paths.append(row.incoming_path)
    if len(paths) != len(model._edges):
        raise ValueError('witness routes do not align with the event graph')
    for (a, b, _), path in zip(model._edges, paths):
        check_budget()
        origin = model._nodes[a][2]
        destination = model._nodes[b][2]
        expected_origin = plan.transfer_map[origin].location if isinstance(origin, str) else origin
        expected_destination = plan.transfer_map[destination].location if isinstance(destination, str) else destination
        if (path.origin != instance.roads.normalize_position(expected_origin)
                or path.destination != instance.roads.normalize_position(expected_destination)):
            raise ValueError('witness path endpoints disagree with its aligned route edge')
        for identifier, position, incoming in ((origin, path.origin, False), (destination, path.destination, True)):
            if not isinstance(identifier, str) or cells[identifier].point is not None:
                continue
            cell = cells[identifier]
            leg = (path.legs[-1] if incoming else path.legs[0]) if path.legs else None
            supported_zero = not path.legs and position.segment_id == cell.segment
            if identifier not in connector_meetings and not supported_zero and (position.segment_id != cell.segment or leg is None or leg.segment_id != cell.segment):
                frozen.add(identifier)
    for identifier in frozen:
        cells[identifier] = Cell(point=plan.transfer_map[identifier].location)
    model.cells = MappingProxyType(cells)
    branches = []
    counts = dict(zero_variable_edges=0, zero_linked_edges=0, zero_directed_edges=0,
                  zero_direction_pinned_edges=0)
    successors = {node: [] for node in model._nodes}
    affected = set()
    for (a, b, _), path in zip(model._edges, paths):
        check_budget()
        origin, destination = model._nodes[a][2], model._nodes[b][2]
        origin_id = origin if isinstance(origin, str) else None
        destination_id = destination if isinstance(destination, str) else None
        if not path.legs:
            branch = _zero_path_branch(instance.roads, path, origin_id, destination_id, cells,
                                       connector_meetings, arc_index, zero_sides or {}, zero_orders or {})
            moving = [identifier for identifier in (origin_id, destination_id)
                      if identifier is not None and cells[identifier].point is None]
            counts['zero_variable_edges'] += bool(moving)
            counts['zero_linked_edges'] += len(moving) == 2 and not branch.coefficients
            counts['zero_directed_edges'] += len(moving) == 2 and bool(branch.coefficients)
            counts['zero_direction_pinned_edges'] += len(moving) == 1 and not branch.coefficients
        else:
            branch = _connector_branch(instance.roads, path, origin_id, destination_id, cells,
                                        connector_meetings, arc_index)
        branches.append(branch)
        successors[a].append(b)
        if branch.coefficients:
            affected.add(b)
    pending = deque(affected)
    while pending:
        check_budget()
        for following in successors[pending.popleft()]:
            if following not in affected:
                affected.add(following)
                pending.append(following)
    model.branches = tuple(branches)
    model.neighborhood_counts = counts
    model.affected_nodes = frozenset(affected)
    model.affected_task_ids = frozenset(tid for tid, node in model._deliveries.items() if node in affected)
    model.unaffected_tardiness_s = sum((F(value) for tid, value in validated_value.tardiness_s.items()
                                      if tid not in model.affected_task_ids), F(0))
    model.frozen_meeting_ids = frozenset(frozen | {key for key in requested if cells[key].point is not None})
    return model


def _containing_interval(domain, roads, segment_id, anchor):

    length = F(roads.segment_map[segment_id].length_m)
    if domain is None or domain.all_road_positions:
        return F(0), length
    intervals = sorted((F(part.start_m), F(part.end_m)) for part in domain.segment_portions
                       if part.segment_id == segment_id)
    merged = []
    for low, high in intervals:
        if merged and low <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(high, merged[-1][1])
        else:
            merged.append((low, high))
    return next(((low, high) for low, high in merged if low <= anchor <= high), None)


def coincident_meeting_groups(value, check_budget=lambda: None):

    parents = {event.id: event.id for event in value.plan.transfers}
    def root(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier
    for route in value.routes.values():
        previous = None
        for row in route:
            check_budget()
            current = row.visit.transfer_id if row.visit.operation == 'transfer' else None
            if previous is not None and current is not None and not row.incoming_path.legs:
                first, second = root(previous), root(current)
                parents[max(first, second)] = min(first, second)
            previous = current
    groups = {}
    for identifier in parents:
        groups.setdefault(root(identifier), set()).add(identifier)
    return {identifier: frozenset(groups[root(identifier)]) for identifier in parents}


def _anchor_check(model, value):

    offsets = {}
    for event in model.plan.transfers:
        cell = model.cells[event.id]
        if cell.point is not None:
            continue
        segment = model.instance.roads.segment_map[cell.segment]
        position = model.instance.roads.normalize_position(event.location)
        offsets[event.id] = (F(position.offset_m) if position.segment_id == cell.segment else
                             F(0) if position.node_id == segment.u else F(segment.length_m))
    paths = [row.incoming_path for wid in sorted(value.routes) for row in value.routes[wid]]
    duration_residual = F(0)
    for branch, path in zip(model.branches, paths):
        model.check_budget()
        duration = branch.constant + sum((coefficient * offsets[key]
            for key, coefficient in branch.coefficients.items()), F(0))
        duration_residual = max(duration_residual, abs(duration - F(path.duration_s)))
        if any(sum((coefficient * offsets[key] for key, coefficient in row.items()), F(0)) > upper
               for row, upper in branch.conditions):
            return dict(supported=False, reason='anchor_direction_condition', leg_duration_exact=str(duration_residual))
    if duration_residual:
        return dict(supported=False, reason='anchor_leg_duration', leg_duration_exact=str(duration_residual))
    state = model._build_lp(model.cells, model.branches)
    if state is None:
        return dict(supported=False, reason='anchor_time_bounds', leg_duration_exact='0')
    actual = [F(0)] * len(state['objective'])
    for identifier, index in state['coordinates'].items():
        actual[index] = offsets[identifier]
    for node, index in state['times'].items():
        if node[0] == 'origin':
            actual[index] = F(model.departure_times[node[1]])
        elif node[0] == 'transfer':
            actual[index] = F(value.transfer_start_times_s[node[1]])
        else:
            actual[index] = F(value.routes[node[1]][node[2]].start_time_s)
    for tid, index in state['tardiness'].items():
        actual[index] = F(value.tardiness_s[tid])
    vector = [float((number - shift) / scale) for number, shift, scale in
              zip(actual, state['shifts'], state['scales'])]
    row_residual = max((max(0., float(a) - b) for a, b in zip(state['matrix'] @ vector, state['rhs'])), default=0.)
    bound_residual = max((max(0., low-x if low is not None else 0., x-high if high is not None else 0.)
                         for x, (low, high) in zip(vector, state['bounds'])), default=0.)
    model._lp_state = state
    return dict(supported=max(row_residual, bound_residual) <= 1e-8,
                reason=None if max(row_residual, bound_residual) <= 1e-8 else 'anchor_schedule_residual',
                leg_duration_exact='0', max_scaled_row=row_residual, max_scaled_bound=bound_residual)
