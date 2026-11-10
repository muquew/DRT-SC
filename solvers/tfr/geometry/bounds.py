from fractions import Fraction as F
import heapq
from itertools import combinations
from math import lcm


_INTEGER_LABEL_MAX_BITS = 4096
_DOMAIN_ASTAR_TARGET_LIMIT = 4
_NATIVE_DOMAIN_CHUNK_UNITS = 64


class NativeQueryPaused(TimeoutError):
    pass


def _integer_domain_state(starts, best, astar):
    queue=[]if astar else[(cost,node)for node,cost in starts.items()]
    if not astar:heapq.heapify(queue)
    return dict(queue=queue,distances=dict(starts),settled=set(),
        initial=tuple(starts.items())if astar else(),init_index=0,initialized=not astar,
        current_node=None,current_cost=None,edge_index=0,best=best,done=False,astar=astar,native_units=0)


def _integer_domain_chunk(roads, graph, ends, state, targets, factor, reverse, progress):


    queue,distances,settled=state['queue'],state['distances'],state['settled']
    initial=state['initial'];init_index=state['init_index'];initialized=state['initialized']
    current=state['current_node'];current_cost=state['current_cost'];edge_index=state['edge_index']
    best,done=state['best'],state['done'];astar=state['astar'];units=0
    limit=_NATIVE_DOMAIN_CHUNK_UNITS
    if progress is not None and '_domain_native_remaining'in progress:
        limit=min(limit,max(0,progress['_domain_native_remaining']))
    try:
        while units<limit and not done:
            if not initialized:
                if init_index==len(initial):
                    initialized=True
                    continue
                roads.check_budget()
                node,cost=initial[init_index]
                potential=_few_target_potential(roads,node,targets,factor,reverse)
                heapq.heappush(queue,(cost+potential,-cost,node))
                init_index+=1;units+=1
                continue
            if current is None:
                roads.check_budget()
                if not queue:
                    done=True
                    break
                if astar:
                    lower,negative_cost,node=heapq.heappop(queue);cost=-negative_cost
                else:cost,node=heapq.heappop(queue);lower=cost
                units+=1
                if node in settled or cost!=distances[node]:continue
                if best is not None and lower>=best:
                    done=True
                    break
                settled.add(node)
                if progress is not None:progress['settled_nodes']=progress.get('settled_nodes',0)+1
                if node in ends:
                    candidate=cost+ends[node]
                    best=candidate if best is None else min(best,candidate)
                current=node;current_cost=cost;edge_index=0
                continue
            edges=graph[current]
            if edge_index==len(edges):
                current=current_cost=None;edge_index=0
                continue
            roads.check_budget()
            neighbor,weight,_=edges[edge_index]
            value=current_cost+weight*factor
            if(best is None or value<best)and(neighbor not in distances or value<distances[neighbor]):
                potential=_few_target_potential(roads,neighbor,targets,factor,reverse)if astar else 0
                distances[neighbor]=value
                heapq.heappush(queue,(value+potential,-value,neighbor)if astar else(value,neighbor))
            edge_index+=1;units+=1
            if progress is not None:progress['relaxed_arcs']=progress.get('relaxed_arcs',0)+1
    except TimeoutError as error:
        raise NativeQueryPaused(str(error))from error
    finally:
        state.update(init_index=init_index,initialized=initialized,current_node=current,current_cost=current_cost,
            edge_index=edge_index,best=best,done=done,native_units=state['native_units']+units)
        if initialized:state['initial']=()
        if progress is not None:
            progress['domain_native_units']=progress.get('domain_native_units',0)+units
            if '_domain_native_remaining'in progress:progress['_domain_native_remaining']-=units


def _few_target_potential(roads, node, targets, factor, reverse):
    from solvers.tfr.geometry.initialization import _potential_value


    return min(connector + factor*_potential_value(roads,node,xy,'time',reverse)
        for xy,connector in targets)


def complete_steps(iterator):

    while True:
        try:
            next(iterator)
        except StopIteration as done:
            return done.value


def _integer_labels_steps(roads, starts, ends, best):


    scale = unit = roads._scales['time']
    if unit.bit_length() > _INTEGER_LABEL_MAX_BITS:
        return None
    for values in (starts.values(), ends.values(), () if best is None else (best,)):
        for value in values:
            roads.check_budget()
            yield
            unit = lcm(unit, value.denominator)
            if unit.bit_length() > _INTEGER_LABEL_MAX_BITS:
                return None
    converted = []
    for values in (starts, ends):
        result = {}
        for node, value in values.items():
            roads.check_budget()
            yield
            result[node] = value.numerator * (unit // value.denominator)
        converted.append(result)
    return (*converted, None if best is None else best.numerator * (unit // best.denominator),
            unit // scale, unit)


def _integer_labels(roads, starts, ends, best):
    return complete_steps(_integer_labels_steps(roads, starts, ends, best))


def affine_cell_minimum(branch, cells):

    keys = sorted(set(branch.coefficients).union(*(set(c) for c, _ in branch.conditions)))
    inequalities = [(tuple(c.get(k, F(0)) for k in keys), F(rhs))
                    for c, rhs in branch.conditions]
    for i, key in enumerate(keys):
        unit = tuple(F(j == i) for j in range(len(keys)))
        inequalities += [(unit, cells[key].high), (tuple(-v for v in unit), -cells[key].low)]
    if not keys:
        return branch.constant if all(rhs >= 0 for _, rhs in inequalities) else None
    if len(keys) > 2:
        raise ValueError('a travel mode can involve at most two meeting coordinates')
    candidates = []
    if len(keys) == 1:
        candidates = [(rhs / row[0],) for row, rhs in inequalities if row[0]]
    else:
        for (a, ar), (b, br) in combinations(inequalities, 2):
            determinant = a[0]*b[1] - a[1]*b[0]
            if determinant:
                candidates.append(((ar*b[1]-a[1]*br)/determinant,
                                   (a[0]*br-ar*b[0])/determinant))
    values = [branch.constant + sum((branch.coefficients.get(k, 0)*x
                                    for k, x in zip(keys, point)), F(0))
              for point in candidates
              if all(sum((a*x for a, x in zip(row, point)), F(0)) <= rhs
                     for row, rhs in inequalities)]
    return min(values) if values else None


def _parts_steps(roads, cells):
    nodes, segments = set(), {}
    for cell in cells:
        roads.check_budget()
        yield
        if cell.point is not None:
            point = roads.normalize(cell.point)
            if not roads.activity_contains(point):
                continue
            if point.node_id is not None:
                nodes.add(point.node_id)
            else:
                segments.setdefault(point.segment_id, []).append((point.offset_m, point.offset_m))
        else:
            segment = roads.roads.segment_map[cell.segment]
            for low, high in roads.allowed_intervals(cell.segment):
                yield
                low, high = max(low, cell.low), min(high, cell.high)
                if low > high:
                    continue
                segments.setdefault(cell.segment, []).append((low, high))
                if low == 0:
                    nodes.add(segment.u)
                if high == roads._lengths[cell.segment]:
                    nodes.add(segment.v)
    return nodes, segments


def _parts(roads, cells):
    return complete_steps(_parts_steps(roads, cells))


def _connectors_steps(roads, parts, leaving):
    nodes, segments = parts
    result = {node: F(0) for node in nodes}
    for sid, intervals in segments.items():
        roads.check_budget()
        yield
        segment, length = roads.roads.segment_map[sid], roads._lengths[sid]
        for arc in roads._arcs[sid]:
            at_v = arc.forward == leaving
            component = roads.endpoint_interval(sid, length if at_v else F(0))
            if component is None:
                continue
            reachable = []
            for low, high in intervals:
                roads.check_budget()
                yield
                if max(low, component[0]) <= min(high, component[1]):
                    reachable.append((max(low, component[0]), min(high, component[1])))
            if not reachable:
                continue
            low, high = min(a for a, _ in reachable), max(b for _, b in reachable)
            if leaving:
                node, span = (segment.v, length-high) if arc.forward else (segment.u, low)
            else:
                node, span = (segment.u, low) if arc.forward else (segment.v, length-high)
            cost = span*roads._unit(arc, 'time')
            result[node] = min(result.get(node, cost), cost)
    return result


def _connectors(roads, parts, leaving):
    return complete_steps(_connectors_steps(roads, parts, leaving))


def domain_travel_minimum(roads, origins, destinations, speed, *, integer_labels=True,
                          routing_stats=None):
    return complete_steps(domain_travel_minimum_steps(roads, origins, destinations, speed,
        integer_labels=integer_labels, routing_stats=routing_stats))


def domain_travel_minimum_steps(roads, origins, destinations, speed, *, integer_labels=True,
                                routing_stats=None, progress=None):


    left = yield from _parts_steps(roads, origins)
    right = yield from _parts_steps(roads, destinations)
    if left[0] & right[0]:
        return F(0)
    best = None
    for sid in left[1].keys() & right[1].keys():
        roads.check_budget()
        yield


        for component_low, component_high in roads.allowed_intervals(sid):
            clipped = []
            for intervals in (left[1][sid], right[1][sid]):
                values = []
                for low, high in intervals:
                    roads.check_budget()
                    yield
                    if max(low, component_low) <= min(high, component_high):
                        values.append((max(low, component_low), min(high, component_high)))
                clipped.append(sorted(values))
            a, b = clipped


            i = j = 0
            while i < len(a) and j < len(b):
                roads.check_budget()
                yield
                if max(a[i][0], b[j][0]) <= min(a[i][1], b[j][1]):
                    return F(0)
                if a[i][1] < b[j][1]:
                    i += 1
                else:
                    j += 1
            for arc in roads._arcs[sid]:


                sources, targets = (a, b) if arc.forward else (b, a)
                j = 0
                for low, high in sources:
                    roads.check_budget()
                    yield
                    while j < len(targets) and targets[j][1] < low:
                        yield
                        j += 1
                    if j == len(targets):
                        break
                    gap = max(F(0), targets[j][0]-high)
                    cost = gap*roads._unit(arc, 'time')
                    best = cost if best is None else min(best, cost)
                    if best == 0:
                        return F(0)
    starts = yield from _connectors_steps(roads, left, True)
    ends = yield from _connectors_steps(roads, right, False)
    if not starts or not ends:
        return None if best is None else best/F(speed)
    factor = None
    unit = None
    if integer_labels:
        labels = yield from _integer_labels_steps(roads, starts, ends, best)
        if labels is not None:
            starts, ends, best, factor, unit = labels
        if routing_stats is not None:
            key = 'integer_queries' if labels is not None else 'fraction_fallback_queries'
            routing_stats[key] = routing_stats.get(key, 0) + 1
            if unit is not None:
                routing_stats['maximum_scale_bits'] = max(routing_stats.get('maximum_scale_bits', 0), unit.bit_length())
    graph=roads._graph
    astar=False;reverse=False;targets=()
    if factor is not None:
        from solvers.tfr.geometry.initialization import _potential_target
        reverse=len(starts)<len(ends)
        goals=starts if reverse else ends
        if len(goals)<=_DOMAIN_ASTAR_TARGET_LIMIT:
            prepared=[]
            for node,connector in goals.items():
                roads.check_budget()
                yield
                xy=_potential_target(roads,node,'time',reverse)
                if xy is None:
                    prepared=[]
                    break
                prepared.append((xy,connector))
            if prepared:
                astar=True;targets=tuple(prepared)
                if reverse:


                    starts,ends=ends,starts
                    graph=roads._reverse_graph
    if routing_stats is not None:
        key='astar_queries'if astar else'dijkstra_queries'
        routing_stats[key]=routing_stats.get(key,0)+1
        if astar and reverse:routing_stats['astar_reverse_queries']=routing_stats.get('astar_reverse_queries',0)+1
    if factor is not None:
        native_state=_integer_domain_state(starts,best,astar)
        while not native_state['done']:
            _integer_domain_chunk(roads,graph,ends,native_state,targets,factor,reverse,progress)
            if not native_state['done']:
                yield
        best=native_state['best'];settled=native_state['settled']
    else:
        distances = dict(starts)
        if astar:
            queue=[]
            for node,cost in starts.items():
                roads.check_budget()
                yield
                queue.append((cost+_few_target_potential(roads,node,targets,factor,reverse),-cost,node))
        else:queue = [(cost, node) for node, cost in starts.items()]
        heapq.heapify(queue)
        settled = set()
        scale = roads._scales['time']
        while queue:
            roads.check_budget()
            yield
            if astar:
                lower,negative_cost,node=heapq.heappop(queue)
                cost=-negative_cost
            else:
                cost,node=heapq.heappop(queue)
                lower=cost
            if node in settled or cost != distances[node]:
                continue
            if best is not None and lower >= best:
                break
            settled.add(node)
            if progress is not None:
                progress['settled_nodes'] = progress.get('settled_nodes', 0)+1
            if node in ends:
                value = cost + ends[node]
                best = value if best is None else min(best, value)
            for neighbor, weight, _ in graph[node]:
                roads.check_budget()
                yield
                if progress is not None:
                    progress['relaxed_arcs'] = progress.get('relaxed_arcs', 0)+1
                value = cost + (weight * factor if factor is not None else F(weight, scale))
                if (best is None or value < best) and (neighbor not in distances or value < distances[neighbor]):
                    distances[neighbor] = value
                    if astar:
                        heapq.heappush(queue,(value+_few_target_potential(roads,neighbor,targets,factor,reverse),-value,neighbor))
                    else:heapq.heappush(queue, (value, neighbor))
    if routing_stats is not None:
        routing_stats['completed_queries'] = routing_stats.get('completed_queries', 0) + 1
        routing_stats['settled_nodes'] = routing_stats.get('settled_nodes', 0) + len(settled)
    return None if best is None else (F(best, unit) if unit is not None else best)/F(speed)
