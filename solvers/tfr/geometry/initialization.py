from fractions import Fraction as F
import heapq

from common.recovery.methods.roads_exact import ExactPosition, _SourceSearch


_NATIVE_POINT_CHUNK_UNITS = 64


def _point_query_state(state, target):


    return dict(phase='done' if target in state.settled else 'prepare',
        queue=state.queue,converted=[],conversion_index=0,astar=False,target_xy=None,
        current_node=None,current_cost=None,edge_index=0,native_units=0)


def _point_query_chunk(roads, graph, state, query, target, metric, reverse, progress):


    from solvers.tfr.geometry.bounds import NativeQueryPaused
    phase=query['phase'];queue=query['queue'];converted=query['converted']
    index=query['conversion_index'];astar=query['astar'];target_xy=query['target_xy']
    current=query['current_node'];cost=query['current_cost'];edge_index=query['edge_index']
    distances,settled=state.distances,state.settled;units=0
    limit=_NATIVE_POINT_CHUNK_UNITS
    if progress is not None and '_domain_native_remaining' in progress:


        limit=min(limit,max(0,progress['_domain_native_remaining']))
    try:
        while units<limit and phase!='done':
            if phase=='prepare':


                target_xy=_potential_target(roads,target,metric,reverse)
                astar=target_xy is not None
                if astar:
                    if progress is not None:progress['astar_queries']=progress.get('astar_queries',0)+1
                    phase='to_f';converted=[];index=0
                else:phase='search'
                continue
            if phase=='to_f':
                if index==len(queue):
                    heapq.heapify(converted);queue=converted;state.queue=queue
                    converted=[];index=0;phase='search'
                    continue
                roads.check_budget()
                old_cost,node=queue[index]
                potential=_potential_value(roads,node,target_xy,metric,reverse)
                converted.append((old_cost+potential,-old_cost,node))
                index+=1;units+=1
                continue
            if phase=='to_g':
                if index==len(queue):
                    heapq.heapify(converted);queue=converted;state.queue=queue
                    converted=[];index=0;phase='done'
                    continue
                roads.check_budget()
                _,negative_cost,node=queue[index]
                converted.append((-negative_cost,node));index+=1;units+=1
                continue
            if current is None:
                if not queue:
                    phase='to_g'if astar else'done'
                    continue
                roads.check_budget()
                if astar:
                    _,negative_cost,node=heapq.heappop(queue);cost=-negative_cost
                else:cost,node=heapq.heappop(queue)
                units+=1
                if node in settled or distances[node]!=cost:continue
                settled.add(node)
                if progress is not None:progress['settled_nodes']=progress.get('settled_nodes',0)+1
                current=node;edge_index=0
                continue
            edges=graph[current]
            if edge_index==len(edges):
                found=current==target
                current=None;cost=None;edge_index=0
                if found:phase='to_g'if astar else'done'
                continue
            roads.check_budget()
            neighbor,travel,distance=edges[edge_index]
            candidate=cost+(travel if metric=='time'else distance)
            if neighbor not in distances or candidate<distances[neighbor]:
                potential=_potential_value(roads,neighbor,target_xy,metric,reverse)if astar else 0
                distances[neighbor]=candidate
                heapq.heappush(queue,(candidate+potential,-candidate,neighbor)if astar else(candidate,neighbor))
            edge_index+=1;units+=1
            if progress is not None:progress['relaxed_arcs']=progress.get('relaxed_arcs',0)+1
    except TimeoutError as error:
        raise NativeQueryPaused(str(error))from error
    finally:
        query.update(phase=phase,queue=queue,converted=converted,conversion_index=index,
            astar=astar,target_xy=target_xy,current_node=current,current_cost=cost,
            edge_index=edge_index,native_units=query['native_units']+units)
        if progress is not None:
            progress['point_native_units']=progress.get('point_native_units',0)+units
            if '_domain_native_remaining'in progress:progress['_domain_native_remaining']-=units


def node_cost_steps(roads, origin, destination, metric='time', *, reverse=False, progress=None):
    roads.check_budget()
    if roads.activity is not None and (not roads.activity_contains(roads.roads.node_position(origin))
            or not roads.activity_contains(roads.roads.node_position(destination))):
        return None
    source, target = (destination, origin) if reverse else (origin, destination)
    graph = roads._reverse_graph if reverse else roads._graph
    key = source, metric, reverse
    state = roads._sources.pop(key, None)
    roads.source_stats['misses' if state is None else 'hits'] += 1
    if state is None:
        state = _SourceSearch({source: 0}, [(0, source)], set())
    query=_point_query_state(state,target)
    while query['phase']!='done':


        _point_query_chunk(roads,graph,state,query,target,metric,reverse,progress)
        if query['phase']!='done':yield
    result = F(state.distances[target], roads._scales[metric]) if target in state.settled else None
    if roads.source_cache_size:
        roads._sources[key] = state
        if len(roads._sources) > roads.source_cache_size:
            roads._sources.popitem(last=False)
            roads.source_stats['evictions'] += 1
    else:
        from common.cache_telemetry import bypass
        bypass(roads.source_stats, 'disabled')
    return result


def cost_steps(roads, origin, destination, speed=1, *, progress=None, metric='time'):
    scale = roads._settings(metric, speed)
    origin, destination = roads.normalize(origin), roads.normalize(destination)
    if not roads.activity_contains(origin) or not roads.activity_contains(destination):
        return None
    if origin == destination:
        return F(0)
    best = None
    if origin.segment_id is not None and origin.segment_id == destination.segment_id:
        delta = destination.offset_m-origin.offset_m
        for arc in roads._arcs[origin.segment_id]:
            roads.check_budget()
            yield
            if ((arc.forward and delta > 0) or (not arc.forward and delta < 0)) and roads.allowed_span(
                    origin.segment_id, origin.offset_m, destination.offset_m):
                value = abs(delta)*roads._unit(arc, metric)
                best = value if best is None else min(best, value)
    for start, first in roads._connectors(origin, leaving=True, metric=metric):
        for end, last in roads._connectors(destination, leaving=False, metric=metric):
            roads.check_budget()
            yield
            middle = yield from node_cost_steps(roads, start, end, metric=metric, progress=progress)
            if middle is not None:
                value = first+middle+last
                best = value if best is None else min(best, value)
    return None if best is None else best/scale


def merge_portions_steps(roads, portions):
    result = {}
    for sid in sorted(portions):
        merged = []
        for low, high in sorted(portions[sid]):
            roads.check_budget()
            yield
            if merged and low <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(merged[-1][1], high)
            else:
                merged.append((low, high))
        result[sid] = tuple(merged)
    return result


def reachable_parts_steps(roads, origin, radius, *, progress=None):

    origin, radius = roads.normalize(origin), F(radius)
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    if not roads.activity_contains(origin):
        return (), {}
    points, portions = {origin}, {}
    if radius == 0:
        return (origin,), {}
    def add(sid, low, high):
        points.add(roads.position(sid, low))
        points.add(roads.position(sid, high))
        if low < high:
            portions.setdefault(sid, []).append((low, high))
    if origin.segment_id is not None:
        offset = origin.offset_m
        component = roads.endpoint_interval(origin.segment_id, offset)
        for arc in roads._arcs[origin.segment_id]:
            roads.check_budget()
            yield
            low, high = ((offset, min(component[1], offset+radius)) if arc.forward
                         else (max(component[0], offset-radius), offset))
            add(origin.segment_id, low, high)
    distances, queue, settled = {}, [], set()
    for node, distance in roads._connectors(origin, leaving=True, metric='distance'):
        roads.check_budget()
        yield
        if distance <= radius and (node not in distances or distance < distances[node]):
            distances[node] = distance
            heapq.heappush(queue, (distance, node))
    while queue:
        roads.check_budget()
        yield
        distance, node = heapq.heappop(queue)
        if node in settled or distance != distances[node]:
            continue
        settled.add(node)
        if progress is not None:
            progress['rho_settled_nodes'] = progress.get('rho_settled_nodes', 0)+1
        points.add(ExactPosition(network_id=roads.roads.network_id, node_id=node))
        remaining = radius-distance
        for arc in roads._out_arcs.get(node, ()):
            roads.check_budget()
            yield
            if progress is not None:
                progress['rho_relaxed_arcs'] = progress.get('rho_relaxed_arcs', 0)+1
            segment = roads.roads.segment_map[arc.segment_id]
            length = roads._lengths[segment.id]
            component = roads.endpoint_interval(segment.id, F(0) if arc.forward else length)
            if component is None:
                continue
            reachable = component[1] if arc.forward else length-component[0]
            span = min(length, remaining, reachable)
            low, high = (F(0), span) if arc.forward else (length-span, length)
            add(segment.id, low, high)
            following = segment.v if arc.forward else segment.u
            candidate = distance+length
            if reachable == length and candidate <= radius and (
                    following not in distances or candidate < distances[following]):
                distances[following] = candidate
                heapq.heappush(queue, (candidate, following))
    intervals = yield from merge_portions_steps(roads, portions)
    return tuple(sorted(points, key=lambda p: (p.node_id or '', p.segment_id or '', p.offset_m or F(0)))), intervals


def domain_parts_steps(roads, domain):
    roads._validate_domain(domain)
    points, portions = {}, {}
    for nid in sorted(roads.roads.node_map if domain.all_road_positions else domain.node_ids):
        roads.check_budget()
        yield
        point = roads.roads.node_position(nid)
        points[(point.node_id, None, None)] = point
    if domain.all_road_positions:
        for segment in roads.roads.segments:
            roads.check_budget()
            yield
            portions[segment.id] = [(F(0), roads._lengths[segment.id])]
    else:
        for portion in domain.segment_portions:
            roads.check_budget()
            yield
            for offset in (portion.start_m, portion.end_m):
                point = roads.roads.segment_position(portion.segment_id, offset)
                points[(point.node_id, point.segment_id, point.offset_m)] = point
            if portion.start_m < portion.end_m:
                portions.setdefault(portion.segment_id, []).append((F(portion.start_m), F(portion.end_m)))
    intervals = yield from merge_portions_steps(roads, portions)
    return tuple(points.values()), intervals


def intersection_steps(roads, left, right):
    lp, li = left
    rp, ri = right
    lp, rp = set(lp), set(rp)
    def contains(point, points, intervals):
        if point in points:
            return True
        if point.segment_id is not None:
            for low, high in intervals.get(point.segment_id, ()):
                yield
                if low <= point.offset_m <= high:
                    return True
        return False
    points, intervals = set(), {}
    for point in lp | rp:
        roads.check_budget()
        yield
        if (yield from contains(point, lp, li)) and (yield from contains(point, rp, ri)):
            points.add(point)
    for sid in li.keys() & ri.keys():
        for a, b in li[sid]:
            for c, d in ri[sid]:
                roads.check_budget()
                yield
                low, high = max(a, c), min(b, d)
                if low <= high:
                    points.update((roads.position(sid, low), roads.position(sid, high)))
                    if low < high:
                        intervals.setdefault(sid, []).append((low, high))
    return tuple(points), intervals


def meeting_cells_steps(item, event, roads, activity_domains=None, permission_anchors=None,
                        resources=None, *, progress=None):
    from solvers.tfr.geometry.search import Cell
    if permission_anchors is None:
        original = getattr(item, 'original_input', item)
        permission_anchors = {w.id: w.location for w in original.workers}
    def expand(domain):
        parts = (yield from domain_parts_steps(roads, domain)) if resources is None else (
            yield from resources.domain_parts_steps(roads, domain))
        points = []
        for point in parts[0]:
            roads.check_budget()
            yield
            points.append(roads.normalize(point))
        return tuple(points), parts[1]
    senders = [(wid, sent) for wid, sent in ((event.worker_i, event.tasks_i_to_j),
                                           (event.worker_j, event.tasks_j_to_i)) if sent]


    universal = item.source_settings.meeting_domain.all_road_positions and all(
        domain.all_road_positions for wid, domain in (activity_domains or {}).items()
        if wid in (event.worker_i, event.worker_j))
    allowed = None if universal and senders else (yield from expand(item.source_settings.meeting_domain))
    for wid, sent in ((event.worker_i, event.tasks_i_to_j), (event.worker_j, event.tasks_j_to_i)):
        worker = item.worker_map[wid]
        if sent:
            ball = yield from reachable_parts_steps(roads,
                permission_anchors[wid], worker.handover_range_m,
                progress=progress)
            allowed = ball if allowed is None else (yield from intersection_steps(roads, allowed, ball))
        if not universal and activity_domains and wid in activity_domains:
            activity = yield from expand(activity_domains[wid])
            allowed = yield from intersection_steps(roads, allowed, activity)
    points, parts = allowed
    cells = []
    for sid in sorted(parts):
        for low, high in sorted(set(parts[sid])):
            roads.check_budget()
            yield
            cells.append(Cell(segment=sid, low=F(low), high=F(high)))
    for point in sorted(points, key=lambda p: (p.node_id or '', p.segment_id or '', p.offset_m or 0)):
        roads.check_budget()
        yield
        cells.append(Cell(point=point))
    return tuple(cells)


_POTENTIAL_DIRECTIONS = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
_POTENTIAL_COORDINATE_BITS = 256


def _prepare_potential(roads, metric, reverse=False):


    local = getattr(roads, '_geometry_potentials', None)
    if local is None:
        local = roads._geometry_potentials = {}
    key = metric, reverse
    if key in local:
        return local[key]
    provider = getattr(roads, '_geometry_resources', None)
    cache_key = None
    if provider is not None:
        from solvers.tfr.geometry.reuse import _MISS
        cache_key = 'exact_directed_potential', provider._router_key(roads), metric, reverse
        cached = provider.indexes.get(cache_key)
        if cached is not _MISS:
            local[key] = cached
            return cached
    unit = 1
    for node in roads.roads.nodes:
        roads.check_budget()
        unit = max(unit, node.x.as_integer_ratio()[1], node.y.as_integer_ratio()[1])


        if unit.bit_length() > _POTENTIAL_COORDINATE_BITS:
            local[key] = None
            return None
    coordinates = {}
    for node in roads.roads.nodes:
        roads.check_budget()
        xn, xd = node.x.as_integer_ratio(); yn, yd = node.y.as_integer_ratio()
        coordinates[node.id] = xn*(unit//xd), yn*(unit//yd)
    ratios = [None]*len(_POTENTIAL_DIRECTIONS)
    graph = roads._reverse_graph if reverse else roads._graph
    for node, edges in graph.items():
        roads.check_budget()
        x, y = coordinates[node]
        for neighbor, time, distance in edges:
            roads.check_budget()
            xx, yy = coordinates[neighbor]
            dx, dy = xx-x, yy-y
            weight = time if metric == 'time' else distance
            for index, (a, b) in enumerate(_POTENTIAL_DIRECTIONS):
                projection = a*dx+b*dy
                if projection > 0:
                    old = ratios[index]
                    if old is None or weight*old[1] < old[0]*projection:
                        ratios[index] = weight, projection
    result = coordinates, tuple((a,b,ratio[0],ratio[1])
        for (a,b),ratio in zip(_POTENTIAL_DIRECTIONS,ratios) if ratio is not None)
    if provider is not None:
        import sys
        size = (512+sys.getsizeof(coordinates)+sys.getsizeof(result[1])+
            sum(sys.getsizeof(pair)+sys.getsizeof(pair[0])+sys.getsizeof(pair[1])
                for pair in coordinates.values())+
            sum(sys.getsizeof(row)+sum(sys.getsizeof(x) for x in row) for row in result[1]))
        provider.indexes.put(cache_key, result, value_bytes=size)


        if cache_key not in provider.indexes.rows:
            result = None
    local[key] = result
    return result


def _potential_target(roads, target, metric, reverse):
    index = _prepare_potential(roads, metric, reverse)
    return None if index is None else index[0][target]


def _potential_value(roads, node, target_xy, metric, reverse):

    coordinates, ratios = _prepare_potential(roads, metric, reverse)
    x, y = coordinates[node]; dx, dy = target_xy[0]-x, target_xy[1]-y
    value = 0
    for a, b, numerator, denominator in ratios:
        projection = a*dx+b*dy
        if projection > 0:
            value = max(value, projection*numerator//denominator)
    return value
