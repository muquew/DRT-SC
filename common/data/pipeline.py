import csv
import glob
import json
from pathlib import Path
import random

from common.config import routing_options
from common.data.coordinates import CoordinateTransform, experimental_xy
from common.data.construction import ConstructionConfig
from common.data.local_density import choose_grid_district, choose_source_depot_district
from common.data.roads.schema import MovementProfile
from common.data.roads.routing import Routing
from common.data.roads.io import load_network
from common.initial_sequences.inputs import PlanningInput, PlanningTask, PlanningWorker


def _read(path):
    from common.initial_sequences.io import _invalid_constant, _unique_pairs
    return json.loads(Path(path).read_text(), parse_constant=_invalid_constant,
                      object_pairs_hook=_unique_pairs)


def _path(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _task(config, **values):
    return PlanningTask(**values, load=config.unit_load,
        pickup_service_time_s=config.pickup_service_time_s,
        delivery_service_time_s=config.delivery_service_time_s)


def _workers(config, values):
    return tuple(PlanningWorker(id=identifier, location=position,
        capacity=config.worker_capacity, available_time_s=config.worker_available_time_s,
        profile=MovementProfile(speed_factor=config.worker_speed_factor))
        for identifier, position in values)


def meituan(source, config, routing, root):

    if config.worker_origin not in ('source','observed_positions') or config.source_depot_id is not None:
        raise ValueError('Meituan workers use source rider observation positions')
    day = str(source['day'])
    with _path(root, source['riders']).open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['dt'] == day]
    if not rows:
        raise ValueError('no rider observations for ' + day)
    origin = min(int(r['dispatch_time']) for r in rows) if config.observation_time_s is None else config.observation_time_s
    observed = {r['courier_id']: r for r in rows if int(r['dispatch_time']) == origin}
    policy_values = dict(source['coordinate_policy'], max_distance_m=config.max_snap_distance_m)
    if policy_values.get('origin_lonlat') is not None:
        policy_values['origin_lonlat'] = tuple(policy_values['origin_lonlat'])
    policy = CoordinateTransform(**policy_values)
    def position(point):
        xy = experimental_xy(tuple(point), policy)
        crs = policy.source_crs if policy.kind == 'crs' else routing.roads.crs
        match = routing.match_point(xy, crs, config.max_snap_distance_m)
        return None if match is None else (match.position, routing.roads.coordinates(match.position))
    orders = {}
    with _path(root, source['orders']).open() as stream:
        for row in csv.DictReader(stream):
            if row['dt'] != day:
                continue
            try:
                for field in ('platform_order_time','dispatch_time','estimate_arrived_time',
                    'estimate_meal_prepare_time','waybill_id','sender_lng','sender_lat','recipient_lng','recipient_lat'):
                    int(row[field])
                if not row.get('order_id'):
                    continue
            except (KeyError, ValueError):
                continue
            old = orders.get(row['order_id'])
            if old is None or (int(row['dispatch_time']),int(row['waybill_id'])) > (int(old['dispatch_time']),int(old['waybill_id'])):
                orders[row['order_id']] = row
    candidates = {}
    for key, row in orders.items():
        pickup = position((int(row['sender_lng']),int(row['sender_lat'])))
        delivery = position((int(row['recipient_lng']),int(row['recipient_lat'])))
        if pickup is not None and delivery is not None:
            candidates[key] = row, pickup, delivery
    riders = {}
    for key, row in observed.items():
        matched = position((int(row['rider_lng']),int(row['rider_lat'])))
        if matched is not None:
            riders[key] = matched
    if config.sampling != 'full_city':
        if config.sampling != 'local_district':
            raise ValueError('Meituan supports full_city or local_district sampling')
        selected = choose_grid_district([(k,p[1],d[1]) for k,(_,p,d) in candidates.items()],
            [(k,v[1]) for k,v in riders.items()], radius_m=config.sampling_radius_m,
            required_tasks=config.task_count, required_workers=config.worker_count)
        if selected['status'] != 'selected':
            raise ValueError('insufficient source density for the requested district')
        candidates = {k:candidates[k] for k in selected['task_ids']}
        riders = {k:riders[k] for k in selected['worker_ids']}
    if len(candidates) < config.task_count or len(riders) < config.worker_count:
        raise ValueError('not enough road-matched source tasks or workers')
    keys = random.Random(config.task_seed).sample(sorted(candidates), config.task_count)
    tasks = []
    for key in keys:
        row, pickup, delivery = candidates[key]
        tasks.append(_task(config, id=day+':'+key, pickup=pickup[0], delivery=delivery[0],
            pickup_ready_time_s=float(max(int(row['platform_order_time']),int(row['estimate_meal_prepare_time']))-origin),
            delivery_ready_time_s=0. if config.customer_ready_time_s is None else config.customer_ready_time_s,
            promised_delivery_time_s=float(int(row['estimate_arrived_time'])-origin),
            release_time_s=float(int(row['platform_order_time'])-origin)))
    keys = random.Random(config.worker_seed).sample(sorted(riders), config.worker_count)
    return tuple(tasks), _workers(config, [(k,riders[k][0]) for k in keys]), float(origin)


def read_sartori(path):

    with Path(path).open() as stream:
        header = {}
        for line in stream:
            if line.strip() == 'NODES':
                break
            if ':' in line:
                key, value = line.split(':', 1)
                header[key] = value.strip()
        nodes = [stream.readline().split() for _ in range(int(header['SIZE']))]
    count = (len(nodes)-1)//2
    return header['NAME'], nodes[0], [(nodes[i],nodes[i+count]) for i in range(1,count+1)]


def sartori(source, config, routing, root):

    if config.worker_origin not in ('source','source_depot') or config.observation_time_s is not None:
        raise ValueError('Sartori workers use source depot positions')
    supplied = source['files']
    if isinstance(supplied, str):
        supplied = [supplied]
    paths = sorted({p for value in supplied for p in glob.glob(str(_path(root,value)))})
    if not paths:
        raise ValueError('no Sartori instance files matched the supplied paths')
    depots, requests, seen = {}, {}, set()
    for path in paths:
        name, depot, pairs = read_sartori(path)
        match = routing.match_point((float(depot[2]),float(depot[1])), 'EPSG:4326', config.max_snap_distance_m)
        if match is None:
            continue
        depots[name] = match.position
        for pickup, delivery in pairs:
            signature = tuple(pickup[1:7]), tuple(delivery[1:7])
            if signature in seen:
                continue
            a = routing.match_point((float(pickup[2]),float(pickup[1])), 'EPSG:4326', config.max_snap_distance_m)
            b = routing.match_point((float(delivery[2]),float(delivery[1])), 'EPSG:4326', config.max_snap_distance_m)
            if a is not None and b is not None:
                seen.add(signature)
                requests[name+':'+pickup[0]] = (a.position,b.position,pickup,delivery)
    if not depots:
        raise ValueError('no source depot matches the road network')
    depot_id = config.source_depot_id or sorted(depots)[0]
    if depot_id not in depots:
        raise ValueError('source_depot_id is not among the supplied source depots')
    depot = depots[depot_id]
    requests = {k:v for k,v in requests.items() if routing.reachable(v[0],v[1])}
    if config.sampling != 'full_city':
        coordinates = [(k,routing.roads.coordinates(v[0]),routing.roads.coordinates(v[1])) for k,v in requests.items()]
        if config.sampling == 'local_district':
            selected = choose_source_depot_district(coordinates,
                [(k,routing.roads.coordinates(v)) for k,v in depots.items()
                 if config.source_depot_id is None or k == depot_id],
                radius_m=config.sampling_radius_m, required_tasks=config.task_count)
            if selected['status'] == 'selected':
                depot = depots[selected['source_depot_id']]
        else:
            selected = choose_grid_district(coordinates, [], radius_m=config.sampling_radius_m,
                required_tasks=config.task_count, required_workers=0)
        if selected['status'] != 'selected':
            raise ValueError('insufficient source density for the requested district')
        requests = {k:requests[k] for k in selected['task_ids']}
    requests = {k:v for k,v in requests.items() if routing.reachable(depot,v[0])}
    if len(requests) < config.task_count:
        raise ValueError('not enough reachable source pickup-delivery pairs')
    keys = random.Random(config.task_seed).sample(sorted(requests), config.task_count)
    tasks = []
    for key in keys:
        a,b,pickup,delivery = requests[key]
        tasks.append(_task(config, id=key, pickup=a, delivery=b,
            pickup_ready_time_s=float(pickup[4])*60.,
            delivery_ready_time_s=float(delivery[4])*60. if config.customer_ready_time_s is None else config.customer_ready_time_s,
            promised_delivery_time_s=float(delivery[5])*60.))
    return tuple(tasks), _workers(config, [('depot_fleet_'+str(i).zfill(4),depot) for i in range(config.worker_count)]), 0.


def build(config_path):

    path = Path(config_path).resolve()
    config, root = _read(path), path.parent
    roads = load_network(_path(root, config['road_file']))
    routing = Routing(roads, **routing_options('formal'))
    construction = ConstructionConfig.from_dict(config['construction'])
    reader = {'meituan':meituan, 'sartori':sartori}[config['dataset']]
    tasks, workers, origin = reader(config['source'], construction, routing, root)
    planning = PlanningInput(instance_id=config.get('instance_id',config['dataset']+'-input'),
        planning_time_s=0., time_origin_s=origin, network_id=roads.network_id,
        snapshot_id=roads.snapshot_id, load_unit=construction.load_unit, tasks=tasks,
        workers=workers, required_task_ids=tuple(t.id for t in tasks))
    from common.initial_sequences.workflow import build_initial_background
    from common.scenarios.workflow import build_scenario
    background = build_initial_background(planning, routing,
        config.get('initial_method','greedydp'), **config['snapshot'])
    scenario = build_scenario(background.replay, background.snapshot,
        config.get('disruptions'), seed=config.get('event_seed',0))
    return background, scenario, roads


def load_case(config_path):

    path = Path(config_path).resolve()
    config = _read(path)
    if 'recovery_input_file' in config:
        from common.execution.io import load_execution_snapshot
        from common.recovery.io import load_recovery_input
        roads = load_network(_path(path.parent,config['road_file']))
        snapshot = load_execution_snapshot(_path(path.parent,config['snapshot_file']))
        item = load_recovery_input(_path(path.parent,config['recovery_input_file']),snapshot,roads)
    else:
        _, scenario, _ = build(path)
        item = scenario.item
    return item, Routing(item.roads, **routing_options('formal'))


def prepare(config_path, output_dir):
    from common.initial_sequences.workflow import save_initial_background
    from common.scenarios.workflow import save_scenario
    path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('preparation output must be empty')
    background, scenario, _ = build(path)
    save_initial_background(background, output/'initial')
    save_scenario(scenario, output/'scenario')
    from common.io import write_json
    from common.recovery.policy_initialization import history_from_replay
    write_json(output/'history.json',history_from_replay(background.replay,background.snapshot.time_s))
    record = dict(road_file=str(_path(path.parent,_read(path)['road_file']).resolve()),
        snapshot_file='initial/snapshot.json', recovery_input_file='scenario/recovery_input.json',
        history_file='history.json')
    (output/'case.json').write_text(json.dumps(record,indent=2)+'\n')
    return output/'case.json'
