from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

from common.recovery.plan import RecoveryPlan, RecoveryVisit
from common.initial_sequences.plan import _total


@dataclass(frozen=True)
class PathWitness:
    origin: object
    destination: object
    duration_s: float
    length_m: float


@dataclass(frozen=True)
class VisitWitness:
    visit: object
    incoming_path: PathWitness
    arrival_time_s: float
    start_time_s: float
    completion_time_s: float
    load_after: float
    carried_task_ids: tuple

    @property
    def position(self):
        return self.incoming_path.destination

    @property
    def waiting_time_s(self):
        return self.start_time_s - self.arrival_time_s


@dataclass
class CertifiedSolution:
    plan: object
    routes: dict
    tardiness_s: dict
    input_digest: str

    @property
    def total_tardiness_s(self):
        return _total(self.tardiness_s.values(),'certified.total_tardiness_s')

    @property
    def total_travel_time_s(self):
        return _total((r.incoming_path.duration_s for rows in self.routes.values() for r in rows),
                      'certified.total_travel_time_s')

    @property
    def total_distance_m(self):
        return _total((r.incoming_path.length_m for rows in self.routes.values() for r in rows),
                      'certified.total_distance_m')

    @property
    def transfer_count(self):
        return len(self.plan.transfers)

    def to_record(self):
        from common.initial_sequences.inputs import _plain
        return dict(plan=self.plan.to_dict(), tardiness_s=self.tardiness_s,
            input_digest=self.input_digest,
            routes={w: [_plain(row) for row in rows] for w, rows in self.routes.items()})

    @classmethod
    def from_record(cls, record):
        from common.data.roads.schema import RoadPosition
        routes = {}
        for worker, rows in record['routes'].items():
            route = []
            for original in rows:
                row = dict(original)
                path = dict(row.pop('incoming_path'))
                path['origin'] = RoadPosition(**path['origin'])
                path['destination'] = RoadPosition(**path['destination'])
                row['visit'] = RecoveryVisit(**row['visit'])
                row['carried_task_ids'] = tuple(row['carried_task_ids'])
                route.append(VisitWitness(incoming_path=PathWitness(**path), **row))
            routes[worker] = tuple(route)
        return cls(RecoveryPlan.from_dict(record['plan']), routes,
                   record['tardiness_s'], record['input_digest'])


def compact_solution(item, value):
    from solvers.tfr.interfaces.boundary import input_digest
    return CertifiedSolution(value.plan, {w: tuple(VisitWitness(row.visit,
        PathWitness(row.incoming_path.origin, row.incoming_path.destination,
                    row.incoming_path.duration_s, row.incoming_path.length_m),
        row.arrival_time_s, row.start_time_s, row.completion_time_s,
        row.load_after, row.carried_task_ids) for row in rows)
        for w, rows in value.routes.items()}, dict(value.tardiness_s), input_digest(item))


def validate_search_plan(item, plan, routing, *, cache=None, ablation='full'):

    from solvers.tfr.interfaces.boundary import PartialBoundaryRecoveryInput
    from common.evaluation.recovery_plan import evaluate_recovery_plan, _evaluate_recovery_plan
    if isinstance(item, PartialBoundaryRecoveryInput):
        if ablation!='no_transfer' or plan.transfers:
            raise ValueError('private subset search requires the strict no-transfer ablation')
        return _evaluate_recovery_plan(item,plan,routing,cache=cache,allow_partial_boundary=True)
    return evaluate_recovery_plan(item,plan,routing,cache=cache)


def retained_background(plan, source, selected):

    from solvers.tfr.structure import release_tasks,structure_key
    selected=set(selected)
    old=release_tasks(source,selected);new=release_tasks(plan,selected)
    if structure_key(old)!=structure_key(new):
        raise ValueError('candidate changed an unselected background arrangement')
    def names(value):
        result={}
        for worker,route in sorted(value.sequences.items()):
            for visit in route:
                if visit.operation=='transfer':result.setdefault(visit.transfer_id,len(result))
        return result
    old_names,new_names=names(old),names(new)
    previous={old_names[event.id]:event for event in old.transfers}
    return {event.id:previous[new_names[event.id]].location for event in new.transfers}


def _public_position(item,position):
    return (item.roads.node_position(position.node_id) if position.node_id is not None else
        item.roads.segment_position(position.segment_id,float(position.offset_m)))


def domain_seed(archive):

    from fractions import Fraction as F
    from shapely.geometry import LineString,Point
    from dataclasses import replace
    from solvers.tfr.geometry.search import Cell
    domains=(archive._initial_domains if archive._initial_domains is not None else
        archive._frontier[0][-1].domains if archive._frontier else None)
    if domains is None or any(not domain for domain in domains):return None
    positions={};cells={};lines={};item=archive.instance
    archive.stats['seed_domain_cells_scanned']=0
    archive.stats['initial_domain_cell_count']=sum(map(len,domains))
    for event,domain in zip(archive._events,domains):
        point=item.roads.coordinates(event.location)
        options=[]
        for cell in domain:
            archive.stats['seed_domain_cells_scanned']+=1
            if cell.point is not None:
                position=_public_position(item,cell.point)
            else:
                low=float(cell.low);high=float(cell.high)
                if F(low)<cell.low:low=math.nextafter(low,math.inf)
                if F(high)>cell.high:high=math.nextafter(high,-math.inf)
                if low>high:continue
                if cell.segment not in lines:lines[cell.segment]=LineString(item.roads.segment_map[cell.segment].geometry)
                offset=max(low,min(high,float(lines[cell.segment].project(Point(point)))))
                position=item.roads.segment_position(cell.segment,offset)
            distance=math.dist(point,item.roads.coordinates(position))
            options.append((distance,int(cell.point is not None),cell.segment or '',position.node_id or '',
                position.segment_id or '',position.offset_m or 0.,position,cell))
        if not options:return None
        *_,position,cell=min(options,key=lambda row:row[:6])
        positions[event.id]=position
        cells[event.id]=Cell(point=position) if cell.point is not None else cell
    plan=replace(archive.plan,transfers=tuple(replace(event,location=positions[event.id]) for event in archive.plan.transfers))
    return plan,cells


def seeded_candidate(archive,work,on_candidate=None):

    from solvers.tfr.work import WorkLimitReached
    from solvers.tfr.geometry.witness_geometry import build_witness_lp
    from common.evaluation.recovery_plan import RecoveryPlanValidationError
    archive.stats.update(domain_seed_attempts=0,domain_seed_validated=0,witness_seed_lp_attempts=0,
        witness_seed_lp_movable=0,witness_seed_lp_accepted=0,witness_seed_lp_improved=0)
    with archive.measure('seed_selection'):prepared=domain_seed(archive)
    if prepared is None:return None
    plan,cells=prepared
    work.charge('geometry_nodes');archive.stats['domain_seed_attempts']+=1
    try:value=archive._validate(plan)
    except RecoveryPlanValidationError:return None
    archive.stats['domain_seed_validated']+=1
    if on_candidate is not None:on_candidate(value)
    movable=set(cells)-set(archive.fixed_meetings)
    connector={event.id for event in value.plan.transfers if event.id in movable
        and event.location.node_id is not None and cells[event.id].point is None}
    model=None
    try:
        work.charge('geometry_nodes')
        try:
            model=build_witness_lp(archive.instance,value,movable,legal_cells=cells,
                connector_meetings=connector,withdrawal_deadlines=archive.withdrawal_deadlines,
                measure=archive.measure)
        except ValueError:


            model=build_witness_lp(archive.instance,value,(),
                withdrawal_deadlines=archive.withdrawal_deadlines,measure=archive.measure)
        model.work_budget=work
        archive.stats['witness_seed_lp_attempts']+=1
        archive.stats['witness_seed_lp_movable']=sum(cell.point is None for cell in model.cells.values())
        candidate=model.primary_candidate()
        if isinstance(candidate,RecoveryPlan):
            try:verified=archive._validate(candidate)
            except RecoveryPlanValidationError:verified=None
            if verified is not None and verified.total_tardiness_s<=value.total_tardiness_s:
                archive.stats['witness_seed_lp_improved']+=int(verified.total_tardiness_s<value.total_tardiness_s)
                value=verified;archive.stats['witness_seed_lp_accepted']+=1
                if on_candidate is not None:on_candidate(value)
    except (ValueError,WorkLimitReached):
        pass
    finally:
        if model is not None:
            for name in ('lp_calls','primary_lp_calls','canonical_lp_calls','primary_lp_completed'):
                archive.stats[name]+=model.stats[name]
    return value


def evaluate_structure(item,plan_record,*,geometry_limit=64,lp_limit=16,lp_maxiter=10_000,
                       source_plan_record=None,selected_tasks=None,routing_seed=None,
                       resources=None,evaluation_cache=None,cancel=lambda:None,first_feasible=False,
                       ablation='full'):

    from common.data.roads.routing import Routing
    from common.evaluation.recovery_plan import evaluate_recovery_plan, RecoveryPlanValidationError
    from solvers.tfr.structure import check_structure
    from solvers.tfr.geometry.search import GeometryArchive
    from solvers.tfr.work import StructureWork, WorkLimitReached
    from solvers.tfr.interfaces.boundary import input_digest
    started = time.perf_counter()
    plan = RecoveryPlan.from_dict(plan_record)
    if ablation not in ('full','no_transfer','sender_position','task_tardiness'):
        raise ValueError('unknown H ablation')
    if ablation=='no_transfer' and plan.transfers:
        return dict(status='structure_invalid',errors=['ablation_no_transfer'])
    if ablation=='sender_position':
        from solvers.tfr.ablation import fixed_sender_plan
        try:plan=fixed_sender_plan(item,plan)
        except ValueError as error:
            return dict(status='structure_invalid',errors=['fixed_sender_position'],error=str(error))
    seed = Routing(item.roads,cache_profile='formal',search_method='scipy') if routing_seed is None else routing_seed
    from common.evaluation.path_facts import capture_input_paths, fresh_path_cache
    input_paths=capture_input_paths(seed)
    seed.register_query_input(item,plans=(plan,))
    from common.evaluation._recovery_cache import RecoveryEvaluationCache
    from common.evaluation._recovery_cache import shared_cache
    from solvers.tfr.geometry.reuse import GeometryResources
    routing = seed if evaluation_cache is not None else seed.fork_query_session()
    cache = evaluation_cache or shared_cache(routing) or RecoveryEvaluationCache(routing,route_key_reuse=True)
    cache.check_routing(routing)
    if ablation in ('sender_position','no_transfer'):
        from solvers.tfr.interfaces.boundary import input_digest
        checked=check_structure(item,plan)
        if not checked.valid:
            return dict(status='structure_invalid',errors=list(checked.errors))
        work=StructureWork(geometry_limit,lp_limit,lp_maxiter)
        if source_plan_record is not None:
            source=RecoveryPlan.from_dict(source_plan_record)
            if (source.input_id,source.network_id,source.snapshot_id)!=(item.instance_id,item.network_id,item.snapshot_id):
                raise ValueError('background source belongs to another input')
            try:
                fixed=retained_background(plan,source,selected_tasks or ())
                if any(plan.transfer_map[name].location!=position for name,position in fixed.items()):
                    raise ValueError('fixed sender position changes retained background')
            except ValueError as error:
                return dict(status='structure_invalid',errors=['background_scope'],error=str(error))
        cancel()
        try:value=validate_search_plan(item,plan,routing,cache=cache,ablation=ablation)
        except RecoveryPlanValidationError as error:
            return dict(status='unresolved',error=str(error),work=work.used,
                elapsed_s=time.perf_counter()-started)
        cancel()
        validation_started=time.perf_counter()
        independent=fresh_path_cache(routing,input_paths)
        try:verified=validate_search_plan(item,value.plan,routing,cache=independent,ablation=ablation)
        except RecoveryPlanValidationError as error:
            return dict(status='validation_rejected',error=str(error),work=work.used,
                validation_attempts=1,validation_s=time.perf_counter()-validation_started,
                elapsed_s=time.perf_counter()-started)
        if verified.total_tardiness_s!=value.total_tardiness_s:
            raise ValueError('independent physical objective differs from candidate')
        record=compact_solution(item,verified).to_record()
        digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),
            allow_nan=False).encode()).hexdigest()
        return dict(status='validated_feasible',solution=record,witness_digest=digest,
            input_digest=input_digest(item),work=work.used,
            geometry=dict(fixed_meeting_count=len(plan.transfers),movable_meeting_count=0,
                geometry_nodes=0,lp_calls=0,lp_iterations=0),
            geometry_scope=('no-transfer physical schedule propagation' if ablation=='no_transfer'
                else 'fixed sender positions with physical schedule propagation'),
            geometry_search_complete=True,geometry_budget_stop=None,global_optimality_claim=False,
            validation_attempts=1,validation_s=time.perf_counter()-validation_started,
            validation_cache_policy='fresh_schedule_with_entry_input_paths',
            elapsed_s=time.perf_counter()-started)
    resources = resources or GeometryResources(item,cancel,scope_key=input_digest(item))
    resources.check_context(item)
    work = StructureWork(geometry_limit, lp_limit, lp_maxiter)
    fixed={}
    if source_plan_record is not None:
        source=RecoveryPlan.from_dict(source_plan_record)
        if (source.input_id,source.network_id,source.snapshot_id)!=(item.instance_id,item.network_id,item.snapshot_id):
            raise ValueError('background source belongs to another input')
        try:fixed=retained_background(plan,source,selected_tasks or ())
        except ValueError as error:
            return dict(status='structure_invalid',errors=['background_scope'],error=str(error),
                work=work.used,elapsed_s=time.perf_counter()-started)
    measured=dict(geometry_validation_attempts=0,geometry_validation_passes=0,
        geometry_validation_s=0.,geometry_phase_s={})
    @contextmanager
    def measure(name):
        beginning=time.perf_counter();passed=False
        if name=='validation':measured['geometry_validation_attempts']+=1
        try:
            yield
            passed=True
        finally:
            elapsed=time.perf_counter()-beginning
            stages=measured['geometry_phase_s'];stages[name]=stages.get(name,0.)+elapsed
            if name=='validation':
                measured['geometry_validation_s']+=elapsed
                measured['geometry_validation_passes']+=int(passed)
    check = check_structure(item, plan)
    if not check.valid:
        return dict(status='structure_invalid', errors=list(check.errors),
                    work=work.used,elapsed_s=time.perf_counter()-started,**measured)
    boundary = getattr(item, 'execution_boundary', None)
    archive = GeometryArchive(item,routing,plan,cancel,work_budget=work,measure=measure,fixed_meetings=fixed,
        validate=lambda candidate:evaluate_recovery_plan(item,candidate,routing,cache=cache),resources=resources,
        pause_after_domains=True,initialization_quantum=16384,
        withdrawal_deadlines=({w.id: None for w in item.workers if w.service_mode == 'handover_only'}
            if boundary is None else dict(boundary.execution_contract.get('withdrawal_deadlines', {}))),
        first_solution_priority=True)
    value, stopped = None, None
    def retain(candidate):
        nonlocal value
        if candidate is not None and (value is None or candidate.total_tardiness_s<value.total_tardiness_s):
            value=candidate


            if archive.best is None or candidate.total_tardiness_s<archive.best.total_tardiness_s:
                archive.best=candidate;archive._best_exact=None
    try:
        from dataclasses import replace
        seed_plan=replace(plan,transfers=tuple(replace(event,location=fixed.get(event.id,event.location))
            for event in plan.transfers))
        cancel()
        archive.stats['structure_seed_attempts']=1
        try:retain(archive._validate(seed_plan))
        except RecoveryPlanValidationError:pass
        if not(first_feasible and value is not None):
            if plan.transfers:
                while archive._initial_domains is None and not archive.done:
                    cancel();retain(archive.step())
                if archive._initial_domains is not None and not archive.done:
                    retain(seeded_candidate(archive,work,on_candidate=retain))
            while not archive.done and not(first_feasible and value is not None):
                cancel();retain(archive.step())
    except WorkLimitReached as error:
        stopped = str(error)
    if value is None:
        return dict(status='unresolved', work=work.used, exhausted=stopped,
                    geometry=dict(archive.stats),elapsed_s=time.perf_counter()-started,**measured)


    cancel()
    independent=fresh_path_cache(routing,input_paths)
    validation_started = time.perf_counter()
    try:
        verified = evaluate_recovery_plan(item, value.plan, routing, cache=independent)
    except RecoveryPlanValidationError as error:
        return dict(status='validation_rejected',error=str(error),work=work.used,validation_attempts=1,
                    validation_s=time.perf_counter()-validation_started,
                    elapsed_s=time.perf_counter()-started,**measured)
    validation_s=time.perf_counter()-validation_started
    if verified.total_tardiness_s != value.total_tardiness_s:
        raise ValueError('independent physical objective differs from candidate')
    record = compact_solution(item, verified).to_record()
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    return dict(status='validated_feasible', solution=record, witness_digest=digest,
        input_digest=input_digest(item),work=work.used,geometry=dict(archive.stats),
        geometry_scope=('first-certified initialization seed' if first_feasible else
            'budgeted full-domain road-segment and travel-mode search'),
        geometry_search_complete=archive.done,geometry_budget_stop=stopped,
        global_optimality_claim=False,
        validation_attempts=1,validation_s=validation_s,
        validation_cache_policy='fresh_schedule_with_entry_input_paths',
        elapsed_s=time.perf_counter()-started,**measured)
