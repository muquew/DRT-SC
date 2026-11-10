from dataclasses import dataclass
from math import isfinite
from common.recovery.inputs import checked_activity_domains

from solvers.tfr.interfaces.boundary import ActionBlock; from solvers.tfr.interfaces.boundary import HoldingGuarantee; from solvers.tfr.interfaces.boundary import BoundaryUncertified; from solvers.tfr.interfaces.boundary import build_boundary; from solvers.tfr.interfaces.boundary import project_input


@dataclass(frozen=True, kw_only=True)
class ReplayBoundaryPolicy:
    all_remaining_service_confirmed: bool
    node_waiting_permitted: bool
    service_point_waiting_permitted: bool
    gate_duration_s: float
    holding_duration_s: float | None = None
    deterministic_current_arc_continuation: bool
    model_evidence: str

    def __post_init__(self):
        for field in ('all_remaining_service_confirmed','node_waiting_permitted',
                      'service_point_waiting_permitted','deterministic_current_arc_continuation'):
            if type(getattr(self,field)) is not bool:
                raise BoundaryUncertified(f'explicit boolean replay policy required: {field}')
        if not self.model_evidence:
            raise BoundaryUncertified('explicit replay-model evidence identifier required')
        if not isfinite(self.gate_duration_s) or self.gate_duration_s<0:
            raise BoundaryUncertified('finite nonnegative gate duration required')
        if self.holding_duration_s is not None:
            raise BoundaryUncertified('legacy finite holding duration is unsupported; omit it or use null for continuing custody')


def replay_action_contract(item,routing,window,policy,*,check_budget=lambda:None,withdrawal_deadlines=None,
                           _materialize_full_suffix=False):


    if not isinstance(policy,ReplayBoundaryPolicy):
        raise BoundaryUncertified('ReplayBoundaryPolicy must be explicitly declared')
    blocks=[];orders={};guarantees={}
    network=item.roads
    domains=checked_activity_domains(item)
    routing=routing.for_activity_domain(None)
    for worker in item.workers:
        check_budget()
        wid=worker.id;state=item.source_snapshot.worker_map[wid]
        worker_routing=routing.for_activity_domain(domains.get(wid))
        if state.fixed_prefix_length:
            raise BoundaryUncertified('service-only fixed prefix needs explicit action-block execution contract')
        if state.available_time_s>window.tau:
            raise BoundaryUncertified('future original worker availability needs explicit action blocks')
        position=worker.location;order=[];legal={}
        clock=window.tau
        suffix_only=False
        def allow(position,service=False):


            if (item.source_settings.meeting_domain.contains(position,network) or
                    (position.node_id is not None and policy.node_waiting_permitted) or
                    (service and policy.service_point_waiting_permitted)):
                legal[position]=HoldingGuarantee(position,None,
                                                 policy.gate_duration_s,policy.model_evidence)


        waiting_service=False
        if state.phase in ('waiting','servicing') and state.remaining_visits:
            visit=state.remaining_visits[0]
            original_task=item.source_snapshot.task_map[visit.task_id].task
            original_position=original_task.pickup if visit.operation=='pickup' else original_task.delivery
            waiting_service=position==original_position
        allow(position,service=waiting_service)
        serial=0
        def travel_leg(leg,origin,*,started=False):
            nonlocal serial,position,clock,suffix_only
            check_budget()
            if not _materialize_full_suffix and not started and (suffix_only or clock>=window.activation_s):
                suffix_only=True
                return
            arc=network.arc_map.get(leg.arc_id)
            if arc is None or not arc.enabled:
                raise BoundaryUncertified('ongoing/old road arc unavailable; safe continuation facts missing')
            segment=network.segment_map[arc.segment_id]
            if origin.segment_id==segment.id:
                start=origin.offset_m
            elif origin.node_id==segment.u:
                start=0.
            elif origin.node_id==segment.v:
                start=segment.length_m
            else:
                raise BoundaryUncertified('old road leg not adjacent to observed position')
            end=leg.end_offset_m
            if (arc.forward and end<start) or (not arc.forward and end>start):
                raise BoundaryUncertified('old road leg reverses its original directed arc')
            if not worker_routing.activity_domain.contains_span(segment.id,start,end):
                raise BoundaryUncertified('ongoing/old road leg exits declared activity domain')
            destination=network.segment_position(segment.id,end)

            duration=routing._part(arc,start,end,worker.profile).duration_s if end!=start else 0.
            if duration<=0:
                if destination!=origin:
                    raise BoundaryUncertified('nonpositive movement duration')
                position=destination
                return
            serial+=1;bid=f'{wid}:travel:{serial}'
            confirmed=started or policy.all_remaining_service_confirmed
            blocks.append(ActionBlock(bid,'travel',(wid,),(origin,),destination,duration,
                policy.model_evidence,confirmed=confirmed,safe_to_complete=confirmed,
                started_at_s=window.tau if started else None))
            order.append(bid);position=destination;allow(position)
            if confirmed:
                clock+=duration
            else:
                suffix_only=True
        for index,visit in enumerate(item.reference_sequences[wid]):
            check_budget()
            task=item.task_map[visit.task_id]
            destination=network.normalize_position(task.pickup if visit.operation=='pickup' else task.delivery)
            if index==0 and state.phase=='servicing':
                service=state.service
                if service is None or service.visit != visit or position != destination:
                    raise BoundaryUncertified('active service lacks its original physical progress')
                serial+=1;bid=f'{wid}:{visit.operation}:{serial}'
                blocks.append(ActionBlock(bid,visit.operation,(wid,),(position,),destination,
                    service.remaining_time_s,policy.model_evidence,task_id=visit.task_id,
                    ready_at_s=window.tau,confirmed=True,safe_to_complete=True,
                    irrevocable=True,started_at_s=service.start_time_s))
                order.append(bid);clock=service.completion_time_s;allow(destination,service=True)
                suffix_only=not _materialize_full_suffix and clock>=window.activation_s
                continue
            if not _materialize_full_suffix and (suffix_only or clock>=window.activation_s):
                suffix_only=True
            if index==0 and state.phase=='travelling':
                if not policy.deterministic_current_arc_continuation or state.travel is None:
                    raise BoundaryUncertified('ongoing travel requires explicit deterministic continuation model')

                elapsed=state.travel.elapsed_time_s
                remaining=[]
                for leg in state.travel.path.legs:
                    if elapsed>=leg.duration_s:
                        elapsed-=leg.duration_s
                    else:
                        remaining.append(leg)
                        break
                if remaining:
                    first=remaining[0]
                    offset=next(i for i,leg in enumerate(state.travel.path.legs) if leg is first)
                    unentered_at_node=(elapsed==0. and position.node_id is not None and
                        position==network.segment_position(first.segment_id,first.start_offset_m))
                    if not unentered_at_node:
                        travel_leg(first,position,started=True)


                        continuation=(state.travel.path.legs[offset+1:]
                                      if worker_routing.activity_domain.all_road_positions else ())
                        for leg in continuation:
                            if not _materialize_full_suffix and (suffix_only or clock>=window.activation_s):
                                suffix_only=True
                                break
                            travel_leg(leg,position)


            if not _materialize_full_suffix and clock>=window.activation_s:
                suffix_only=True
            if position!=destination and (_materialize_full_suffix or not suffix_only):
                path=worker_routing.route(position,destination,worker.profile)
                if path is None:
                    raise BoundaryUncertified('old service continuation has no known finite path')
                for leg in path.legs:
                    if not _materialize_full_suffix and (suffix_only or clock>=window.activation_s):
                        suffix_only=True
                        break
                    travel_leg(leg,position)
            if not suffix_only or _materialize_full_suffix:
                allow(destination,service=True)
            serial+=1;bid=f'{wid}:{visit.operation}:{serial}'
            ready=task.pickup_ready_time_s if visit.operation=='pickup' else task.delivery_ready_time_s
            service_start=max(clock,ready) if ready is not None else clock
            if not _materialize_full_suffix and (clock>=window.activation_s or service_start>=window.activation_s):
                suffix_only=True
            confirmed=policy.all_remaining_service_confirmed and (_materialize_full_suffix or not suffix_only)
            duration=task.service_duration_s(visit.operation)
            blocks.append(ActionBlock(bid,visit.operation,(wid,),(position,),destination,duration,
                policy.model_evidence,task_id=visit.task_id,confirmed=confirmed,safe_to_complete=confirmed))
            order.append(bid)
            if confirmed:
                position=destination;clock=service_start+duration
            if not confirmed or worker.service_mode!='delivery':
                suffix_only=True
        orders[wid]=tuple(order)
        guarantees[wid]=tuple(legal.values())
    return orders,tuple(blocks),guarantees
