from copy import deepcopy
from dataclasses import dataclass
import json
import math

from common.recovery.plan import RecoveryPlan
from common.fingerprint import digest as digest_key
from solvers.tfr.quality import FeedbackPriority, select_groups
from solvers.tfr.structure import structure_key


@dataclass(frozen=True)
class StructureProposal:
    source_witness_digest: str
    plan: RecoveryPlan
    group: tuple

    def to_record(self):
        return dict(source=self.source_witness_digest, plan=self.plan.to_dict(), group=tuple(self.group))

    def __reduce__(self):


        return type(self).from_record, (self.to_record(),)

    @classmethod
    def from_record(cls, record):
        return cls(record['source'], RecoveryPlan.from_dict(record['plan']), tuple(record['group']))


def _generation_budget(ledger):
    return {task: dict(total=sum(row['generated'] for row in workers.values()),
        workers={worker: row['generated'] for worker, row in workers.items()})
        for task, workers in ledger.items()}


def _opportunity(obs, group):

    priority = FeedbackPriority(obs, group)
    holders = obs.local.holders
    added = []
    for task in group:
        obs.tick()
        owner = holders.get(task)
        if owner is None:
            ranks = (priority.pickup(task, worker) for worker in obs.workers(task))
        else:
            ranks = [priority.receiver(task, owner, worker) for worker in obs.workers(task, owner) if worker != owner]
            if owner in obs.local.basic_workers(task):ranks.append(priority.receiver(task, owner, None))
        best = min(ranks, default=(math.inf,) * 3)
        added.append(best[0])
    original = math.fsum(obs.evaluated.tardiness_s[task] for task in group)
    estimate = math.fsum(added)-original
    return estimate if math.isfinite(estimate) else math.inf


def _choose_group(obs, attempts):
    if getattr(obs,'ablation','full')=='task_tardiness':


        def priority(group):
            key=json.dumps(group,separators=(',',':'))
            count=attempts.get(key,attempts.get(group,0))
            return count,-math.fsum(obs.evaluated.tardiness_s[task]for task in group),len(group),tuple(group)
        return min(obs.groups,key=priority)


    ordered = select_groups(obs.groups, attempts, slots=len(obs.groups))
    choices = list(ordered[:3])
    if obs.round_index % 2 and obs.groups and obs.groups[0] not in choices:
        choices[-1:] = [obs.groups[0]]
    return min(choices, key=lambda group: (_opportunity(obs, group),
        attempts.get(json.dumps(group, separators=(',', ':')), 0), obs.groups.index(group)))


def full_revisions(obs, *, history=None, group_attempts=None, resume_state=None,
                   progress=None, max_structures=3, exclude_structure_digests=(),selected_group=None):


    from solvers.tfr.local_solver import construct_structures, _spatial_work
    if type(max_structures) is not int or not 1 <= max_structures <= 3:
        raise ValueError('H feedback emits at most three complete structures')
    if selected_group is not None:
        selected_group=tuple(selected_group)
        if selected_group not in obs.groups:
            raise ValueError('selected group is outside the issued full H domain')
    source = obs.source_witness_digest
    prior = deepcopy(history or {})
    attempts = dict(group_attempts or {})
    if resume_state is not None:
        if resume_state.get('source') != source:
            raise ValueError('feedback cursor belongs to a different solution')
        if resume_state.get('max_structures') != max_structures:
            raise ValueError('feedback cursor candidate allowance changed')
        if selected_group is not None and selected_group!=tuple(resume_state['group']):
            raise ValueError('selected group differs from the saved full H frontier')
        prior, attempts = deepcopy(resume_state['history']), dict(resume_state['group_attempts'])
        if resume_state.get('completed'):
            proposals = [StructureProposal.from_record(row) for row in resume_state['slots']]
            return dict(proposals=proposals, history=prior, group_attempts=attempts,
                search_radii=dict(resume_state['search_radii']), generation_budget=deepcopy(resume_state['generation_budget']),
                statistics=deepcopy(resume_state.get('statistics', {})),
                expand_tasks=tuple(resume_state.get('expand_tasks', ())),
                failure_reasons=list(resume_state.get('failure_reasons', ())))
        group = tuple(resume_state['group'])
        if group not in obs.groups:raise ValueError('H feedback group differs from its source domain')
        construction = deepcopy(resume_state.get('construction'))
    else:
        if not obs.groups:
            return dict(proposals=[], history=prior, group_attempts=attempts, search_radii=dict(obs.search_radii),
                generation_budget={}, statistics={}, expand_tasks=(), failure_reasons=['empty_group_domain'])
        group = tuple(_choose_group(obs, attempts)if selected_group is None else selected_group)
        key = json.dumps(group, separators=(',', ':'))
        attempts[key] = attempts.get(key, 0) + 1
        prior.setdefault('attempts', {})[key] = prior.get('attempts', {}).get(key, 0) + 1
        construction = None
    closed = set(prior.get('closed_structures', ()))
    excluded = closed | set(exclude_structure_digests) | {digest_key(structure_key(obs.plan))}
    prior['closed_structures'] = closed
    proposals = []
    statistics = {}
    failure_reasons = []

    def state(*, completed=False):
        ledger = {} if construction is None else construction.get('ledger', {})
        return dict(source=source, group=group,
            max_structures=max_structures,
            history=deepcopy(prior), group_attempts=dict(attempts), round_groups=(group,),
            construction=deepcopy(construction), generation_budget=_generation_budget(ledger),
            search_radii=dict(obs.search_radii), spatial_search=_spatial_work(obs.local.stats),
            slots=[proposal.to_record() for proposal in proposals], field_counts=[0] * len(proposals),
            failure_reasons=list(failure_reasons), expand_tasks=() if proposals else group,
            statistics=deepcopy(statistics), completed=completed)

    def capture(value):
        nonlocal construction
        construction = deepcopy(value)
        if progress is not None:progress(state())


    if progress is not None:progress(state())
    result = construct_structures(obs.item, base_record=obs.plan.to_dict(), task_ids=group,
        max_structures=max_structures, resume_state=construction, revision_domain='full',
        ablation=getattr(obs,'ablation','full'),
        routing_seed=obs.routing, reuse_routing=True, evaluation_cache=obs.local.evaluation_cache,
        local_context=obs.local, cancel=obs.tick, progress=capture,
        exclude_structure_digests=excluded)
    statistics = dict(result.get('statistics', {}))
    statistics.update(complete_structures_generated=result.get('complete_structures_generated', len(result['plans'])),
        complete_structures_rejected=result.get('complete_structures_rejected', 0),
        construction_rejections=dict(result.get('construction_rejections', {})))
    if construction is None:
        construction = dict(ledger=result.get('ledger', {}))
    else:
        construction['ledger'] = deepcopy(result.get('ledger', construction.get('ledger', {})))
    for record in result['plans']:
        plan = RecoveryPlan.from_dict(record)
        identity = digest_key(structure_key(plan))
        if identity in excluded:continue
        excluded.add(identity);closed.add(identity)
        proposals.append(StructureProposal(source, plan, group))
    prior['closed_structures'] = closed
    if not proposals:failure_reasons.append('no_new_complete_structure')
    if progress is not None:progress(state(completed=True))
    return dict(proposals=proposals, history=prior, group_attempts=attempts,
        search_radii=dict(obs.search_radii), generation_budget=_generation_budget(construction.get('ledger', {})),
        statistics=statistics, expand_tasks=() if proposals else group, failure_reasons=failure_reasons)
