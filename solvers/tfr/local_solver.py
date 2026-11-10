from common.config import TFR_DEFAULTS
from dataclasses import dataclass, asdict
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import os
import time

from common.recovery.plan import RecoveryPlan
from solvers.tfr.evaluator import CertifiedSolution, evaluate_structure
from solvers.tfr.structure import check_structure, release_tasks, structure_key


_UNSET = object()


def _spatial_work(stats):
    return {key:value for key,value in stats.items() if key.startswith(('spatial_','handover_'))
        or key in ('initial_search_radius_m','max_search_radius_m','radius_expansions',
                   'candidate_protocol','local_empty_queries','relation_queries','qualified_relations')}


def _restore_spatial_work(stats,previous):
    for key,value in previous.items():
        if type(value)not in(int,float):continue
        if key=='initial_search_radius_m':continue
        if 'max_'in key or 'peak_'in key or key in ('spatial_cache_entries','spatial_cache_nodes'):
            stats[key]=max(stats.get(key,0),value)
        else:stats[key]=stats.get(key,0)+value


@dataclass(frozen=True)
class LocalConfig:
    max_rounds: int = TFR_DEFAULTS['max_rounds']
    patience: int = TFR_DEFAULTS['patience']
    proposal_slots: int = TFR_DEFAULTS['proposal_slots']
    geometry_nodes: int = TFR_DEFAULTS['geometry_nodes']
    lp_calls: int = TFR_DEFAULTS['lp_calls']
    lp_maxiter: int = TFR_DEFAULTS['lp_maxiter']
    seed: int = TFR_DEFAULTS['seed']
    revision_domain: str = 'full'
    initial_domain: str = 'full'
    ablation: str = TFR_DEFAULTS['ablation']

    def __post_init__(self):
        if self.revision_domain!='full' or self.initial_domain!='full':raise ValueError('H uses complete service-structure reconstruction')
        if self.ablation not in ('full','no_transfer','sender_position','task_tardiness'):raise ValueError('unknown H ablation')
        for name in ('max_rounds','patience','proposal_slots','geometry_nodes','lp_calls','lp_maxiter'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(name+' must be positive')
        if self.proposal_slots!=3:raise ValueError('H emits at most three complete structures per revision')
        if type(self.seed) is not int:raise ValueError('seed must be an integer')


@dataclass
class LocalResult:
    plan: object
    status: str
    diagnostics: dict

    def to_dict(self):
        return dict(method=self.diagnostics.get('method','h'),
                    protocol='local_recovery',status=self.status,
                    T_s=None if self.plan is None else self.plan.total_tardiness_s,
                    diagnostics=self.diagnostics)


def construct_structures(item, base_record=None, task_ids=None, max_structures=3, resume_state=None, revision_domain='full', routing_seed=None, ablation='full', *, reuse_routing=False, evaluation_cache=None, local_context=None, cancel=None, progress=None, exclude_structure_digests=()):

    from common.data.roads.routing import Routing
    from common.recovery.solver import SolverConfig
    from common.recovery.adapters._common import SearchEngine
    from common.recovery.adapters._local_search import InsertionBudget
    from solvers.tfr.local_insertions import insertions
    from solvers.tfr.priority import construction_priority
    routing = Routing(item.roads,cache_profile='formal',search_method='scipy') if routing_seed is None else routing_seed if reuse_routing else routing_seed.fork_query_session()
    routing.register_query_input(item)
    engine = SearchEngine(item, routing, SolverConfig(method='greedy', meeting_pool='legacy', objective_mode='total_tardiness', seed=0), evaluation_cache=evaluation_cache, local_context=local_context)
    priority = construction_priority(engine)
    if cancel is not None:
        original_check = engine.check_budget

        def check():
            cancel()
            original_check()
        engine.check_budget = check
    local = engine.local
    tasks = local.repair_task_ids if task_ids is None else frozenset(task_ids)
    base = local.base_plan if base_record is None else release_tasks(RecoveryPlan.from_dict(base_record), tasks)
    if not tasks:
        return dict(plans=[base.to_dict()], statistics=dict(local.stats), ledger={}, complete_structures_generated=1, complete_structures_rejected=0, construction_rejections={})
    ordered = sorted(tasks, key=lambda t: (int(engine.holders.get(t) not in {w.id for w in item.workers if w.service_mode == 'handover_only'}), item.task_map[t].promised_delivery_time_s, t))
    ledger = InsertionBudget(engine, tasks, state=None if resume_state is None else resume_state['ledger'])
    excluded = set(exclude_structure_digests)
    beam = [(0.0, 0.0, base)]
    first_task = 0
    if resume_state is not None:
        if tuple(resume_state['ordered']) != tuple(ordered):
            raise ValueError('construction cursor task order differs')
        first_task = resume_state['next_task']
        beam = [(late, distance, RecoveryPlan.from_dict(record)) for late, distance, record in resume_state['beam']]
        engine.stats.update(resume_state['statistics'])
        local.stats = engine.stats['local_scope']
        local.stats.update(spatial_cache_entries=0, spatial_cache_nodes=0)
    complete_generated = 0 if resume_state is None else resume_state.get('complete_structures_generated', len(beam) if first_task == len(ordered) else 0)
    complete_rejected = 0 if resume_state is None else resume_state.get('complete_structures_rejected', 0)
    rejections = {} if resume_state is None else dict(resume_state.get('construction_rejections', {}))
    for index, task in enumerate(ordered):
        if index < first_task:
            continue
        pending = frozenset(ordered[index + 1:])
        options = []
        seen = set()
        complete_generated = 0
        complete_rejected = 0
        import heapq
        from itertools import islice
        streams = [iter(insertions(engine, plan, task, ledger, domain=revision_domain, pending_tasks=pending, ablation=ablation)) for _, _, plan in beam]
        serial = 0

        def next_option(parent):
            nonlocal serial, complete_generated, complete_rejected
            old_late, old_distance, plan = beam[parent]
            remaining = max_structures - complete_generated
            if not pending and remaining <= 0:
                return None
            stream = islice(streams[parent], remaining) if not pending and (not excluded) else streams[parent]
            for delta in stream:
                if not pending and complete_generated >= max_structures:
                    return None
                candidate = delta.materialize()
                key = structure_key(candidate)
                if not pending and excluded:
                    from common.fingerprint import digest as digest_key
                    if digest_key(key) in excluded:
                        ledger.charge_evaluation(task, delta.worker_id)
                        rejections['previously_explored'] = rejections.get('previously_explored', 0) + 1
                        continue
                complete_generated += int(not pending)
                if key in seen:
                    rejections['duplicate'] = rejections.get('duplicate', 0) + 1
                    complete_rejected += int(not pending)
                    continue
                seen.add(key)
                if not ledger.charge_evaluation(task, delta.worker_id):
                    rejections['evaluation_budget'] = rejections.get('evaluation_budget', 0) + 1
                    complete_rejected += int(not pending)
                    continue
                checked = check_structure(item, candidate, pending=pending)
                if checked.errors or (not pending and checked.gaps):
                    for code in (*checked.errors, *(('capacity',) if checked.gaps else ())):
                        rejections[code] = rejections.get(code, 0) + 1
                    complete_rejected += int(not pending)
                    continue
                changed = tuple(delta.changed_sequences())
                delta_late, delta_distance = priority.change(plan, candidate, changed)
                late = old_late + delta_late
                distance = old_distance + delta_distance
                if math.isnan(late) or math.isnan(distance):
                    late, distance = (math.inf, math.inf)
                if pending:
                    from solvers.tfr.local_insertions import capacity_risk
                    risk = capacity_risk(item, candidate, ignored=pending)
                else:
                    risk = (len(checked.gaps), sum((gap.excess for gap in checked.gaps)))
                serial += 1
                return (*risk, late, distance, serial, parent, candidate)
            return None
        frontier = []
        try:
            for parent in range(len(beam)):
                candidate = next_option(parent)
                if candidate is not None:
                    heapq.heappush(frontier, candidate)
            while frontier and len(options) < max_structures:
                candidate = heapq.heappop(frontier)
                options.append(candidate)
                if len(options) < max_structures:
                    following = next_option(candidate[-2])
                    if following is not None:
                        heapq.heappush(frontier, following)
        finally:
            for stream in streams:
                stream.close()
        if not options:
            return dict(plans=[], statistics={**local.stats, **engine.stats}, ledger=ledger.by_task, failed_task=task, complete_structures_generated=complete_generated, complete_structures_rejected=complete_rejected, construction_rejections=rejections)
        options.sort(key=lambda row: row[:5])
        beam = [(late, distance, plan) for count, excess, late, distance, ordinal, parent, plan in options]
        if progress is not None:
            progress(dict(next_task=index + 1, ordered=tuple(ordered), complete_structures_generated=complete_generated, complete_structures_rejected=complete_rejected, construction_rejections=rejections, beam=[(late, distance, plan.to_dict()) for late, distance, plan in beam], ledger=ledger.by_task, statistics=dict(engine.stats)))
    return dict(plans=[p.to_dict() for _, _, p in beam], complete_structures_generated=complete_generated, complete_structures_rejected=complete_rejected, construction_rejections=rejections, statistics={**local.stats, **engine.stats}, ledger=ledger.by_task)


def _rule_prefix(previous, kind, identity):
    return hashlib.sha256(previous+json.dumps((kind, identity),
        separators=(',', ':'),ensure_ascii=False).encode()).digest()


def rule_revisions(item, solution_record, initial_T, best_T, round_index, stale_rounds, elapsed_s, resume_state=None, revision_domain='full', temporal=None, routing_seed=None, history=None, search_radii=None, group_attempts=None, exclude_structure_digests=(), selected_group=None, ablation='full', *, reuse_routing=False, local_context=None, cancel=None, progress=None):

    from common.data.roads.routing import Routing
    from common.recovery.local import LocalRecoveryContext
    from solvers.tfr.interfaces.revisions import RevisionObservation
    value = CertifiedSolution.from_record(solution_record)
    routing = Routing(item.roads,cache_profile='formal',search_method='scipy') if routing_seed is None else routing_seed if reuse_routing else routing_seed.fork_query_session()
    routing.register_query_input(item)
    local = local_context or LocalRecoveryContext(item, routing, analyze_reference=False)
    if local.item is not item or local.routing is not routing.for_activity_domain(None):
        raise ValueError('rule context belongs to a different input or routing session')
    spatial_before = _spatial_work(local.stats)
    previous_spatial = (resume_state or {}).get('spatial_search', {})

    def spatial_work():
        result = _spatial_work(local.stats)
        for key, value in tuple(result.items()):
            if type(value) not in (int, float):
                continue
            if key == 'initial_search_radius_m' or 'max_' in key or 'peak_' in key:
                result[key] = max(value, previous_spatial.get(key, value))
            elif key in ('spatial_cache_entries', 'spatial_cache_nodes'):
                continue
            else:
                result[key] = previous_spatial.get(key, 0) + value - spatial_before.get(key, 0)
        return result
    from solvers.tfr.quality import own_tardiness_groups
    obs = RevisionObservation(item, value, local, initial_T, best_T, round_index=round_index, stale_rounds=stale_rounds, elapsed_s=elapsed_s, revision_domain=revision_domain, temporal=temporal, check=cancel, groups=own_tardiness_groups(value) if ablation == 'task_tardiness' else None, search_radii=dict((resume_state or {}).get('search_radii', search_radii or {})))
    obs.ablation = ablation
    from solvers.tfr.quality import joint_feedback_groups
    from solvers.tfr.feedback_search import full_revisions
    if ablation != 'task_tardiness':
        obs.groups = joint_feedback_groups(value, obs.groups, analysis=obs.temporal)
    result = full_revisions(obs, history=history, group_attempts=group_attempts, resume_state=resume_state, progress=progress, exclude_structure_digests=exclude_structure_digests, selected_group=selected_group)
    result['rejections'] = result.get('failure_reasons', [])
    return result


NORMAL = ('work_complete', 'no_improvement', 'zero_tardiness', 'empty_action_domain')


class LocalEnvironment:


    def __init__(self, item, routing, *, config=None, s_ref=1.0, origin=None, training=False, group_chooser=None):
        self.item, self.routing = (item, routing)
        self.config = config or LocalConfig()
        if s_ref < 1 or not math.isfinite(s_ref):
            raise ValueError('s_ref must be finite and at least one second')
        self.s_ref = s_ref
        self.origin = time.perf_counter() if origin is None else origin
        self.current = self.best = None
        self.initial_T = None
        self.round_index = self.stale_rounds = 0
        self.tried, self.events = (set(), [])
        self.rule_history = {}
        self.group_attempts = {}
        self.search_radii = {}
        self.teacher_callback = None
        self.group_chooser = group_chooser
        self.training_clock = bool(training)
        self.status = self.first_solution_s = self.revision_origin = None
        self.deployment_elapsed_s = 0.0
        self._decision_started = None
        self.local = self._rule_local = None
        self._restored_online_spatial = {}
        self._temporal_cache = OrderedDict()
        self._quality_groups_cache = OrderedDict()
        self.initialization = {}
        self._initial_build_state = self._rule_build_state = None
        self._initial_plans = None
        self._initial_cursor = 0
        self._initial_task_ids = ()
        self.pending_round = None
        self.decision_context = None
        self._pause_reason = None
        self._evaluation_interrupted = False
        self._inflight_operation = None
        self.resource_interruption_history = []
        self._coordination_cache = self._coordination_resources = None
        from common.evaluation.path_facts import capture_input_paths
        self._input_path_facts = capture_input_paths(routing)
        from solvers.tfr.geometry_pool import GeometryPool
        self.geometry_pool = GeometryPool(item, routing, self.config, check=self.check, publish=self._geometry_incumbent)
        self.work_statistics = dict(geometry_nodes=0, lp_calls=0, lp_iterations=0, complete_structure_requests=0, returned_structure_evaluations=0, independent_validations=0, independent_validation_s=0.0, complete_validation_attempts=0, complete_validation_passes=0, geometry_validation_s=0.0, evaluation_worker_wall_s=0.0, geometry_phase_s={}, geometry_cache={}, geometry_stats_last={}, unknown_evaluation_units=0, unknown_construction_units=0, unknown_rule_units=0, repeated_unfinished_evaluation_units=0, construction_tasks_completed=0, construction_generated_branches=0, construction_joint_evaluations=0, complete_initial_structures_generated=0, complete_initial_structures_rejected=0, construction_rejections={}, rule_generated_branches=0, raw_slots=0, proposals_generated=0, complete_proposals=0, decoder_fields=0, decode_failures=0, policy_graph_encodings=0, feature_graph_nodes_peak=0, feature_graph_edges_peak=0, illegal_proposals=0, duplicate_proposals=0, unresolved_proposals=0, accepted_proposals=0, best_improvements=0, unexecuted_slots=0, quality_gate_rejections=0, pool_duplicate_proposals=0, geometry_stats_sum={}, geometry_stats_max={}, unknown_reasons={}, rule_failures={})

    @staticmethod
    def _ledger_totals(state):
        rows = (state or {}).get('ledger', {})
        return tuple((sum((row.get(name, 0) for workers in rows.values() for row in workers.values())) for name in ('generated', 'evaluated')))

    def _unknown_work(self, kind, reason):
        self.work_statistics['unknown_' + kind + '_units'] += 1
        reasons = self.work_statistics['unknown_reasons']
        key = kind + ':' + reason
        reasons[key] = reasons.get(key, 0) + 1
        self.resource_interruption_history.append(dict(reason=reason, operation=kind, recorded_elapsed_s=time.perf_counter() - self.origin, work_amount='unknown_uncommitted_unit', round=self.round_index))

    def budget_compliance(self):
        if self.round_index > self.config.max_rounds:
            return 'round_limit_exceeded'
        unknown = any((self.work_statistics['unknown_' + kind + '_units'] for kind in ('evaluation', 'construction', 'rule')))
        return 'unverified_after_interrupted_unit' if unknown else 'verified_recorded_work_limits'

    def is_formal_complete(self):
        return self.status in NORMAL and self.budget_compliance() == 'verified_recorded_work_limits'

    def work_record(self):
        record = dict(self.work_statistics)
        if self.geometry_pool is not None:
            record['two_pool_geometry'] = self.geometry_pool.snapshot()
        construction = self.initialization.get('statistics', (self._initial_build_state or {}).get('statistics', {}))
        record['spatial_search'] = dict(construction=_spatial_work(construction.get('local_scope', construction)), rule_workers=dict(self.work_statistics.get('rule_spatial_search', {})), online_observation=dict(self._restored_online_spatial) if self.local is None else _spatial_work(self.local.stats))
        unknown = self.work_statistics['unknown_evaluation_units'] > 0
        for name in ('geometry_nodes', 'lp_calls', 'lp_iterations', 'independent_validations', 'independent_validation_s', 'complete_validation_attempts', 'complete_validation_passes', 'geometry_validation_s'):
            record['known_' + name] = record[name]
            if unknown:
                record[name] = None
        for kind, names in (('construction', ('construction_generated_branches', 'construction_joint_evaluations', 'complete_initial_structures_generated', 'complete_initial_structures_rejected')), ('rule', ('rule_generated_branches', 'decoder_fields', 'proposals_generated'))):
            for name in names:
                record['known_' + name] = record[name]
                if self.work_statistics['unknown_' + kind + '_units']:
                    record[name] = None
        record.update(evaluation_totals_complete=not unknown, construction_totals_complete=self.work_statistics['unknown_construction_units'] == 0, rule_totals_complete=self.work_statistics['unknown_rule_units'] == 0, counter_scope='actual returned worker receipts; interrupted work is unknown, known counts are lower bounds', recovery_units='completed D tasks, generated scripts, evaluated candidates and closed rounds; only unfinished units may repeat', timing_components_nested=True, complexity_claim=False)
        return record

    def _geometry_incumbent(self, value):
        if self.initial_T is None:
            self.current = self.best = value
            self.initial_T = value.total_tardiness_s
            self.first_solution_s = time.perf_counter() - self.origin
            self.revision_origin = time.perf_counter()
            self._publish(value, 'h_initial')
        elif value.total_tardiness_s < self.best.total_tardiness_s:
            self.current = self.best = value
            self.work_statistics['best_improvements'] += 1
            if self.status in NORMAL and self.round_index < self.config.max_rounds:
                self.status = None
                self.stale_rounds = 0
            if self.pending_round is not None:
                self.pending_round['accepted_at'] = self._revision_elapsed()
            self._publish(value, 'geometry_improvement')

    def _sync_geometry_work(self):
        totals = self.geometry_pool.stats
        for name in ('geometry_nodes', 'lp_calls', 'lp_iterations'):
            self.work_statistics[name] = totals[name]
        self.work_statistics['complete_validation_attempts'] = totals['physical_checks']
        self.work_statistics['complete_validation_passes'] = totals['physical_passes']
        self.work_statistics['geometry_validation_s'] = totals['validation_s']
        self.work_statistics['geometry_phase_s'] = dict(totals.get('phase_s', {}))

    def geometry_turn(self):
        from solvers.tfr.geometry.bounds import NativeQueryPaused
        if not self.geometry_pool:
            return False
        self._inflight_operation = 'evaluation'
        try:
            self.geometry_pool.step()
        except NativeQueryPaused:
            self._inflight_operation = None
            self._sync_geometry_work()
            self.resource_interruption_history.append(dict(reason='native_query_pause_exact', operation='evaluation', recorded_elapsed_s=time.perf_counter() - self.origin, work_amount='exact_saved_query_state', round=self.round_index))
            raise
        except OSError:
            self._inflight_operation = None
            self.status = 'disk_protection_stop'
            return False
        self._inflight_operation = None
        self._sync_geometry_work()
        return self.status != 'disk_protection_stop'

    def _capture_construction(self, state):
        state = deepcopy(state)
        previous = self._initial_build_state
        old_gen, old_eval = self._ledger_totals(previous)
        new_gen, new_eval = self._ledger_totals(state)
        self.work_statistics['construction_generated_branches'] += max(0, new_gen - old_gen)
        self.work_statistics['construction_joint_evaluations'] += max(0, new_eval - old_eval)
        self.work_statistics['construction_tasks_completed'] += max(0, state.get('next_task', 0) - (previous or {}).get('next_task', 0))
        for field in ('generated', 'rejected'):
            key = 'complete_structures_' + field
            self.work_statistics['complete_initial_structures_' + field] += max(0, state.get(key, 0) - (previous or {}).get(key, 0))
        self.work_statistics['construction_rejections'] = dict(state.get('construction_rejections', {}))
        self._initial_build_state = state

    def _capture_rules(self, state):
        state = deepcopy(state)
        previous = self._rule_build_state or {}
        totals = self.work_statistics.setdefault('rule_spatial_search', {})
        for key, value in state.get('spatial_search', {}).items():
            if type(value) not in (int, float):
                totals[key] = value
            elif key == 'initial_search_radius_m' or 'max_' in key or 'peak_' in key or (key in ('spatial_cache_entries', 'spatial_cache_nodes')):
                totals[key] = max(totals.get(key, 0), value)
            else:
                totals[key] = totals.get(key, 0) + max(0, value - previous.get('spatial_search', {}).get(key, 0))
        self.work_statistics['proposals_generated'] += max(0, len(state['slots']) - len(previous.get('slots', ())))
        old_fields = sum(previous.get('field_counts', ()))
        self.work_statistics['decoder_fields'] += max(0, sum(state.get('field_counts', ())) - old_fields)
        old_gen = sum((row['total'] for row in previous.get('generation_budget', {}).values()))
        new_gen = sum((row['total'] for row in state.get('generation_budget', {}).values()))
        self.work_statistics['rule_generated_branches'] += max(0, new_gen - old_gen)
        for reason in state.get('failure_reasons', ())[len(previous.get('slots', ())):]:
            if reason:
                failures = self.work_statistics.setdefault('rule_failures', {})
                failures[reason] = failures.get(reason, 0) + 1
                if 'structure:' in reason:
                    self.work_statistics['illegal_proposals'] += 1
        self.rule_history[state['source']] = state['history']
        if self.config.revision_domain == 'full':
            self.group_attempts = dict(state.get('group_attempts', self.group_attempts))
        self.search_radii.update(state.get('search_radii', {}))
        if self.decision_context is not None:
            self.decision_context['search_radii'] = dict(self.search_radii)
        self._rule_build_state = state

    def build_rules(self):
        context = self.decision_context
        if context is None:
            raise RuntimeError('begin_decision must precede rule generation')
        observation = self.observation(context=context)
        selected_group = None if self.group_chooser is None else self.group_chooser(observation)
        self.last_group_observation = observation
        self.last_selected_group = selected_group
        return self._run(rule_revisions, self.item, self.current.to_record(), self.initial_T, context['best_T'], context['round_index'], context['stale_rounds'], context['elapsed_s'], self._rule_build_state, self.config.revision_domain, observation.temporal, self.routing, self.rule_history.get(observation.source_witness_digest), dict(self.search_radii), dict(self.group_attempts), tuple(self.geometry_pool.entries) if self.geometry_pool else (), selected_group, self.config.ablation, on_progress=self._capture_rules)

    def request_pause(self, reason='user_stop'):
        self._pause_reason = reason

    def _run(self, operation, *args, on_progress=None):
        kind = 'construction' if operation is construct_structures else 'rule' if operation is rule_revisions else 'evaluation'
        self._inflight_operation = kind
        started = time.perf_counter()
        try:
            cache = self.geometry_pool.cache
            if operation is construct_structures:
                result = operation(*args, reuse_routing=True, evaluation_cache=cache, cancel=self.check, progress=on_progress)
            elif operation is _evaluate_with_limits:
                result = operation(*args, evaluation_cache=cache, cancel=self.check)
            else:
                if self._rule_local is None:
                    from common.recovery.local import LocalRecoveryContext
                    self._rule_local = LocalRecoveryContext(self.item, self.routing, analyze_reference=False, evaluation_cache=cache)
                result = operation(*args, reuse_routing=True, local_context=self._rule_local, cancel=self.check, progress=on_progress)
            return dict(status='returned', result=result, elapsed_s=time.perf_counter() - started)
        except TimeoutError:
            self._unknown_work(kind, 'external_pause')
            return dict(status='external_pause', elapsed_s=time.perf_counter() - started)
        finally:
            self._inflight_operation = None

    def _evaluate(self, plan):
        from solvers.tfr.geometry.bounds import NativeQueryPaused
        self.work_statistics['complete_structure_requests'] += 1
        if self._evaluation_interrupted:
            self.work_statistics['repeated_unfinished_evaluation_units'] += 1
            self._evaluation_interrupted = False
        if self._pause_reason is not None:
            self._unknown_work('evaluation', 'external_pause')
            self._evaluation_interrupted = True
            return dict(status='external_pause', elapsed_s=0.0)
        started = time.perf_counter()
        key = self.geometry_pool.offer(plan)
        if key is None:
            self.work_statistics['returned_structure_evaluations'] += 1
            self.work_statistics['illegal_proposals'] += 1
            return dict(status='returned', result=dict(status='structure_invalid'))
        if self.geometry_pool.entries[key]['closed']:
            self.work_statistics['duplicate_proposals'] += 1
            return dict(status='returned', result=dict(status='duplicate'))
        self._inflight_operation = 'evaluation'
        try:
            value = self.geometry_pool.start(key)
            while value is None and (not self.geometry_pool.entries[key]['closed']):
                result = self.geometry_pool.step()
                if result is not None and self.geometry_pool.key(result.plan) == key:
                    value = result
                else:
                    value = self.geometry_pool.entries[key]['best']
        except NativeQueryPaused as error:
            self._sync_geometry_work()
            self._evaluation_interrupted = False
            self.resource_interruption_history.append(dict(reason='native_query_pause_exact', operation='evaluation', recorded_elapsed_s=time.perf_counter() - self.origin, work_amount='exact_saved_query_state', round=self.round_index))
            return dict(status='external_pause', elapsed_s=time.perf_counter() - started, committed_query_state=True)
        except TimeoutError as error:
            self._sync_geometry_work()
            self._unknown_work('evaluation', str(error))
            self._evaluation_interrupted = True
            return dict(status='external_pause', elapsed_s=time.perf_counter() - started)
        except OSError as error:
            self._sync_geometry_work()
            return dict(status='disk_protection_stop', elapsed_s=time.perf_counter() - started, error=str(error))
        finally:
            self._inflight_operation = None
        self._sync_geometry_work()
        self.work_statistics['returned_structure_evaluations'] += 1
        return dict(status='returned', elapsed_s=time.perf_counter() - started, result=dict(status='unresolved') if value is None else dict(status='validated_feasible', solution=value.to_record()))

    def _publish(self, value, source):
        elapsed = time.perf_counter() - self.origin
        boundary = getattr(self.item, 'execution_boundary', None)
        fixed = 0.0 if boundary is None else boundary.observed_tardiness_s + boundary.committed_tardiness_upper_s
        event = dict(elapsed_s=elapsed, T_s=value.total_tardiness_s, suffix_T_s=value.total_tardiness_s, fixed_T_s=fixed, total_T_s=value.total_tardiness_s + fixed, source=source, round=self.round_index)
        context = self.geometry_pool.candidate_context
        if context is not None:
            event.update(structure_key=context['entry_key'], candidate_phase=context['phase'])
        self.events.append(event)

    def reset(self):
        if self._initial_plans is None:
            built = self._run(construct_structures, self.item, None, None, 3, self._initial_build_state, self.config.initial_domain, self.routing, self.config.ablation, on_progress=self._capture_construction)
            if built['status'] != 'returned':
                self.status = built['status']
                return None
            previous = self._initial_build_state or {}
            old_gen, old_eval = self._ledger_totals(previous)
            new_gen, new_eval = self._ledger_totals(built['result'])
            self.work_statistics['construction_generated_branches'] += max(0, new_gen - old_gen)
            self.work_statistics['construction_joint_evaluations'] += max(0, new_eval - old_eval)
            self.work_statistics['complete_initial_structures_generated'] += max(0, built['result'].get('complete_structures_generated', len(built['result']['plans'])) - previous.get('complete_structures_generated', 0))
            self.work_statistics['complete_initial_structures_rejected'] += max(0, built['result'].get('complete_structures_rejected', 0) - previous.get('complete_structures_rejected', 0))
            self.work_statistics['construction_rejections'] = dict(built['result'].get('construction_rejections', {}))
            ledger = built['result'].get('ledger', {})
            self._initial_task_ids = tuple(ledger)
            rows = [row for workers in ledger.values() for row in workers.values()]
            self.initialization = {k: v for k, v in built['result'].items() if k not in ('plans', 'ledger')}
            self.initialization['ledger_totals'] = dict(tasks=len(ledger), worker_rows=len(rows), **{name: sum((row.get(name, 0) for row in rows)) for name in ('generated', 'evaluated', 'emitted')}, max_generated_per_worker=max((row.get('generated', 0) for row in rows), default=0))
            self._initial_plans = built['result']['plans']
            self._initial_build_state = None
        while self._initial_cursor < len(self._initial_plans):
            plan = self._initial_plans[self._initial_cursor]
            reply = self._evaluate(plan)
            if reply['status'] != 'returned':
                self.status = reply['status']
                return None
            self._initial_cursor += 1
            result = reply['result']
            if result['status'] == 'validated_feasible':
                if self.geometry_pool is None:
                    self.current = self.best = CertifiedSolution.from_record(result['solution'])
                    self.initial_T = self.best.total_tardiness_s
                    self.first_solution_s = time.perf_counter() - self.origin
                    self.revision_origin = time.perf_counter()
                    self._publish(self.best, 'h_initial')
                else:
                    for remaining in self._initial_plans[self._initial_cursor:]:
                        self.geometry_pool.offer(remaining)
                self._initial_plans = None
                self._initial_task_ids = ()
                self.status = 'zero_tardiness' if self.initial_T == 0 else None
                return self.observation()
        self._initial_task_ids = ()
        self.status = 'initialization_failed'
        return None

    def observation(self, *, value=None, context=None):
        from common.recovery.local import LocalRecoveryContext
        from solvers.tfr.interfaces.revisions import RevisionObservation, source_witness_digest
        from solvers.tfr.temporal import analyze
        value = self.current if value is None else value
        if value is None:
            return None
        if self.local is None:
            self.local = LocalRecoveryContext(self.item, self.routing, analyze_reference=False, evaluation_cache=self._coordination_cache)
            _restore_spatial_work(self.local.stats, self._restored_online_spatial)
            self._restored_online_spatial = {}
        digest = source_witness_digest(self.item, value)
        if digest not in self._temporal_cache:
            self._temporal_cache[digest] = analyze(value, source_key=digest)
        self._temporal_cache.move_to_end(digest)
        while len(self._temporal_cache) > 2:
            self._temporal_cache.popitem(last=False)
        from solvers.tfr.quality import own_tardiness_groups
        observation = RevisionObservation(self.item, value, self.local, self.initial_T, self.best.total_tardiness_s if context is None else context['best_T'], round_index=self.round_index if context is None else context['round_index'], stale_rounds=self.stale_rounds if context is None else context['stale_rounds'], elapsed_s=self._revision_elapsed() if context is None else context['elapsed_s'], tried=set(self.tried) if context is None else set(context['tried']), max_rounds=self.config.max_rounds, patience=self.config.patience, revision_domain=self.config.revision_domain, check=self.check, temporal=self._temporal_cache[digest], witness_digest=digest, groups=own_tardiness_groups(value) if self.config.ablation == 'task_tardiness' else None, search_radii=dict(self.search_radii if context is None else context.get('search_radii', self.search_radii)))
        observation.ablation = self.config.ablation
        if self.config.revision_domain == 'full':
            from solvers.tfr.quality import joint_feedback_groups
            if digest not in self._quality_groups_cache:
                self._quality_groups_cache[digest] = observation.groups if self.config.ablation == 'task_tardiness' else joint_feedback_groups(value, observation.groups, analysis=self._temporal_cache[digest])
            self._quality_groups_cache.move_to_end(digest)
            while len(self._quality_groups_cache) > 2:
                self._quality_groups_cache.popitem(last=False)
            observation.groups = self._quality_groups_cache[digest]
            observation.full_group_attempts = dict(self.group_attempts)
        return observation

    def _revision_elapsed(self):
        if not self.training_clock:
            return 0.0 if self.revision_origin is None else max(0.0, time.perf_counter() - self.revision_origin)
        return self.deployment_elapsed_s + (0.0 if self._decision_started is None else time.perf_counter() - self._decision_started)

    def begin_decision(self):
        if self._decision_started is not None:
            raise RuntimeError('a macro decision is already active')
        if self.pending_round is not None:
            self.decision_context = self.pending_round['decision_context']
        elif self._rule_build_state is None or self.decision_context is None:
            self.decision_context = dict(best_T=self.best.total_tardiness_s, round_index=self.round_index, stale_rounds=self.stale_rounds, elapsed_s=self._revision_elapsed(), tried=set(self.tried), search_radii=dict(self.search_radii))
        self._decision_started = time.perf_counter()
        self._inflight_operation = 'rule'
        source = None if self.pending_round is None else CertifiedSolution.from_record(self.pending_round['source_solution'])
        return self.observation(value=source, context=self.decision_context)

    def _close_decision(self):
        elapsed = 0.0 if self._decision_started is None else time.perf_counter() - self._decision_started
        if self.training_clock:
            self.deployment_elapsed_s += elapsed
        else:
            self.deployment_elapsed_s = self._revision_elapsed()
        self._decision_started = None
        return elapsed

    def check(self):
        if self._pause_reason is not None:
            raise TimeoutError(self._pause_reason)

    @property
    def has_pending_macro(self):
        return self.pending_round is not None

    def _candidate_key(self, source, plan):
        return hashlib.sha256(json.dumps((source, structure_key(plan), self.config.geometry_nodes, self.config.lp_calls, self.config.lp_maxiter), separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

    def _stage_macro(self, macro):
        obs = self.observation(context=self.decision_context)
        if macro is None:
            raise ValueError('a revision macro is required')
        if macro.source_witness_digest != obs.source_witness_digest:
            raise ValueError('revision source differs from current solution')
        if len(macro.slots) > self.config.proposal_slots:
            raise ValueError('too many revision proposals')
        self.search_radii.update(macro.search_radii)
        self.decision_context['search_radii'] = dict(self.search_radii)
        self.work_statistics['raw_slots'] += len(macro.slots)
        slots = []
        for slot in macro.slots:
            if slot.script is not None:
                raise ValueError('full H revisions require complete service structures')
            structure = slot.structure
            if structure is not None:
                self.work_statistics['complete_proposals'] += 1
            if slot.status == 'duplicate':
                self.work_statistics['duplicate_proposals'] += 1
            if slot.status.startswith('decode_failure') or slot.status == 'construction_failed':
                self.work_statistics['decode_failures'] += 1
            slots.append(dict(status=slot.status, script=None, plan=None if structure is None else structure.to_dict()))
        expand_tasks = set(macro.expand_tasks) | set((self._rule_build_state or {}).get('expand_tasks', ()))
        self._rule_build_state = None
        self._inflight_operation = None
        self.pending_round = dict(source=obs.source_witness_digest, source_solution=self.current.to_record(), slots=slots, cursor=0, outcomes=[], old_best=self.decision_context['best_T'], decision_context=dict(self.decision_context), expand_tasks=tuple(sorted(expand_tasks)), accepted=False, accepted_at=None, teacher_done=False)
        return True

    def _teacher(self, pending):
        if pending['accepted'] and (not pending['teacher_done']) and (self.teacher_callback is not None):
            from solvers.tfr.feedback_search import StructureProposal
            source = CertifiedSolution.from_record(pending['source_solution'])
            slot = pending['slots'][pending['cursor'] - 1]
            proposal = StructureProposal(pending['source'], RecoveryPlan.from_dict(slot['plan']), ())
            self.teacher_callback(self.observation(value=source, context=pending['decision_context']), proposal, self.current)
            pending['teacher_done'] = True

    def _interrupt(self, reason):
        self.status = reason
        self._inflight_operation = None
        deployed = self._close_decision()
        return (self.observation(), 0.0, True, dict(closed=False, normal_terminal=False, status=self.status, outcomes=[] if self.pending_round is None else self.pending_round['outcomes'], deployment_elapsed_s=deployed))

    def step(self, macro=None):
        from solvers.tfr.interfaces.revisions import DecodeFailure
        if self._decision_started is None:
            self.begin_decision()
        if self.status is not None:
            deployed = self._close_decision()
            return (self.observation(), 0.0, True, dict(closed=True, normal_terminal=self.status in NORMAL, status=self.status, deployment_elapsed_s=deployed))
        if self.pending_round is None and (not self._stage_macro(macro)):
            return self._interrupt('disk_protection_stop')
        pending = self.pending_round
        source = CertifiedSolution.from_record(pending['source_solution'])
        obs = self.observation(value=source, context=pending['decision_context'])
        while (self.geometry_pool is not None or not pending['accepted']) and pending['cursor'] < len(pending['slots']):
            try:
                self.check()
            except TimeoutError as error:
                return self._interrupt(str(error))
            slot = pending['slots'][pending['cursor']]
            if slot['script'] is None and slot.get('plan') is None or slot['status'] not in ('candidate', 'demonstration'):
                pending['outcomes'].append(slot['status'])
                pending['cursor'] += 1
                continue
            try:
                plan = RecoveryPlan.from_dict(slot['plan'])
                checked = check_structure(self.item, plan, check_budget=self.check)
                if not checked.valid:
                    raise DecodeFailure('structure:' + ','.join(checked.errors) + (':capacity' if checked.gaps else ''))
            except DecodeFailure as error:
                self.work_statistics['decode_failures'] += 1
                if 'structure:' in str(error):
                    self.work_statistics['illegal_proposals'] += 1
                pending['outcomes'].append('decode_failure:' + str(error))
                pending['cursor'] += 1
                continue
            key = self._candidate_key(obs.source_witness_digest, plan)
            pool_duplicate = self.geometry_pool is not None and self.geometry_pool.key(plan) in self.geometry_pool.entries
            if structure_key(plan) == structure_key(obs.plan) or key in self.tried or pool_duplicate:
                self.work_statistics['duplicate_proposals'] += 1
                self.work_statistics['pool_duplicate_proposals'] += int(pool_duplicate)
                pending['outcomes'].append('duplicate')
                pending['cursor'] += 1
                continue
            reply = self._evaluate(plan.to_dict())
            if reply['status'] != 'returned':
                return self._interrupt(reply['status'])
            self.tried.add(key)
            pending['cursor'] += 1
            result = reply['result']
            pending['outcomes'].append(result['status'])
            if result['status'] == 'validated_feasible':
                value = CertifiedSolution.from_record(result['solution'])
                if value.total_tardiness_s > self.initial_T:
                    self.work_statistics['quality_gate_rejections'] += 1
                    pending['outcomes'][-1] = 'quality_gate_rejected'
                else:
                    self.current = value
                    pending['accepted'] = True
                    self.work_statistics['accepted_proposals'] += 1
                    if value.total_tardiness_s < self.best.total_tardiness_s:
                        self.work_statistics['best_improvements'] += 1
                        self.best = value
                        pending['accepted_at'] = self._revision_elapsed()
                        self._publish(value, 'revision_improvement')
        self._teacher(pending)
        if pending['accepted']:
            self.work_statistics['unexecuted_slots'] += len(pending['slots']) - pending['cursor']
        if not pending['accepted']:
            tasks = set(pending['expand_tasks'])
            for slot in pending['slots']:
                if slot['script'] is not None:
                    tasks.update(json.loads(slot['script']['choices'][0][1]))
            for task in sorted(tasks):
                current = self.search_radii.get(task, self.local.search_radius_m)
                radius = self.local.next_search_radius(task, current)
                if radius is None:
                    continue
                previous = set(self.local.actor_pool(task, radius_m=current))
                following = next((row for row in self.local.iter_worker_rings(task, radius_m=radius) if set(row[1]) - previous), None)
                if following is not None:
                    self.search_radii[task] = following[0]
        self.round_index += 1
        improved = self.best.total_tardiness_s < pending['old_best']
        self.stale_rounds = 0 if improved else self.stale_rounds + 1
        if self.best.total_tardiness_s == 0:
            self.status = 'zero_tardiness'
        elif self.round_index >= self.config.max_rounds:
            self.status = 'work_complete'
        elif self.stale_rounds >= self.config.patience:
            self.status = 'no_improvement'
        next_obs = self.observation()
        if self.status is None and (not next_obs.groups):
            self.status = 'empty_action_domain'
        reward = 0.0 if not improved else math.exp(-pending['accepted_at'] / self.s_ref) * (pending['old_best'] - self.best.total_tardiness_s) / self.initial_T
        outcomes = pending['outcomes']
        self.pending_round = None
        self.decision_context = None
        deployed = self._close_decision()
        next_obs.elapsed_s = self.deployment_elapsed_s
        return (next_obs, reward, self.status is not None, dict(closed=True, normal_terminal=self.status in NORMAL, status=self.status, outcomes=outcomes, deployment_elapsed_s=deployed))

    def finalize(self):
        if self.best is None or self.status not in NORMAL:
            return
        started = time.perf_counter()
        from common.evaluation.path_facts import fresh_path_cache
        from solvers.tfr.evaluator import validate_search_plan
        verified = validate_search_plan(self.item, self.best.plan, self.routing, cache=fresh_path_cache(self.routing, self._input_path_facts), ablation=self.config.ablation)
        if verified.total_tardiness_s != self.best.total_tardiness_s:
            raise ValueError('final physical objective differs from incumbent')
        self.work_statistics['independent_validations'] += 1
        self.work_statistics['independent_validation_s'] += time.perf_counter() - started
        self.work_statistics['complete_validation_attempts'] += 1
        self.work_statistics['complete_validation_passes'] += 1


def _evaluate_with_limits(item,record,nodes,calls,iterations,source=None,selected=None,routing_seed=None,
                          first_feasible=False,ablation='full',*,resources=None,evaluation_cache=None,cancel=lambda:None):
    return evaluate_structure(item,record,geometry_limit=nodes,lp_limit=calls,lp_maxiter=iterations,
        source_plan_record=source,selected_tasks=selected,routing_seed=routing_seed,first_feasible=first_feasible,
        resources=resources,evaluation_cache=evaluation_cache,cancel=cancel,ablation=ablation)


def solve_local(item,routing,config=None,observer=None,*,s_ref=1.,teacher_callback=None,origin=None):
    env=LocalEnvironment(item,routing,config=config,s_ref=s_ref,origin=origin)
    env.teacher_callback=teacher_callback
    return _run_local(env,observer)


def _run_local(env,observer):
    from solvers.tfr.interfaces.revisions import MacroAction,ProposalSlot
    obs = env.reset()
    while obs is not None:


        if env.status is not None:
            if env.status not in NORMAL or not env.geometry_pool:break
            old=env.best.total_tardiness_s
            if not env.geometry_turn():break
            if env.best.total_tardiness_s<old and env.round_index<env.config.max_rounds:
                env.status=None;env.stale_rounds=0
                obs=env.observation()
            continue
        if env.has_pending_macro:
            obs,reward,done,info=env.step()
            if env.geometry_pool and not env.geometry_turn():break
            obs=env.observation()
            continue
        if not obs.groups:
            env.status='empty_action_domain'
            continue
        env.begin_decision()
        built=env.build_rules()
        if built['status']!='returned':
            env._interrupt(built['status'])
            break
        from solvers.tfr.feedback_search import StructureProposal
        slots=tuple(ProposalSlot(None,(),'construction_failed') if p is None else
            ProposalSlot(None,(),'candidate',structure=p.plan) if isinstance(p,StructureProposal) else
            ProposalSlot(p['script'],(),'candidate') for p in built['result']['proposals'])
        macro=MacroAction(obs.source_witness_digest,slots,
            search_radii=built['result'].get('search_radii',{}),
            expand_tasks=tuple(built['result'].get('expand_tasks',())))
        obs,reward,done,info=env.step(macro)
        if env.geometry_pool and not env.geometry_turn():break
        obs=env.observation()
    env.finalize()
    if observer is not None and env.best is not None:
        observer(env.best)
    return _local_result(env)


def _local_result(env):
    return LocalResult(env.best, env.status, dict(protocol='local_recovery',revision_domain=env.config.revision_domain,
        first_solution_s=env.first_solution_s, elapsed_s=time.perf_counter()-env.origin,
        rounds=env.round_index, stale_rounds=env.stale_rounds,
        initial_T=env.initial_T, statistics=getattr(env, 'initialization', {}),
        events=env.events,work=env.work_record(),algorithm_terminated_normally=env.status in NORMAL,
        budget_compliance=env.budget_compliance(),resource_interruption_history=env.resource_interruption_history,
        normal_completion=env.is_formal_complete()))
