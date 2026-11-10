from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import time

from common.evaluation._recovery_cache import RecoveryEvaluationCache
from common.evaluation.recovery_plan import evaluate_recovery_plan, RecoveryPlanValidationError
from common.recovery.plan import RecoveryPlan
from solvers.tfr.evaluator import CertifiedSolution, compact_solution, seeded_candidate, validate_search_plan
from solvers.tfr.geometry.search import GeometryArchive
from solvers.tfr.geometry.reuse import GeometryResources
from solvers.tfr.interfaces.boundary import input_digest
from solvers.tfr.structure import check_structure, structure_key
from solvers.tfr.work import StructureWork, WorkLimitReached


class GeometryPool:
    def __init__(self, item, routing, config, *, check=lambda: None, publish=None, progress=None,
                 cooperative_quantum=16384):
        if type(cooperative_quantum) is not int or cooperative_quantum < 1:
            raise ValueError('cooperative quantum must be a positive integer')
        self.item, self.routing, self.config = item, routing, config
        self.check, self.publish = check, publish
        self.progress = progress
        self.cooperative_quantum = cooperative_quantum
        self.candidate_context = None
        routing.register_query_input(item)
        from common.evaluation._recovery_cache import shared_cache
        self.cache = shared_cache(routing) or RecoveryEvaluationCache(routing, route_key_reuse=True)
        self.resources = GeometryResources(item, check, scope_key=input_digest(item))
        self.entries = OrderedDict()
        self.queue = []
        self.active_key = None
        self.last_verified = None
        self.global_best_T_s = None
        self._background_owner = None
        self.stats = dict(turns=0, structures=0, completed=0, budget_stopped=0,
            physical_checks=0, physical_passes=0, physical_receipt_hits=0, validation_s=0., geometry_s=0.,
            geometry_nodes=0, lp_calls=0, lp_iterations=0, phase_s={}, turn_phase_s={})

    def _activate(self,key):


        self.active_key=key

    @staticmethod
    def key(plan):
        from common.fingerprint import digest as digest_key
        return digest_key(structure_key(plan))

    def __bool__(self):
        return bool(self.queue)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:yield
        finally:
            elapsed=time.perf_counter()-started
            phases=self.stats.setdefault('phase_s',{})
            phases[name]=phases.get(name,0.)+elapsed
            if name == 'geometry':self.stats['geometry_s'] += elapsed

    def validate(self, plan):
        self.check()
        if self.last_verified is not None and self.last_verified.plan == plan:
            self.stats['physical_receipt_hits'] += 1
            return self.last_verified
        self.stats['physical_checks'] += 1
        started=time.perf_counter()
        try:value = validate_search_plan(self.item,plan,self.routing,cache=self.cache,
            ablation=getattr(self.config,'ablation','full'))
        finally:self.stats['validation_s']+=time.perf_counter()-started
        self.stats['physical_passes']+=1
        self.last_verified = value
        return value

    def _archive(self, plan, work):
        boundary = getattr(self.item, 'execution_boundary', None)
        return GeometryArchive(self.item, self.routing, plan, self.check,
            validate=self.validate, measure=self.measure, resources=self.resources,
            first_solution_priority=True, work_budget=work,
            global_upper=lambda:self.global_best_T_s,
            pause_after_domains=True,
            initialization_quantum=self.cooperative_quantum,
            withdrawal_deadlines=({w.id:None for w in self.item.workers if w.service_mode=='handover_only'}
                if boundary is None else dict(boundary.execution_contract.get('withdrawal_deadlines', {}))))

    def offer(self, plan):
        if not isinstance(plan, RecoveryPlan):plan = RecoveryPlan.from_dict(plan)
        ablation=getattr(self.config,'ablation','full')
        if ablation=='no_transfer' and plan.transfers:return None
        if ablation=='sender_position':
            from solvers.tfr.ablation import fixed_sender_plan
            try:plan=fixed_sender_plan(self.item,plan)
            except ValueError:return None
        key = self.key(plan)
        if key in self.entries:return key
        checked = check_structure(self.item, plan)
        if not checked.valid:return None
        work = StructureWork(self.config.geometry_nodes, self.config.lp_calls, self.config.lp_maxiter)
        self.entries[key] = dict(plan=plan, work=work, archive=None, seeded=False,
            closed=False, best=None, stop=None)
        self.queue.append(key)
        self.stats['structures'] += 1
        return key

    def _account(self, entry, before):
        for name, value in entry['work'].used.items():self.stats[name] += value-before[name]
        before.update(entry['work'].used)

    def _progress(self, key, entry, before):
        self._account(entry, before)
        if entry['archive'] is not None:self._compact(entry)
        if self.progress is not None:self.progress()

    def _candidate(self, key, entry, value, *, phase):
        if value is None:return None
        previous = entry['best']
        if previous is not None and value.total_tardiness_s >= previous.total_tardiness_s:return None
        compact = value if isinstance(value, CertifiedSolution) else compact_solution(self.item, value)
        entry['best'] = compact
        if self.global_best_T_s is None or compact.total_tardiness_s<self.global_best_T_s:
            self.global_best_T_s=compact.total_tardiness_s

        archive = entry['archive']
        if archive is not None and (archive.best is None or compact.total_tardiness_s < archive.best.total_tardiness_s):
            archive.best, archive._best_exact = compact, None
        self.candidate_context = dict(entry_key=key, phase=phase)
        try:
            if self.publish is not None:self.publish(compact)
        finally:self.candidate_context = None
        return compact

    def _compact(self, entry):
        archive = entry['archive']
        if archive is None:return
        for name in ('best', '_pending_value'):
            value = getattr(archive, name)
            if value is not None and not isinstance(value, CertifiedSolution):
                setattr(archive, name, compact_solution(self.item, value))

    def _close(self, key, entry):
        if entry['closed']:return
        entry['closed'] = True
        if self._background_owner==key:self._background_owner=None
        if key in self.queue:self.queue.remove(key)
        self.stats['budget_stopped' if entry['stop'] is not None else 'completed'] += 1


        entry['summary'] = (dict(entry['archive'].stats) if entry['archive'] is not None else
            dict(fixed_meeting_count=len(entry['plan'].transfers),movable_meeting_count=0,
                geometry_nodes=0,lp_calls=0,lp_iterations=0))
        entry['frontier'] = (entry['archive'].frontier_summary() if entry['archive'] is not None else
            dict(fixed_position_schedule_complete=True))
        entry['archive'] = None
        entry['best_T_s']=None if entry['best'] is None else entry['best'].total_tardiness_s
        entry['best']=entry['plan']=None

    def start(self, key):

        entry = self.entries[key]
        if entry['seeded'] or entry['closed']:return entry['best']
        self._activate(key)
        if getattr(self.config,'ablation','full') in ('sender_position','no_transfer'):


            try:
                value=self.validate(entry['plan'])
                self._candidate(key,entry,value,phase=('no_transfer_schedule'
                    if self.config.ablation=='no_transfer' else 'fixed_sender_schedule'))
            except RecoveryPlanValidationError:
                pass
            entry['seeded']=True
            result=entry['best']
            self._close(key,entry)
            counter='no_transfer_schedules' if self.config.ablation=='no_transfer' else 'fixed_sender_schedules'
            self.stats[counter]=self.stats.get(counter,0)+1
            self._progress(key,entry,dict(entry['work'].used))
            return result
        before = dict(entry['work'].used)
        archive = entry['archive']
        if archive is None:archive = entry['archive'] = self._archive(entry['plan'], entry['work'])
        try:

            if not entry.get('fast_attempted'):
                try:
                    value = archive._validate(entry['plan'])
                    self._candidate(key, entry, value, phase='structure_seed')
                except RecoveryPlanValidationError:pass
                entry['fast_attempted']=True
                self._progress(key, entry, before)
            if entry['plan'].transfers:
                self.check();archive.step()
                self._progress(key, entry, before)
                self._local_seed(key,entry)
            else:
                self._candidate(key, entry, archive.step(), phase='geometry')
                self._progress(key, entry, before)
            entry['seeded'] = True
        except WorkLimitReached as error:
            entry['stop'] = str(error);entry['seeded'] = True
        finally:
            self._account(entry, before)
            self._compact(entry)
        result=entry['best']
        if archive.done or entry['stop'] is not None:self._close(key, entry)
        return result

    def _local_seed(self,key,entry):
        archive=entry['archive']
        if entry.get('local_seed_attempted') or archive._initial_domains is None or archive.done:return
        def publish(value):
            phase=('local_seed_lp' if archive.stats.get('witness_seed_lp_accepted',0) else 'structure_seed')
            return self._candidate(key,entry,value,phase=phase)
        value=seeded_candidate(archive,entry['work'],on_candidate=publish)
        publish(value)
        entry['local_seed_attempted']=True

    def step(self):

        self.check()
        if not self.queue:return None
        key=self._background_owner
        owner=self.entries.get(key)
        archive=None if owner is None else owner['archive']
        if (key not in self.queue or archive is None or not archive._background_nodes
                or archive._background_objective is not None):
            key=self.queue[0]
            archive=self.entries[key]['archive']
            self._background_owner=(key if archive is not None and archive._background_nodes
                and archive._background_objective is None else None)
        self._activate(key)
        if self._background_owner==key:
            self.stats['background_owner_turns']=self.stats.get('background_owner_turns',0)+1
        entry = self.entries[key]
        started=time.perf_counter()
        phase=('seed' if not entry['seeded'] else
            'initialization:'+entry['archive'].initialization_phase
            if not entry['archive']._initialized else 'search')
        if not entry['seeded']:
            result = self.start(key)
        else:
            before = dict(entry['work'].used)
            result = None
            try:
                result = self._candidate(key, entry, entry['archive'].step(), phase='geometry')
                self._local_seed(key,entry)
            except WorkLimitReached as error:entry['stop'] = str(error)
            finally:
                self._account(entry, before)
                self._compact(entry)
            if entry['archive'].done or entry['stop'] is not None:self._close(key, entry)
        if key in self.queue:
            self.queue.remove(key);self.queue.append(key)
        self.stats['turns'] += 1

        self._progress(key, entry, dict(entry['work'].used))
        phases=self.stats.setdefault('turn_phase_s',{})
        phases[phase]=phases.get(phase,0.)+time.perf_counter()-started
        return result


    def snapshot(self):
        from common.recovery.methods.roads_exact import _SourceSearch
        states = {}
        for entry in self.entries.values():
            archive = entry['archive']
            if archive is None:continue
            for cursor in (archive._initialization_cursor, archive._expansion_cursor):
                if cursor is None:continue
                while cursor is not None:
                    frame=getattr(cursor,'gi_frame',None)
                    state=None if frame is None else frame.f_locals.get('state')
                    if isinstance(state,_SourceSearch):states[id(state)]=state
                    cursor=getattr(cursor,'gi_yieldfrom',None)
        self.resources.observe_private_sources(tuple(states.values()))
        return dict(self.stats, pending_structures=len(self.queue),
            global_best_T_s=self.global_best_T_s,
            background_warm_owner=self._background_owner,
            cooperative_quantum=self.cooperative_quantum,
            physical_cache=dict(self.cache.stats), resources=self.resources.snapshot(),
            global_optimality_claim=False)
