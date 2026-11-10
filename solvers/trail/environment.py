from dataclasses import replace
import math
import time
from solvers.tfr.local_solver import LocalEnvironment, NORMAL

def geometry_turn(env):
    started = time.perf_counter()
    try:
        return env.geometry_turn()
    finally:
        if env.training_clock:
            env.deployment_elapsed_s += time.perf_counter() - started

def settle_environment(env):
    while env.status in NORMAL and env.geometry_pool:
        previous = env.best.total_tardiness_s
        if not geometry_turn(env):
            break
        if env.best.total_tardiness_s < previous and env.round_index < env.config.max_rounds:
            env.status = None
            env.stale_rounds = 0
    return env.status in NORMAL and (not env.geometry_pool)

class StructureEnvironment(LocalEnvironment):

    def __init__(self, *args, **kwargs):
        self.initial_structure_keys = set()
        self.policy_structure_keys = set()
        self.policy_gain_T = 0.0
        self.initial_archive_gain_T = 0.0
        self.policy_actions = 0
        self.policy_fields = 0
        self.decoded_structures = 0
        self.policy_decode_failures = 0
        self.initialized = False
        super().__init__(*args, **kwargs)

    def _geometry_incumbent(self, value):
        previous = None if self.best is None else self.best.total_tardiness_s
        context = self.geometry_pool.candidate_context or {}
        key = context.get('entry_key') or self.geometry_pool.key(value.plan)
        super()._geometry_incumbent(value)
        if self.initialized and previous is not None and (value.total_tardiness_s < previous):
            gain = previous - value.total_tardiness_s
            if key in self.policy_structure_keys and key not in self.initial_structure_keys:
                self.policy_gain_T += gain
            else:
                self.initial_archive_gain_T += gain

    def reset(self):
        obs = super().reset()
        if obs is None:
            return None
        self.initial_structure_keys = set(self.geometry_pool.entries)
        self.initialized = True
        return None if settle_environment(self) else self.observation()

    def step_policy(self, macro):
        if self._decision_started is None:
            self.begin_decision()
        obs = self.observation(context=self.decision_context)
        if macro.source_witness_digest != obs.source_witness_digest:
            raise ValueError('The structure action belongs to another current solution')
        if not 0 < len(macro.slots) <= self.config.proposal_slots:
            raise ValueError('A structure action requires one to three proposal slots')
        for slot in macro.slots:
            if slot.status in ('candidate', 'demonstration'):
                if slot.structure is None or slot.script is None or (not slot.fields):
                    raise ValueError('A candidate requires its complete structure and field trajectory')
                if slot.script.source_witness_digest != obs.source_witness_digest:
                    raise ValueError('The structure script belongs to another current solution')
                choices = tuple(((field.kind, field.candidates[field.selected].identity) for field in slot.fields))
                if slot.fields[0].kind != 'group' or slot.fields[-1].kind != 'end' or choices != tuple(((choice.kind, choice.identity) for choice in slot.script.choices)):
                    raise ValueError('The complete structure and policy field trajectory differ')
        (policy_before, initial_before) = (self.policy_gain_T, self.initial_archive_gain_T)
        previous = self.best.total_tardiness_s
        self.policy_actions += 1
        self.work_statistics['policy_graph_encodings'] += 1
        self.work_statistics['decoder_fields'] += sum((len(slot.fields) for slot in macro.slots))
        if macro.graph is not None:
            self.work_statistics['feature_graph_nodes_peak'] = max(self.work_statistics['feature_graph_nodes_peak'], len(macro.graph.identities))
            self.work_statistics['feature_graph_edges_peak'] = max(self.work_statistics['feature_graph_edges_peak'], len(macro.graph.edges))
        for slot in macro.slots:
            self.policy_fields += len(slot.fields)
            self.policy_decode_failures += slot.status.startswith('decode_failure')
            if slot.structure is not None:
                self.decoded_structures += 1
                key = self.geometry_pool.key(slot.structure)
                if key not in self.initial_structure_keys:
                    self.policy_structure_keys.add(key)
        executable = replace(macro, slots=tuple((replace(slot, script=None) for slot in macro.slots)))
        (_, _, _, info) = self.step(executable)
        if not info.get('closed'):
            return (self.observation(), 0.0, True, info)
        if self.geometry_pool:
            geometry_turn(self)
        done = settle_environment(self)
        policy_gain = self.policy_gain_T - policy_before
        initial_gain = self.initial_archive_gain_T - initial_before
        actual_gain = previous - self.best.total_tardiness_s
        if not math.isclose(policy_gain + initial_gain, actual_gain, rel_tol=1e-09, abs_tol=1e-06):
            raise ValueError('Attributed structure gains differ from the actual global-best decrease')
        scale = max(self.initial_T, 1.0)
        info.update(normal_terminal=done, status=self.status, reward_attribution=dict(policy_gain_T=policy_gain, initial_archive_gain_T=initial_gain, actual_global_gain_T=actual_gain))
        return (None if done else self.observation(), policy_gain / scale, done, info)

    def policy_diagnostics(self):
        best_key = None if self.best is None else self.geometry_pool.key(self.best.plan)
        return dict(actions=self.policy_actions, decoder_fields=self.policy_fields, decoded_structures=self.decoded_structures, decode_failures=self.policy_decode_failures, policy_structure_count=len(self.policy_structure_keys), policy_gain_T=self.policy_gain_T, initial_archive_gain_T=self.initial_archive_gain_T, final_best_origin='policy' if best_key in self.policy_structure_keys - self.initial_structure_keys else 'initial_H' if best_key in self.initial_structure_keys else None)
