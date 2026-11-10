from dataclasses import dataclass, replace
from pathlib import Path
import math
import os
import torch
from torch import nn
from torch.distributions import Categorical
from solvers.tfr.interfaces.revisions import FEATURE_DIM, FIELD_TYPES, RevisionObservation, MacroAction, ProposalSlot, FieldDecision, DecodeFailure
from . import decoder as codec
CONTEXT_PREFIX = 'decision-task:'
NODE_FEATURE_DIM = 24
RELATIONS = 16

@dataclass
class RelationGraph:
    identities: tuple[str, ...]
    features: object
    edges: object
    progress: tuple[float, ...]

def _row(values, *, origin=None, destination=None):
    row = [float(v) for v in values] + [0.0] * (NODE_FEATURE_DIM - len(values))
    if origin is not None:
        row[20:22] = [v / 1000.0 for v in origin]
    if destination is not None:
        row[22:24] = [v / 1000.0 for v in destination]
    return tuple(row)

def graph_observation(obs: RevisionObservation):
    obs.expandable = False
    tasks = {t for group in obs.groups for t in group}
    workers = {w.id for w in obs.item.workers if tasks.intersection(w.carried_task_ids)}
    holders = {t: w.id for w in obs.item.workers for t in w.carried_task_ids}
    for task in sorted(tasks):
        obs.workers(task, holders.get(task))
        workers.update(obs.worker_pools[task])
    for (worker, route) in obs.plan.sequences.items():
        if any((v.task_id in tasks for v in route)):
            workers.add(worker)
    for event in obs.plan.transfers:
        if tasks.intersection(event.tasks_i_to_j + event.tasks_j_to_i):
            workers.update((event.worker_i, event.worker_j))
    temporal = obs.temporal
    visible = {(worker, index) for worker in workers for index in range(len(obs.evaluated.routes[worker]))}
    meetings = {temporal.rows[key].visit.transfer_id for key in visible if temporal.rows[key].visit.operation == 'transfer'}
    for identifier in meetings:
        visible.update(temporal.meeting_visits[identifier].values())
    all_workers = workers | {worker for (worker, index) in visible}
    identities = []
    features = []
    edges = []
    index = {}

    def add(identity, row):
        if identity not in index:
            index[identity] = len(identities)
            identities.append(identity)
            features.append(row)
        return index[identity]

    def edge(a, b, kind):
        edges.append((a, b, kind))
        edges.append((b, a, kind + 1))
    scale = max(obs.initial_T, 1.0)
    boundary = getattr(obs.item, 'execution_boundary', None)
    for task in sorted(tasks):
        data = obs.item.task_map[task]
        origin = obs.item.worker_map[holders[task]].location if task in holders else data.pickup
        add('task:' + task, _row((1, 0, 0, 0, data.load, obs.evaluated.tardiness_s[task] / scale, (data.promised_delivery_time_s - obs.item.time_s) / 3600.0, 0.0 if data.pickup_ready_time_s is None else (data.pickup_ready_time_s - obs.item.time_s) / 3600.0, (data.delivery_ready_time_s - obs.item.time_s) / 3600.0, int(task in holders), temporal.task_wait(task) / 3600.0, temporal.task_downstream(task) / scale), origin=obs.item.roads.coordinates(origin), destination=obs.item.roads.coordinates(data.delivery)))
    for worker in sorted(all_workers):
        data = obs.item.worker_map[worker]
        release = obs.item.time_s if boundary is None else boundary.release_times[worker]
        downstream = 0.0 if not obs.evaluated.routes[worker] else temporal.visit_downstream((worker, 0))
        add('worker:' + worker, _row((0, 1, 0, 0, data.capacity, data.profile.speed_factor, (release - obs.item.time_s) / 3600.0, sum((obs.item.task_map[t].load for t in data.carried_task_ids)), len(data.carried_task_ids), len(obs.plan.sequences[worker]) / 100.0, int(data.service_mode == 'handover_only'), obs.local.search_radius_m / 1000.0, data.handover_range_m / 1000.0, downstream / scale), origin=obs.item.roads.coordinates(obs.anchor(worker)), destination=obs.item.roads.coordinates(data.location)))
    for task in sorted(tasks):
        if task in holders:
            edge(index['task:' + task], index['worker:' + holders[task]], 6)
    for (worker, offset) in sorted(visible):
        row = temporal.rows[worker, offset]
        visit = row.visit
        identity = f'visit:{worker}:{offset}'
        vi = add(identity, _row((0, 0, 1, 0, (row.arrival_time_s - obs.item.time_s) / 3600.0, (row.start_time_s - obs.item.time_s) / 3600.0, (row.completion_time_s - obs.item.time_s) / 3600.0, row.waiting_time_s / 3600.0, row.incoming_path.duration_s / 3600.0, row.load_after, int(visit.operation == 'pickup'), int(visit.operation == 'delivery'), int(visit.operation == 'transfer'), temporal.visit_downstream((worker, offset)) / scale, len(row.carried_task_ids)), origin=obs.item.roads.coordinates(row.incoming_path.origin), destination=obs.item.roads.coordinates(row.position)))
        edge(index['worker:' + worker], vi, 0)
        previous = (worker, offset - 1)
        if previous in visible:
            predecessor = index[f'visit:{worker}:{offset - 1}']
            edge(predecessor, vi, 2)
            if (previous, (worker, offset)) in temporal.tight_visit_edges:
                edge(predecessor, vi, 14)
        elif offset == 0:
            release = obs.item.time_s if boundary is None else boundary.release_times[worker]
            if abs(release + row.incoming_path.duration_s - row.start_time_s) <= 1e-07 and row.waiting_time_s <= 1e-07:
                edge(index['worker:' + worker], vi, 14)
        if visit.operation in ('pickup', 'delivery') and visit.task_id in tasks:
            edge(index['task:' + visit.task_id], vi, 4)
        elif visit.operation == 'transfer':
            event = obs.plan.transfer_map[visit.transfer_id]
            (outgoing, incoming) = (event.tasks_i_to_j, event.tasks_j_to_i) if worker == event.worker_i else (event.tasks_j_to_i, event.tasks_i_to_j)
            for task in outgoing:
                if task in tasks:
                    edge(index['task:' + task], vi, 10)
            for task in incoming:
                if task in tasks:
                    edge(index['task:' + task], vi, 12)
    for identifier in sorted(meetings):
        event = obs.plan.transfer_map[identifier]
        pair = [temporal.rows[key] for (worker, key) in sorted(temporal.meeting_visits[identifier].items())]
        (left, right) = pair
        ei = add('meeting:' + identifier, _row((0, 0, 0, 1, (left.start_time_s - obs.item.time_s) / 3600.0, (left.completion_time_s - obs.item.time_s) / 3600.0, (left.arrival_time_s - obs.item.time_s) / 3600.0, (right.arrival_time_s - obs.item.time_s) / 3600.0, left.waiting_time_s / 3600.0, right.waiting_time_s / 3600.0, sum((obs.item.task_map[t].load for t in event.tasks_i_to_j + event.tasks_j_to_i)), len(event.tasks_i_to_j), len(event.tasks_j_to_i), temporal.event_downstream(identifier) / scale, obs.item.source_settings.transfer_duration_s / 3600.0), destination=obs.item.roads.coordinates(event.location)))
        for (worker, offset) in temporal.meeting_visits[identifier].values():
            edge(index[f'visit:{worker}:{offset}'], ei, 8)
    if not identities:
        add('empty', _row((0.0,)))
    return RelationGraph(tuple(identities), torch.tensor(features, dtype=torch.float32), torch.tensor(edges, dtype=torch.long).reshape(-1, 3), obs.progress())

class RelationLayer(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.message = nn.ModuleList((nn.Linear(hidden, hidden, bias=False) for _ in range(RELATIONS)))
        self.self_update = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edges):
        result = torch.zeros_like(x)
        degree = torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)
        for (relation, linear) in enumerate(self.message):
            selection = edges[:, 2] == relation
            (source, destination) = (edges[selection, 0], edges[selection, 1])
            if source.numel():
                result.index_add_(0, destination, linear(x[source]))
                degree.index_add_(0, destination, torch.ones((len(source), 1), device=x.device))
        return self.norm(x + torch.relu(self.self_update(x) + result / degree.clamp_min(1)))

def _names(node):
    if node is None:
        return ()
    return node if isinstance(node, tuple) else (node,)

def _without_context(node):
    return tuple((name for name in _names(node) if not name.startswith(CONTEXT_PREFIX)))

class PrefixContext:

    def __init__(self, graph):
        values = graph.features.detach().cpu() if isinstance(graph.features, torch.Tensor) else torch.tensor(graph.features)
        self.rows = {name: tuple(map(float, row)) for (name, row) in zip(graph.identities, values.tolist())}
        self.holders = {}
        edges = graph.edges.tolist() if isinstance(graph.edges, torch.Tensor) else graph.edges
        for (left, right, kind) in edges:
            (a, b) = (graph.identities[left], graph.identities[right])
            if kind == 6 and a.startswith('task:') and b.startswith('worker:'):
                self.holders[a[5:]] = b[7:]
        self.tasks = ()
        self.index = 0
        self.worker = None
        self.receiver = None

    @property
    def task(self):
        return self.tasks[self.index] if self.index < len(self.tasks) else None

    def _point(self, name, offset=22):
        row = self.rows.get(name)
        return None if row is None else (row[offset], row[offset + 1])

    def augment(self, kind, candidates):
        task = self.task if kind not in ('group', 'end') else None
        if task is None:
            return tuple((replace(c, node=names[0] if len(names) == 1 else names or None) for c in candidates for names in (_without_context(c.node),)))
        row = self.rows.get('task:' + task)
        if row is None:
            raise DecodeFailure('current_task_missing_from_source_graph')
        origin = (row[20], row[21])
        delivery = (row[22], row[23])
        anchor = self._point('worker:' + self.worker, 20) if self.worker is not None else origin
        service = delivery if kind == 'delivery_gap' else origin if kind in ('pickup_worker', 'pickup_gap') else anchor or origin
        output = []
        for candidate in candidates:
            base = list(candidate.features[:4]) + [0.0] * (16 - min(4, len(candidate.features)))
            base[4:10] = [row[4], row[6], row[7], row[8], row[5], row[9]]
            names = _without_context(candidate.node)
            point = None
            if kind in ('pickup_worker', 'receiver') and candidate.value is not None:
                point = self._point('worker:' + candidate.value)
            elif names:
                point = next((self._point(name) for name in names if self._point(name) is not None), None)
            elif kind == 'meeting' and candidate.value is None:
                point = anchor
            if point is not None:
                (dx, dy) = (point[0] - service[0], point[1] - service[1])
                (ex, ey) = (point[0] - delivery[0], point[1] - delivery[1])
                base[10:16] = [dx, dy, math.log1p(math.hypot(dx, dy)), ex, ey, math.log1p(math.hypot(ex, ey))]
            output.append(replace(candidate, features=tuple(base), node=names + (CONTEXT_PREFIX + task,)))
        return tuple(output)

    def consume(self, kind, selected):
        if kind == 'group':
            self.tasks = tuple(sorted(selected.value))
            self.index = 0
            self.worker = self.holders.get(self.task)
            self.receiver = None
        elif kind == 'pickup_worker':
            self.worker = selected.value
        elif kind == 'receiver':
            self.receiver = selected.value
        elif kind == 'meeting' and selected.value is not None:
            self.worker = self.receiver
        elif kind == 'receiver_gap':
            self.worker = self.receiver
        elif kind == 'delivery_gap':
            self.index += 1
            self.worker = self.holders.get(self.task)
            self.receiver = None

def prepare_macro(macro):
    context = PrefixContext(macro.graph)
    slots = []
    for slot in macro.slots:
        fields = []
        for decision in slot.fields:
            candidates = context.augment(decision.kind, decision.candidates)
            fields.append(replace(decision, candidates=candidates))
            context.consume(decision.kind, candidates[decision.selected])
        slots.append(replace(slot, fields=tuple(fields)))
    return replace(macro, slots=tuple(slots))

class StructurePolicy(nn.Module):

    def __init__(self, hidden=128, *, proposal_slots=3):
        super().__init__()
        if hidden != 128:
            raise ValueError('frozen TRAIL hidden size is 128')
        self.node_input = nn.Linear(NODE_FEATURE_DIM, hidden)
        self.layers = nn.ModuleList((RelationLayer(hidden) for _ in range(2)))
        self.choice_input = nn.Linear(FEATURE_DIM, hidden)
        self.field_type = nn.Embedding(len(FIELD_TYPES), 16)
        self.query = nn.Linear(hidden + 16, hidden)
        self.initial = nn.Linear(hidden + 6, hidden)
        self.gru = nn.GRUCell(hidden + 16, hidden)
        self.value_head = nn.Sequential(nn.Linear(hidden + 6, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer('node_mean', torch.zeros(NODE_FEATURE_DIM))
        self.register_buffer('node_std', torch.ones(NODE_FEATURE_DIM))
        self.register_buffer('choice_mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('choice_std', torch.ones(FEATURE_DIM))
        self.register_buffer('normalization_fitted', torch.tensor(False))
        self.update_step = 0
        if type(proposal_slots) is not int or not 1 <= proposal_slots <= 3:
            raise ValueError('At most three complete structure proposals are supported')
        self.hidden = hidden
        self.proposal_slots = proposal_slots
        self.task_query = nn.Linear(hidden, hidden, bias=False)

    @property
    def device(self):
        return self.node_mean.device

    def fit_normalization(self, training_macros):
        training_macros = tuple((prepare_macro(macro) for macro in training_macros))
        if self.normalization_fitted.item():
            raise ValueError('normalization is already frozen')
        with torch.no_grad():
            for (kind, mean, std) in (('nodes', self.node_mean, self.node_std), ('choices', self.choice_mean, self.choice_std)):
                count = 0
                total = torch.zeros_like(mean)
                squared = torch.zeros_like(total)
                for macro in training_macros:
                    values = macro.graph.features if kind == 'nodes' else [c.features for slot in macro.slots for decision in slot.fields for c in decision.candidates]
                    tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1, mean.numel())
                    count += len(tensor)
                    total += tensor.sum(0)
                    squared += tensor.square().sum(0)
                if count == 0:
                    raise ValueError('empty training features')
                mean.copy_(total / count)
                std.copy_((squared / count - mean.square()).clamp_min(0).sqrt().clamp_min(1.0))
            self.normalization_fitted.fill_(True)

    def encode(self, graph):
        values = torch.as_tensor(graph.features, dtype=torch.float32, device=self.device)
        if values.ndim != 2 or values.shape[1] != NODE_FEATURE_DIM:
            raise ValueError('Invalid relation graph feature dimensions')
        x = torch.relu(self.node_input((values - self.node_mean) / self.node_std))
        edges = torch.as_tensor(graph.edges, dtype=torch.long, device=self.device).reshape(-1, 3)
        for layer in self.layers:
            x = layer(x, edges)
        pooled = torch.cat((x.mean(0), torch.tensor(graph.progress, dtype=torch.float32, device=self.device)))
        hidden = torch.tanh(self.initial(pooled))
        return (x, hidden, self.value_head(pooled).squeeze(-1), {name: i for (i, name) in enumerate(graph.identities)})

    def distribution(self, kind, candidates, hidden, nodes, node_index):
        tasks = {name[len(CONTEXT_PREFIX):] for c in candidates for name in _names(c.node) if name.startswith(CONTEXT_PREFIX)}
        if kind not in ('group', 'end'):
            if len(tasks) != 1:
                raise ValueError('A task-dependent policy field requires one causal current-task context')
            task = next(iter(tasks))
            key = 'task:' + task
            if key not in node_index:
                raise ValueError('Current-task context is outside the source graph')
            hidden = torch.tanh(hidden + self.task_query(nodes[node_index[key]]))
        elif tasks:
            raise ValueError('A global field cannot inherit another task context')
        values = torch.tensor([c.features for c in candidates], dtype=torch.float32, device=self.device)
        embeddings = torch.relu(self.choice_input((values - self.choice_mean) / self.choice_std))
        entity_rows = []
        for candidate in candidates:
            names = candidate.node if isinstance(candidate.node, tuple) else (candidate.node,)
            entities = [node_index[name] for name in names if name in node_index]
            entity_rows.append(nodes[entities].mean(0) if entities else torch.zeros(128, device=self.device))
        embeddings = embeddings + torch.stack(entity_rows)
        field = self.field_type(torch.tensor(FIELD_TYPES.index(kind), device=self.device))
        query = self.query(torch.cat((hidden, field)))
        logits = embeddings @ query / math.sqrt(128)
        mask = torch.tensor([c.legal for c in candidates], dtype=torch.bool, device=self.device)
        if not mask.any():
            raise DecodeFailure('all_masked:' + kind)
        distribution = Categorical(logits=logits.masked_fill(~mask, float('-inf')))
        return (distribution, embeddings, field)

    def generate(self, obs, deterministic=False):
        graph = codec.graph_observation(obs)
        context = PrefixContext(graph)
        with torch.no_grad():
            (nodes, hidden, value, node_index) = self.encode(graph)
            slots = []
            total_log = 0.0
            seen = {codec.structure_key(obs.plan)}
            for _ in range(self.proposal_slots):
                fields = []

                def choose(kind, raw_candidates):
                    nonlocal hidden, total_log
                    candidates = context.augment(kind, raw_candidates)
                    (distribution, embedding, field) = self.distribution(kind, candidates, hidden, nodes, node_index)
                    choice = distribution.logits.argmax() if deterministic else distribution.sample()
                    selected = int(choice)
                    logp = float(distribution.log_prob(choice))
                    fields.append(FieldDecision(kind, candidates, selected, logp, sum((c.legal for c in candidates)) == 1))
                    total_log += logp
                    hidden = self.gru(torch.cat((embedding[selected], field)), hidden)
                    context.consume(kind, candidates[selected])
                    return selected
                try:
                    (script, plan) = codec.decode_structure(obs, choose, validate_structure=False)
                    identity = codec.structure_key(plan)
                    status = 'duplicate' if identity in seen else 'candidate'
                    seen.add(identity)
                    slots.append(ProposalSlot(script, tuple(fields), status, plan))
                except DecodeFailure as error:
                    slots.append(ProposalSlot(None, tuple(fields), 'decode_failure:' + str(error)))
            return MacroAction(obs.source_witness_digest, tuple(slots), graph, total_log, float(value), self.update_step, dict(obs.search_radii), ())

    def evaluate_macro(self, macro):
        if any((slot.status in ('candidate', 'demonstration') and (not slot.fields or slot.fields[0].kind != 'group' or slot.fields[-1].kind != 'end') for slot in macro.slots)):
            raise ValueError('A complete structure field sequence is required')
        macro = prepare_macro(macro)
        (nodes, hidden, value, node_index) = self.encode(macro.graph)
        logs = []
        entropies = []
        for slot in macro.slots:
            for decision in slot.fields:
                (distribution, embedding, field) = self.distribution(decision.kind, decision.candidates, hidden, nodes, node_index)
                selected = torch.tensor(decision.selected, device=self.device)
                logs.append(distribution.log_prob(selected))
                if not decision.forced:
                    entropies.append(distribution.entropy())
                hidden = self.gru(torch.cat((embedding[decision.selected], field)), hidden)
        zero = value * 0
        return (torch.stack(logs).sum() if logs else zero, value, torch.stack(entropies).mean() if entropies else zero)

    def demonstration(self, obs, target_plan, group, *, graph=None):
        return codec.teacher_macro(obs, target_plan, group, graph=graph)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(hidden=self.hidden, proposal_slots=self.proposal_slots, state_dict=self.state_dict(), update_step=self.update_step)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @classmethod
    def load(cls, path, device='cpu'):
        row = torch.load(path, map_location=device, weights_only=True)
        model = cls(row['hidden'], proposal_slots=row['proposal_slots']).to(device)
        model.load_state_dict(row['state_dict'])
        model.update_step = row['update_step']
        return model
