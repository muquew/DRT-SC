from solvers.tfr.interfaces.boundary import FrozenSnapshotWindow; from solvers.tfr.interfaces.boundary import build_boundary; from solvers.tfr.interfaces.boundary import project_input
from solvers.tfr.interfaces.replay import ReplayBoundaryPolicy; from solvers.tfr.interfaces.replay import replay_action_contract


TIME_SEMANTICS = 'offline_frozen_snapshot'


def prepare_frozen_snapshot_boundary(item, routing, *, policy, withdrawal_deadlines):


    if isinstance(policy, dict):
        policy = ReplayBoundaryPolicy(**policy)
    window = FrozenSnapshotWindow(item.time_s)
    orders, blocks, holding = replay_action_contract(item, routing, window, policy,
        withdrawal_deadlines=withdrawal_deadlines)
    boundary = build_boundary(item, orders, blocks, window, holding, routing,
        withdrawal_deadlines=withdrawal_deadlines)
    return project_input(item, boundary)
