import torch
from solvers.tfr.local_solver import LocalConfig, NORMAL, _local_result
from .environment import StructureEnvironment, settle_environment
from .network import StructurePolicy

def solve_policy(item, routing, policy, config=None, *, training=False, origin=None):
    env = StructureEnvironment(item, routing, config=config or LocalConfig(), training=training, origin=origin)
    obs = env.reset()
    while obs is not None:
        if settle_environment(env):
            break
        if env.status is not None:
            break
        obs = env.begin_decision()
        macro = policy.generate(obs, deterministic=not training)
        (obs, _, done, info) = env.step_policy(macro)
        if not info.get('closed') or done:
            break
    if env.status in NORMAL:
        env.finalize()
    result = _local_result(env)
    result.diagnostics.update(method='trail', full_structure_decoding=True, policy=env.policy_diagnostics())
    return result

def solve(item, routing, checkpoint_path, config=None, seed=0, *, origin=None):
    torch.manual_seed(seed)
    policy = StructurePolicy.load(checkpoint_path)
    policy.eval()
    return solve_policy(item, routing, policy, config, origin=origin)
