import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description='Train a full service-structure TRAIL policy with H demonstrations and PPO')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--validation-inputs', nargs='*', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--h-config', type=Path)
    p.add_argument('--device', choices=('cpu','cuda'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    from common.io import read_json
    from common.config import TRAIL_DEFAULTS, TFR_DEFAULTS
    from common.data.pipeline import load_case
    from common.recovery.preparation import prepare_recovery_input
    from solvers.tfr.local_solver import LocalConfig
    from .training import TrainingConfig, train
    values = dict(TRAIL_DEFAULTS) if args.config is None else read_json(args.config)
    if args.device is not None:
        values['device'] = args.device
    h_values = dict(TFR_DEFAULTS) if args.h_config is None else read_json(args.h_config)
    h_values['seed'] = args.seed
    def prepared(paths):
        return [prepare_recovery_input(*load_case(path)) for path in paths]
    result = train(prepared(args.inputs),args.output,
        validation_samples=prepared(args.validation_inputs),config=TrainingConfig(**values),
        h_config=LocalConfig(**h_values),seed=args.seed)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
