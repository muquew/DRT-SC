import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True,
        choices=('greedy','alns','grasp','rlns','hu','deco','h','trail'))
    parser.add_argument('--input', required=True, type=Path,
        help='prepared case.json or raw-data preparation recipe')
    parser.add_argument('--config', type=Path, help='native method parameter JSON')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='write one solution JSON; default prints JSON')
    parser.add_argument('--library', type=Path, help='Hu scenario library')
    parser.add_argument('--checkpoint', type=Path, help='TRAIL weights or DECO model directory')
    parser.add_argument('--history', type=Path, help='DECO observed position history')
    parser.add_argument('--python', dest='python_executable', type=Path,
        help='optional DECO TensorFlow interpreter')
    args = parser.parse_args(argv)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    from common.io import read_json, write_json
    from common.recovery.observation import cpu_time
    from common.data.pipeline import load_case
    from common.runner import solve
    started, cpu_started = perf_counter(), cpu_time()
    item, routing = load_case(args.input)
    parameters = {key: str(value.resolve()) for key, value in (
        ('library_path',args.library),('checkpoint_path',args.checkpoint),
        ('history_path',args.history),('python_executable',args.python_executable)) if value is not None}
    if args.method == 'deco' and args.history is None:
        history = read_json(args.input).get('history_file')
        if history is not None:
            parameters['history_path'] = str((args.input.resolve().parent / history).resolve())
    result = solve(item, args.method, routing, seed=args.seed,
        config=None if args.config is None else read_json(args.config), parameters=parameters,
        origin=started, cpu_origin=cpu_started)
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        write_json(args.output, result)
        print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
