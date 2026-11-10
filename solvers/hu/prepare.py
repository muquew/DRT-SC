from pathlib import Path
import argparse
from common.io import read_json, write_json
from common.data.roads.io import load_network
from common.execution.state import ExecutionSnapshot
from common.recovery.inputs import RecoverySettings
from solvers.hu.support import prepare_support_library


def prepare_loaded(snapshot, roads, settings, specs, config, output, *,
                   total_iterations=None, candidate_count=None, workers=1):
    parameters = dict(config.get('parameters', {}))
    parameters.update(config.get('offline_overrides', {}))
    parameters.setdefault('seed', config.get('seed', 0))
    if total_iterations is not None:
        parameters['total_iterations'] = total_iterations
    if candidate_count is not None:
        parameters['candidate_count'] = candidate_count
    if workers == 1:
        library = prepare_support_library(snapshot, roads, settings, specs, **parameters)
    else:
        from solvers.hu.parallel import prepare_parallel_loaded
        library = prepare_parallel_loaded(snapshot, roads, settings, specs, workers=workers, **parameters)
    write_json(output, library)
    return library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--total-iterations', type=int)
    parser.add_argument('--candidate-count', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    root = args.config.resolve().parent
    resolve = lambda path: (root / path).resolve()
    roads = load_network(resolve(config['roads']))
    record = read_json(resolve(config['snapshot']))
    snapshot = ExecutionSnapshot.from_dict(record.get('execution_snapshot', record))
    settings = config['settings']
    if isinstance(settings, str):
        settings = read_json(resolve(settings))
    settings = dict(settings)
    if 'handover_range_m' in settings:
        radius = settings.pop('handover_range_m')
        settings['handover_ranges_m'] = {w.worker_id: radius for w in snapshot.workers}
    settings = RecoverySettings.from_dict(settings)
    specs = config['scenarios']
    if isinstance(specs, str):
        specs = read_json(resolve(specs))
    library = prepare_loaded(snapshot, roads, settings, specs, config, args.output,
        total_iterations=args.total_iterations, candidate_count=args.candidate_count, workers=args.workers)
    print('Prepared', len(library['scenarios']), 'Hu scenarios')


if __name__ == '__main__':
    main()
