import json
import os
import sys
import argparse
from time import perf_counter, process_time
from solvers.deco.policy import DecoComponent, DecoState


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('seed',type=int,nargs='?')
    parser.add_argument('--device',choices=('cpu','auto','gpu'),default=os.environ.get('DRT_SC_DECO_DEVICE','cpu'))
    arguments=parser.parse_args()
    started, cpu = perf_counter(), process_time()
    model = DecoComponent.load(arguments.checkpoint, device=arguments.device)
    if arguments.seed is not None:
        model.rng = model.np.random.RandomState(arguments.seed)
    print(json.dumps(dict(status='ready', wall_s=perf_counter()-started, cpu_s=process_time()-cpu,
                          training=model.metadata())), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('close'):
                break
            started, cpu = perf_counter(), process_time()
            try:
                state = DecoState(**request['state'])
                result = model.choose_action(state)
                result.update(status='ok', wall_s=perf_counter()-started, cpu_s=process_time()-cpu)
            except Exception as error:
                result = dict(status='error', error=f'{type(error).__name__}: {error}',
                              wall_s=perf_counter()-started, cpu_s=process_time()-cpu)
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
