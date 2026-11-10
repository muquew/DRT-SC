from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from solvers.hu.support import prepare_support_library, assemble_library, validate_preparation_parameters

_CONTEXT = None


def _scene(index):
    snapshot, roads, settings, specs, parameters = _CONTEXT
    library = prepare_support_library(snapshot, roads, settings, (specs[index],),
        scenario_indices=(index,), **parameters)
    return library['scenarios'][0]


def prepare_parallel_loaded(snapshot, roads, settings, specs, *, workers=2, **parameters):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    specs = tuple(specs)
    if not specs:
        raise ValueError('Hu preparation requires hypothetical event scenes')
    if workers == 1:
        return prepare_support_library(snapshot, roads, settings, specs, **parameters)
    resolved = validate_preparation_parameters({name:value for name,value in parameters.items()
        if name != 'routing_options'})
    global _CONTEXT

    _CONTEXT = snapshot, roads, settings, specs, parameters
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            scenes = list(pool.map(_scene, range(len(specs))))
        return assemble_library(snapshot, roads, settings, scenes, resolved)
    finally:
        _CONTEXT = None
