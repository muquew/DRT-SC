from copy import deepcopy
from functools import lru_cache
import os
from common.io import read_json


BASELINE_DEFAULTS = {'defaults': {'max_evaluations': None,
              'worker_candidates': None,
              'meeting_candidates': 8,
              'transfer_policy': 'source',
              'meeting_pool': 'legacy',
              'objective_zero_stopping': False,
              'parameters': {}},
 'methods': {'greedy': {'iterations': 1},
             'alns': {'iterations': 5,
                      'parameters': {'removal_fraction': 0.4,
                                     'segment_size': 1,
                                     'reaction_factor': 0.1,
                                     'cooling_rate': 0.6309573444801932}},
             'grasp': {'iterations': 2,
                       'parameters': {'alpha': 0.2,
                                      'swap_passes': 1,
                                      'local_actions': 8,
                                      'rcl_pool_limit': 32}},
             'rlns': {'iterations': None,
                      'parameters': {'protocol': 'paper',
                                     'paper_restarts': 1,
                                     'paper_no_improve': 5,
                                     'remove_min': 5,
                                     'remove_max': 8,
                                     'history_length': 3,
                                     'blink_probability': 0.05,
                                     'max_rounds': 5}},
             'deco': {'iterations': 1,
                      'parameters': {'encounter_radius_m': 10.0,
                                     'min_remaining_time_s': 1200.0,
                                     'max_assigned_tasks': 6,
                                     'device': 'cpu',
                                     'reward_mode': 'total_tardiness'}},
             'hu': {'iterations': None,
                    'parameters': {'online_local_passes': 3, 'online_local_actions': 16}}}}

CACHE_DEFAULTS = {'routing_compat': {'cache_size': 128,
                    'source_cache_size': 0,
                    'source_cache_node_limit': 10000000,
                    'source_cache_byte_limit': 67108864},
 'routing_formal': {'cache_size': 65536,
                    'source_cache_size': 4096,
                    'source_cache_node_limit': 2000000,
                    'source_cache_byte_limit': 67108864},
 'routing_training': {'cache_size': 4096,
                      'source_cache_size': 4096,
                      'source_cache_node_limit': 2000000,
                      'source_cache_byte_limit': 67108864},
 'evaluation': {'route_max_entries': 20000,
                'route_max_visits': 200000,
                'component_max_entries': 20000,
                'component_max_visits': 200000,
                'path_max_entries': 20000,
                'path_max_legs': 2000000},
 'h_physical': {'max_entries': 8192, 'max_bytes': 134217728},
 'h_publication': {'max_entries': 8192, 'max_bytes': 134217728},
 'h_geometry': {'index_entries': 16,
                'index_bytes': 167772160,
                'domain_entries': 32,
                'domain_bytes': 67108864,
                'scalar_entries': 512,
                'scalar_bytes': 33554432},
 'h_compiled': {'max_entries': 2048, 'max_legs': 200000, 'input_buffer_bytes': 134217728},
 'exact_roads': {'source_entries': 128},
 'spatial_search': {'max_entries': 16, 'max_nodes': 50000},
 'h_archive': {'travel_entries': 256, 'audit_entries': 256},
 'h_structure': {'visit_key_entries': 8192,
                 'route_key_entries': 4096,
                 'route_key_visits': 65536,
                 'potential_entries': 8192,
                 'travel_entries': 32768,
                 'estimate_entries': 32,
                 'witness_path_entries': 8192,
                 'shortlist_entries': 8,
                 'shortlist_soft_nodes': 200000},
 'h_temporal': {'point_entries': 8192,
                'reachability_entries': 8192,
                'slot_entries': 256,
                'closure_entries': 64,
                'closure_nodes': 50000},
 'memo': {'worker_entries': 4096,
          'worker_cargo_items': 128,
          'projection_entries': 64,
          'meeting_entries': 512}}

DISRUPTION_DEFAULTS = {'events': [{'kind': 'pickup_delay', 'delay_s': 900.0, 'seed_offset': 0},
            {'kind': 'reception_delay', 'delay_s': 900.0, 'seed_offset': 1},
            {'kind': 'delivery_relocation',
             'relocation_band_m': [1000.0, 3000.0],
             'seed_offset': 2},
            {'kind': 'worker_withdrawal', 'seed_offset': 3},
            {'kind': 'road_time_change', 'time_factor': 2.0, 'seed_offset': 4}],
 'recovery': {'handover_range_m': 500.0, 'transfer_duration_s': 30.0},
 'mix': {'count': 8,
         'weights': {'pickup_delay': 1,
                     'reception_delay': 1,
                     'delivery_relocation': 2,
                     'worker_withdrawal': 2,
                     'road_time_change': 2}}}

HU_DEFAULTS = {'total_iterations_per_scenario': 500, 'candidate_count': 3}

TFR_DEFAULTS = {'max_rounds': 50,
 'patience': 10,
 'proposal_slots': 3,
 'geometry_nodes': 64,
 'lp_calls': 16,
 'lp_maxiter': 10000,
 'seed': 0,
 'ablation': 'full'}

TRAIL_DEFAULTS = {'device': 'cuda',
 'bc_limit': 512,
 'bc_epochs': 5,
 'batch_size': 16,
 'bc_learning_rate': 0.0003,
 'transitions': 4096,
 'startup_limit': 512,
 'rollout_size': 256,
 'ppo_epochs': 4,
 'ppo_batch_size': 16,
 'learning_rate': 3e-05,
 'clip': 0.2,
 'gae_lambda': 0.95,
 'value_coefficient': 0.5,
 'entropy_coefficient': 0.01,
 'max_gradient_norm': 0.5,
 'target_kl': 0.02,
 'anchor_coefficient': 0.1}

SERVICE_DEFAULTS = {'decision_window': 4,
 'structure_window': 4,
 'active_geometry_limit': 2,
 'pool_max_work_streak': 2,
 'geometry_service_work_units': 65536}

SEARCH_DEFAULTS = {'search_radius_m': 3000}


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def _resolve(environ):
    profiles = deepcopy(CACHE_DEFAULTS)
    sources = {name: {key: 'default' for key in values} for name, values in profiles.items()}
    override = environ.get('DRT_CACHE_CONFIG')
    if override:
        for name, values in read_json(override)['profiles'].items():
            if name not in profiles or set(values) - set(profiles[name]):
                raise ValueError('unknown cache profile or field: ' + name)
            profiles[name].update(values)
            sources[name].update({key: str(override) for key in values})
    for name, values in profiles.items():
        for key, value in values.items():
            _integer(value, name+'.'+key)
    return dict(profiles=profiles, sources=sources)


@lru_cache(maxsize=1)
def _snapshot():
    return _resolve(os.environ)


def cache_record(name, overrides=None, *, environ=None):
    document = _snapshot() if environ is None else _resolve(environ)
    values, sources = dict(document['profiles'][name]), dict(document['sources'][name])
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError('unknown cache capacity: ' + name+'.'+key)
        if value is not None:
            values[key] = _integer(value, key)
            sources[key] = 'argument'
    return dict(profile=name, effective=values, sources=sources)


def cache_limits(name, **overrides):
    return cache_record(name, overrides)['effective']


def routing_options(purpose='formal', overrides=None):
    return {'cache_profile': purpose, 'search_method': 'scipy', **(overrides or {})}


def service_value(name):
    return SERVICE_DEFAULTS[name]
