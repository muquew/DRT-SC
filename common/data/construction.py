from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True, kw_only=True)
class ConstructionConfig:
    task_count: int
    worker_count: int
    unit_load: float = 1.
    worker_capacity: float = 4.
    load_unit: str = 'requests'
    worker_available_time_s: float = 0.
    worker_speed_factor: float = 1.
    pickup_service_time_s: float = 0.
    delivery_service_time_s: float = 0.
    customer_ready_time_s: float | None = None
    max_snap_distance_m: float = 100.
    task_seed: int = 0
    worker_seed: int = 0
    worker_origin: str = 'source'
    source_depot_id: str | None = None
    observation_time_s: float | None = None
    sampling: str = 'full_city'
    sampling_radius_m: float | None = None

    def __post_init__(self):
        for name in ('task_count', 'worker_count'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name+' must be a positive integer')
        for name in ('task_seed', 'worker_seed'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name+' must be an integer')
        positive = ('unit_load','worker_capacity','worker_speed_factor')
        nonnegative = ('worker_available_time_s','pickup_service_time_s','delivery_service_time_s','max_snap_distance_m')
        optional=('customer_ready_time_s','observation_time_s','sampling_radius_m')
        for name in positive+nonnegative+optional:
            value=getattr(self,name)
            if value is None and name in optional:continue
            if type(value)not in(int,float):raise ValueError(name+' must be finite')
            try:value=float(value)
            except OverflowError as error:raise ValueError(name+' exceeds the finite numeric range')from error
            if not math.isfinite(value):raise ValueError(name+' must be finite')
            if name in positive and value<=0 or name in nonnegative and value<0:
                raise ValueError(name+' has an invalid sign')
            object.__setattr__(self,name,0. if value==0. else value)
        if not isinstance(self.load_unit,str) or not self.load_unit.strip():
            raise ValueError('load_unit must name the shared task/capacity unit')
        if self.worker_origin not in ('source','observed_positions','source_depot'):
            raise ValueError('unknown worker_origin')
        if self.source_depot_id is not None and (not isinstance(self.source_depot_id,str)or not self.source_depot_id.strip()):
            raise ValueError('source_depot_id must name an original source or be None')
        if self.sampling not in ('full_city','local_district','work_district'):
            raise ValueError('unknown source sampling strategy')
        if self.sampling=='full_city' and self.sampling_radius_m is not None:
            raise ValueError('full_city does not use a sampling radius')
        if self.sampling!='full_city' and (type(self.sampling_radius_m) not in (int,float)
                or not math.isfinite(self.sampling_radius_m) or self.sampling_radius_m<=0):
            raise ValueError('local sampling requires a positive sampling_radius_m')
        from common.initial_sequences.inputs import _exceeds_capacity
        if _exceeds_capacity(self.unit_load,self.worker_capacity):
            raise ValueError('unit_load exceeds every worker capacity in this uniform construction')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record,dict):raise ValueError('construction configuration must be an object')
        unknown=set(record)-{f.name for f in fields(cls)}
        if unknown:raise ValueError('unknown construction parameters: '+', '.join(sorted(unknown)))
        return cls(**record)
