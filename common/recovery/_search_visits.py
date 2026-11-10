from dataclasses import dataclass
from typing import Literal

from common.data.roads.schema import RoadPosition
from common.initial_sequences.inputs import _identifier
from common.recovery.plan import RecoveryVisit


@dataclass(frozen=True, kw_only=True)
class TransferStop(RecoveryVisit):


    operation: Literal['transfer_stop'] = 'transfer_stop'
    location: RoadPosition
    stop_id: str
    created_iteration: int = 0

    def __post_init__(self):
        if self.operation != 'transfer_stop':
            raise ValueError('transfer_stop.operation: expected transfer_stop')
        if self.task_id is not None or self.transfer_id is not None:
            raise ValueError('transfer_stop: task_id and transfer_id must be None')
        if not isinstance(self.location, RoadPosition):
            raise ValueError('transfer_stop.location: expected RoadPosition')
        _identifier(self.stop_id, 'transfer_stop.stop_id')
        if type(self.created_iteration) is not int or self.created_iteration < 0:
            raise ValueError('transfer_stop.created_iteration: expected a nonnegative integer')
