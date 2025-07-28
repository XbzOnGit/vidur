from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


class NetworkChange(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        pipeline_stage: int,
        comm_id: int,
        new_bandwidth_factor: float
    ):
        self._arrived_at = arrived_at
        self._pipeline_stage = pipeline_stage
        self._comm_id = comm_id
        self._new_bandwidth_factor = new_bandwidth_factor
    
    @property
    def arrived_at(self):
        return self._arrived_at
    
    @property
    def pipeline_stage(self):
        return self._pipeline_stage
    
    @property
    def comm_id(self):
        return self._comm_id
    
    @property
    def new_bandwidth_factor(self):
        return self._new_bandwidth_factor