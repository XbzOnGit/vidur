from typing import List

from vidur.entities import NetworkChange
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class NetworkChangeEvent(BaseEvent):
    # def __init__(self, time: float, pipeline_stage: int, comm_id: int, new_bandwidth_factor: float):
    def __init__(self, time: float, net_c: NetworkChange):
        super().__init__(time, EventType.NETWORK_CHANGE)
        # Currently must be > 0.0.
        assert net_c.new_bandwidth_factor > 0.0
        self._net_c = net_c
        
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        execution_time_predictor = scheduler.get_execution_time_predictor()
        execution_time_predictor.network_bandwidth_factor_assign(self._net_c._pipeline_stage, 
                                                                 self._net_c._comm_id, 
                                                                 self._net_c._new_bandwidth_factor)
        return []
