from typing import List, Optional
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.entities.kvitem import KVObjectMetadata


logger = init_logger(__name__)


# Possible future use on pin and unpin.
# Currently this is always blocking.
class ComputeEndEvent(BaseEvent):
    def __init__(self, time: float, transform_ready_list: List[KVObjectMetadata], batch_stage_end_event: Optional[BaseEvent]):
        super().__init__(time, EventType.GPU_COMPUTE_END)
        self._transform_ready_list = transform_ready_list
        self._batch_stage_end_event = batch_stage_end_event
    def append_item(self, kv_obj: KVObjectMetadata):
        self._transform_ready_list.append(kv_obj)
    def handle_event(
            self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        if self._batch_stage_end_event is not None:
            # Currently not using this, just block the time.
            assert len(self._transform_ready_list) == 0
            return self._batch_stage_end_event.handle_event(scheduler, metrics_store)
        else:
            for kv_obj in self._transform_ready_list:
                kv_obj.storage_info.mark_ready(kv_obj)
            return []
                



