from typing import List, Tuple
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


logger = init_logger(__name__)


class TransformEndEvent(BaseEvent):
    def __init__(self, time: float, item_list):
        # Make item_list KV cache READY.
        super().__init__(time, EventType.TRANSFORM_END)

        # TODO: Pass computation device here and mark in use or not.
        self._item_list = item_list
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        # TODO: Mark computation device as free, if computation device has that state.
        # NOTE: Currently also ARRIVING in transform.
        for item_tuple in self._item_list:
            item, compression_level = item_tuple
            item.mark_ready(compression_level)
        return []
    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "item_list": self._item_list,
        }


