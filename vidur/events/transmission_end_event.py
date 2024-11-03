from typing import List, Tuple
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.entities.kvitem import StorageInfo, KVObjectMetadata


logger = init_logger(__name__)

class TransmissionEndEvent(BaseEvent):
    def __init__(self, time: float, item_list: List[KVObjectMetadata]):
        # Make item_list KV cache READY.
        super().__init__(time, EventType.TRANSMISSION_END)
        # TODO: Pass channel here and mark in use or not.
        self._item_list = item_list
    def append_item(self, item: KVObjectMetadata):
        self._item_list.append(item)
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        # TODO: Mark channel as free, if channel has that state.
        for kv_obj in self._item_list:
            assert isinstance(kv_obj, KVObjectMetadata)
            storage_info: StorageInfo = kv_obj.storage_info
            assert storage_info is not None, f"storage info is None on kv_obj {kv_obj._id}"
            assert kv_obj.associated_event == self
            assert storage_info.mark_ready(kv_obj)
        return []
    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "item_list": self._item_list,
        }

