from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
import atexit

logger = init_logger(__name__)

all_requests_arrived = []
def print_requests_statistics():
    total_cnt = len(all_requests_arrived)
    hit_cnt = 0
    miss_cnt = 0
    total_hit_len = 0
    for req in all_requests_arrived:
        kv_hit_length = req.kv_cache_hit_length
        total_hit_len += kv_hit_length
        if kv_hit_length > 0:
            hit_cnt += 1
        else:
            miss_cnt += 1
    logger.info(f"request_cnt: {total_cnt}")
    logger.info(f"hit_cnt: {hit_cnt}")
    logger.info(f"total hit length: {total_hit_len}")
    logger.info(f"miss_cnt: {miss_cnt}")
    
    if hit_cnt > 0:
        logger.info(f"average hit length: {total_hit_len / hit_cnt}")
    else:
        logger.info(f"average hit length: 0")

atexit.register(print_requests_statistics)

class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request
        all_requests_arrived.append(request)
        
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request": self._request.id,
        }

