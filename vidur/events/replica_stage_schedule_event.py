from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int):
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)

        self._replica_id = replica_id
        self._stage_id = stage_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        stage_scheduler = scheduler._replica_schedulers[
            self._replica_id
        ]._replica_stage_schedulers[self._stage_id]

        self._batch, self._batch_stage, execution_time, \
            start_fir_time, end_exec_time, new_full_blocks_list = stage_scheduler.on_schedule(self.time)
        
        if not (self._batch and self._batch_stage):
            return []
        self._batch_stage.on_schedule(self.time)
        metrics_store.on_replica_stage_schedule(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
        )

        self._is_last_stage = stage_scheduler.is_last_stage
        # print(f"end_execution_time: {end_exec_time}")
        # print(f"self.time + self._batch_stage.execution_time: {self.time + self._batch_stage.execution_time}")
        # print(f"Recorded time to batch stage {self._batch_stage.id} end takes {self._batch_stage.execution_time}")
        # Float point error, replace self.time + self._batch_stage.execution_time with end_exec_time
        # But then it will cause a later assertion to fail --> time == self._scheduled_at + self._execution_time.
        # So change that assertion to allow for a small error.
        return [
            BatchStageEndEvent(
                end_exec_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
