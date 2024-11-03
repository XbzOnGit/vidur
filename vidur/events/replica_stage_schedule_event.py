from typing import List

from vidur.events import BaseEvent
from vidur.events.compute_end_event import ComputeEndEvent
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
        # Retrieve happens here.
        self._batch, self._batch_stage, execution_time, after_retrieve_time = stage_scheduler.on_schedule(self.time)
        self._time = after_retrieve_time
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

        # Ocupy GPU device.
        # And wait for it to be done.
        # This is blocking.
        launch_time, compute_time = stage_scheduler.gpu_compute_device.compute(self._batch_stage.execution_time, self.time)
        compute_end_time = launch_time + compute_time
        compute_end_event = ComputeEndEvent(compute_end_time, [], None)
        global_simulator = self.simulator
        global_simulator.add_events([compute_end_event])
        block_time = global_simulator.loop_until(compute_end_event)
        return [
            BatchStageEndEvent(
                block_time,
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
