from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


class Channel(BaseEntity):
    def __init__(self) -> None:
        super().__init__()
        self._id = Channel.generate_id()
        self._last_time_in_use = 0.0
    # Just input bytes.
    # Assuming always launching in correct time order, and no preemption, no scheduling.
    # NOTE: Only model conguestion, pass throughput as parameter.
    # For modeling contention like disk and CPU fetch to GPU directly at the same time.
    def transmit(self, byte_number: int, launch_time: float, thput: float) -> Tuple[float, float]:
        if launch_time < self._last_time_in_use:
            launch_time = self._last_time_in_use
        transmit_time = byte_number / thput
        self._last_time_in_use = launch_time + transmit_time
        return launch_time, transmit_time
    @property
    def last_time_in_use(self):
        return self._last_time_in_use
