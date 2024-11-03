from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)

class ComputationDevice(BaseEntity):
    def __init__(self) -> None:
        super().__init__()
        self._id = ComputationDevice.generate_id()
        self._last_time_in_use = 0.0

    def compute(self, compute_time: float, launch_time: float) -> Tuple[float, float]:
        if launch_time < self._last_time_in_use:
            launch_time = self._last_time_in_use
        self._last_time_in_use = launch_time + compute_time
        return launch_time, compute_time
    
    @property
    def last_time_in_use(self):
        return self._last_time_in_use
