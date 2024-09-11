from typing import List, Union, Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


class CPU(BaseEntity):
    def __init__(self):
        self._last_time_in_use = 0.0
    def run_task(self, task_time, current_time) -> float:
        if current_time >= self._last_time_in_use:
            self._last_time_in_use = current_time + task_time
            return self._last_time_in_use
        else:
            self._last_time_in_use += task_time
            return self._last_time_in_use
    @property
    def last_time_in_use(self):
        return self._last_time_in_use