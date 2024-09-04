from typing import List, Union, Tuple
import random
import atexit
from collections import deque

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


class Channel(BaseEntity):
    def __init__(self, thput: float, space_per_token_per_layer: int):
        self._thput = thput
        self._last_time_in_use = 0.0
        self._space_per_token_per_layer = space_per_token_per_layer
    def transmit(self, token_number, current_time, num_layers) -> Tuple[float, float, float]:
        #print(f"num_layers: {num_layers}")
        #print(f"token_number: {token_number}")
        #print(f"current_time: {current_time}")
        per_layer_size = token_number * self._space_per_token_per_layer
        per_layer_time = per_layer_size / self._thput
        if current_time >= self._last_time_in_use:
            self._last_time_in_use = current_time + per_layer_time * num_layers
            first_layer_time = current_time + per_layer_time
            #print(f"returns 1: {self._last_time_in_use}, {first_layer_time}, {per_layer_time}")
            return self._last_time_in_use, first_layer_time, per_layer_time
        else:
            original_last_time_in_use = self._last_time_in_use
            self._last_time_in_use += per_layer_time * num_layers
            first_layer_time = original_last_time_in_use + per_layer_time
            #print(f"returns 2: {self._last_time_in_use}, {first_layer_time}, {per_layer_time}")
            return self._last_time_in_use, first_layer_time, per_layer_time
    def get_per_layer_time(self, token_number):
        per_layer_size = token_number * self._space_per_token_per_layer
        per_layer_time = per_layer_size / self._thput
        return per_layer_time

    @property
    def last_time_in_use(self):
        return self._last_time_in_use