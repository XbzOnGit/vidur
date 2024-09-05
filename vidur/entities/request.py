from typing import Tuple, List, Optional

from vidur.entities.base_entity import BaseEntity
from vidur.entities.kv_block_trie import KVBlock, get_general_kv_block_size
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
        tokens: Optional[List[str]] = None,
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = num_processed_tokens
        self.tokens = tokens

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0
        self._kv_cache_hit_length = 0
        self._kv_fetch_time = 0.0
        self._kv_insert_time = 0.0
        self._kv_inserted = False
        self._full_kvblocks = None

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)
    
    @property
    def full_kvblocks(self) -> List[KVBlock]:
        if self._full_kvblocks is None:
            self._full_kvblocks = []
            total_tokens = self._num_prefill_tokens + self._num_decode_tokens
            block_size = get_general_kv_block_size()
            assert block_size > 0
            full_kv_block_cnt = total_tokens // block_size
            for i in range(full_kv_block_cnt):
                start_idx = i * block_size
                end_idx = (i + 1) * block_size
                kvblock = KVBlock(block_size)
                kvblock.set_tokens(self.tokens[start_idx:end_idx])
                self._full_kvblocks.append(kvblock)
        return self._full_kvblocks

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1
    
    @property
    def kv_cache_hit_length(self) -> bool:
        return self._kv_cache_hit_length
    
    @property
    def kv_fetch_time(self) -> float:
        return self._kv_fetch_time
    
    @property
    def kv_insert_time(self) -> float:
        return self._kv_insert_time
    
    @property
    def kv_inserted(self) -> bool:
        return self._kv_inserted
    
    def set_replica_scheduler(self, replica_scheduler):
        self._replica_scheduler = replica_scheduler

    # For KV cache.
    def set_num_processed_tokens(self, num_processed_tokens: int) -> None:
        self._num_processed_tokens = num_processed_tokens

    def set_prefill_complete(self, timestamp):
        self._is_prefill_complete = True
        if self._prefill_completed_at == 0:
            self._prefill_completed_at = timestamp
    
    def reset_prefill_complete(self):
        self._is_prefill_complete = False
        self._prefill_completed_at = 0

    def set_kv_cache_hit_length(self, set_len):
        # print(f"Request {self._id} set kv cache hit length to {set_len}")
        self._kv_cache_hit_length = set_len

    def add_kv_fetch_time(self, fetch_time):
        self._kv_fetch_time += fetch_time

    def add_kv_insert_time(self, insert_time):
        self._kv_insert_time += insert_time
    
    def set_kv_inserted(self, inserted):
        self._kv_inserted = inserted

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return

        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        # print(f"Request {self._id} from {self._num_processed_tokens} to {self._num_processed_tokens + num_tokens_processed}")
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time

        assert self._num_processed_tokens <= self.total_tokens

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1
            # Here is possible to add one more block.
            # So do this in kvstorage.

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            # print(f"request {self.id} completed.")
            for kv_controller in self._replica_scheduler.get_all_kv_controllers():
                # print(f"Request {self._id} completed")
                if kv_controller is not None:
                    kv_controller.request_callback_on_restart_and_complete(self)
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        # print(f"Request {self._id} adds {execution_time}")
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "completed_at": self._completed_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
        }

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1
        self._kv_cache_hit_length = 0
        self._kv_inserted = False
        self._kv_fetch_time = 0
        self._kv_insert_time = 0
        for kv_controller in self._replica_scheduler.get_all_kv_controllers():
            # print(f"Request {self._id} restarting")
            if kv_controller is not None:
                kv_controller.request_callback_on_restart_and_complete(self)
