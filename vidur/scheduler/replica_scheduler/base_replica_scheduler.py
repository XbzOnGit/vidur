from abc import ABC, abstractmethod
from typing import List

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner
from vidur.entities import Storage

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages
        self._last_on_schedule_time = 0

        # FIXME: Currently one storage(highest level) for one replica.
        # FIXME: Now just prefix cache and lru.
        available_memory = (
            replica.total_memory_gb
            * 1024**3
            * (1 - replica.memory_margin_fraction)
        )
        # FIXME: assuming fp16.
        # The second 2 for Key and Value.
        print(f"cache_lookup_type: {replica_scheduler_config.cache_lookup_type}")
        print(f"cache_evict_type: {replica_scheduler_config.cache_evict_type}")
        print(f"cache_evict_op: {replica_scheduler_config.cache_evict_op}")
        self._replica_kv_cache = None
        if replica_scheduler_config.cache_lookup_type is not None:
            self._replica_kv_cache = Storage(replica.id, int(replica_config.inter_req_kvcache_fraction * available_memory), 
                                            replica_scheduler_config.cache_lookup_type, 
                                            replica_scheduler_config.cache_evict_type, 
                                            replica_scheduler_config.cache_evict_op, 
                                            True, 2 * 2 * replica.attention_head_dim * replica.num_kv_heads)
        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}
        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1
        lookup_result = None
        if self._replica_kv_cache is not None:
            lookup_result = self._replica_kv_cache.lookup(request.tokens)
        if lookup_result is not None:
            if len(lookup_result) != request.num_processed_tokens:
                request.set_kv_cache_hit_length(len(lookup_result))
                # print(f"request_id: {request.id}\nlookup hit with {len(lookup_result)} tokens\noriginally {request.num_processed_tokens} tokens processed\n")
            max_of_lookup_and_now = max(len(lookup_result), request.num_processed_tokens)
            request.set_num_processed_tokens(max_of_lookup_and_now)
        if request.num_processed_tokens >= request.num_prefill_tokens:
            request.set_prefill_complete(self._last_on_schedule_time)
            return 1
        return request.num_prefill_tokens - request.num_processed_tokens

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    def on_batch_end(self, batch: Batch) -> None:
        # Insert some tokens into cache.
        idx = 0
        for req in batch.requests:
            num_tokens_this_batch = batch.num_tokens[idx]
            if num_tokens_this_batch + req.num_processed_tokens >= req.num_prefill_tokens:
                insert_tokens = req.tokens
                # Only cache prefill part, cos we only have content for it.
                extraoverhead = 0
                if self._replica_kv_cache is not None:
                    extraoverhead = self._replica_kv_cache.insert(insert_tokens)
                # Now no extra overhead.
                assert extraoverhead == 0
            idx += 1
        self.sub_on_batch_end(batch)

    @abstractmethod
    def sub_on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self, timestamp) -> List[Batch]:
        scheduled_batches = []
        self._last_on_schedule_time = timestamp
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches
