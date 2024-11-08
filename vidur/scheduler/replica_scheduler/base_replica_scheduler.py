from abc import ABC, abstractmethod
from typing import List
import math

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
    CacheEngineConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner
from vidur.utils.parse_cli import parse_compression
from vidur.entities.kvitem import get_compress_level_manager

class KVSizeCalculator:
    def __init__(self, memory_planner):
        self._uncompressed_size_per_token_per_pipeline_stage = memory_planner.get_memory_per_token_per_pipeline_stage()
        assert type(self._uncompressed_size_per_token_per_pipeline_stage) == int
        print(f"uncompressed size per token per pipeline stage: {self._uncompressed_size_per_token_per_pipeline_stage}")
    def get_kv_size(self, token_number: int, compress_level: int) -> int:
        compress_level_manager = get_compress_level_manager()
        if compress_level == 0:
            assert type(token_number) == int
            assert type(self._uncompressed_size_per_token_per_pipeline_stage) == int
            return token_number * self._uncompressed_size_per_token_per_pipeline_stage
        else:
            return math.ceil(token_number * \
                             self._uncompressed_size_per_token_per_pipeline_stage\
                                  * compress_level_manager.to_rate[compress_level])

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
        simulator,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

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

        self._kv_size_calculator = KVSizeCalculator(memory_planner)
        compression_tuple = parse_compression(replica_config.compression)
        compress_level_manager = get_compress_level_manager()
        for level, compression in enumerate(compression_tuple):
            compress_level_manager.add_level(level, compression[0], 
                                             compression[1], compression[2], compression[3])
        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}
        cache_engine_config_per_pp = CacheEngineConfig(replica_config.eviction_policy,
                                                replica_config.ours_v1_token_thres,
                                                replica_config.memory_size_per_pp,
                                                replica_config.disk_size_per_pp,
                                                replica_config.disk_cpu_thput,
                                                replica_config.cpu_disk_thput,
                                                replica_config.cpu_gpu_thput,
                                                replica_config.gpu_cpu_thput,
                                                replica_config.disk_gpu_thput,
                                                replica_config.gpu_disk_thput,
                                                replica_config.contention_model,
                                                replica_config.gpu_prefix_cache,
                                                replica_config.cache_chunk_size,
                                                replica_config.cache_log,
                                                replica_config.store_policy)

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
                self._kv_size_calculator,
                cache_engine_config_per_pp,
                simulator,
            )
            for stage_id in range(num_stages)
        }

    @property
    def acc_exec_time(self) -> float:
        return sum(
            stage_scheduler.acc_exec_time
            for stage_scheduler in self._replica_stage_schedulers.values()
        )

    @property
    def kv_size_calculator(self) -> KVSizeCalculator:
        return self._kv_size_calculator

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

        return request.num_prefill_tokens

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

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self) -> List[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches
