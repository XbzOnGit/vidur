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
from vidur.entities import KVStorageController

logger = init_logger(__name__)


def _parse_memory_size(memory_str: str) -> int:
    memory_str = memory_str.upper()
    if memory_str.endswith("GB"):
        return int(memory_str[:-2]) * 1024**3
    if memory_str.endswith("MB"):
        return int(memory_str[:-2]) * 1024**2
    if memory_str.endswith("KB"):
        return int(memory_str[:-2]) * 1024
    return int(memory_str)

def _parse_thput(thput_str: str) -> float:
    thput_str = thput_str.upper()
    if thput_str.endswith("GB/S"):
        return float(thput_str[:-4]) * 1024**3
    if thput_str.endswith("MB/S"):
        return float(thput_str[:-4]) * 1024**2
    if thput_str.endswith("KB/S"):
        return float(thput_str[:-4]) * 1024
    return float(thput_str)

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
        available_memory_per_stage = available_memory // num_stages
        # FIXME: assuming fp16.
        # The second 2 for Key and Value.
        print(f"cache_lookup_type: {replica_scheduler_config.cache_lookup_type}")
        print(f"cache_evict_type: {replica_scheduler_config.cache_evict_type}")
        print(f"cache_evict_op: {replica_scheduler_config.cache_evict_op}")
        print(f"space per token on one stage: {(2 * 2 * replica.attention_head_dim * replica.num_kv_heads * replica.num_layers) // num_stages}")
        self._replica_kv_controllers = []
        read_buffer_fraction = 0.0
        if replica_scheduler_config.cache_lookup_type is not None:
            layer_pipeline = True if replica_scheduler_config.layer_pipeline.upper() == "TRUE" else False
            gpu_write_through_cpu = True if replica_scheduler_config.gpu_write_through_cpu.upper() == "TRUE" else False
            read_pipeline_buffer = True if replica_scheduler_config.read_pipeline_buffer.upper() == "TRUE" else False
            read_buffer_fraction = replica_scheduler_config.read_buffer_fraction
            cpu_sysbuf_fraction = replica_scheduler_config.cpu_sysbuf_fraction
            controller = KVStorageController(replica_scheduler_config.block_size, layer_pipeline, replica.num_layers // num_stages,
                                             read_pipeline_buffer, gpu_write_through_cpu)
            self._replica_kv_controllers.append(controller)
            # Space per token for each stage(each node).
            space_per_token = (2 * 2 * replica.attention_head_dim * replica.num_kv_heads * replica.num_layers) // num_stages
            space_per_token_per_layer = space_per_token // replica.num_layers
            space_per_block = replica_scheduler_config.block_size * space_per_token
            # Here no per TP, cos considered together.
            
            num_blocks = int(available_memory_per_stage // space_per_block)
            assert num_blocks > 0
            read_thput = 0
            write_thput = 0
            if len(replica_scheduler_config.cpu_memory_size) > 0:
                read_thput = _parse_thput(replica_scheduler_config.cpu_gpu_thput)
                write_thput = _parse_thput(replica_scheduler_config.gpu_cpu_thput)
            
            controller.append_layer(num_blocks, read_thput, write_thput, 
                                    replica_scheduler_config.cache_evict_type, 
                                    replica_scheduler_config.cache_evict_op,
                                    replica_scheduler_config.read_buffer_fraction, 
                                    space_per_token_per_layer)
            if len(replica_scheduler_config.cpu_memory_size) > 0:
                # FIXME: Now only CPU.
                cpu_memory_size = _parse_memory_size(replica_scheduler_config.cpu_memory_size)
                cpu_num_blocks = int(cpu_memory_size // space_per_block)
                assert cpu_num_blocks > 0
                controller.append_layer(cpu_num_blocks, 0, 0, replica_scheduler_config.cache_evict_type, "discard", 
                                        replica_scheduler_config.cpu_sysbuf_fraction, space_per_token_per_layer)
        else:
            for _ in range(num_stages):
                self._replica_kv_controllers.append(None)
        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )
        
        memory_planner = MemoryPlanner(self._replica_config, replica, read_buffer_fraction)

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
                self._replica_kv_controllers[stage_id],
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

    def on_batch_end(self, batch: Batch) -> None:
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
        for batch in scheduled_batches:
            for req in batch.requests:
                req.set_replica_scheduler(self)
        return scheduled_batches

    def get_kv_controller(self, stage_id: int) -> KVStorageController:
        return self._replica_kv_controllers[stage_id]

    def get_all_kv_controllers(self) -> List[KVStorageController]:
        return self._replica_kv_controllers
