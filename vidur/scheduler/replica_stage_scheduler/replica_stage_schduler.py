from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.config import CacheEngineConfig
from vidur.entities.compute import ComputationDevice
from vidur.utils.parse_cli import parse_size

class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        kv_size_calculator,
        cache_engine_config: CacheEngineConfig,
        simulator,
    ) -> None:
        self._simulator = simulator
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor
        self._kv_size_calculator = kv_size_calculator
        self._evict_policy = cache_engine_config.eviction_policy
        self._cpu_memory_size = 0
        if len(cache_engine_config.cpu_memory_size) > 0:
            self._cpu_memory_size = parse_size(cache_engine_config.cpu_memory_size)
        self._disk_size = 0
        if len(cache_engine_config.disk_size) > 0:
            self._disk_size = parse_size(cache_engine_config.disk_size)
        self._gpu_compute_device = ComputationDevice()
        from vidur.entities.cache_engine import CacheEngine
        self._cache_engine = CacheEngine(cache_engine_config, self)
        self._acc_exec_time = 0.0
        
        self._batch_queue = []
        self._is_busy = False

    @property
    def acc_exec_time(self):
        return self._acc_exec_time
    @property
    def simulator(self):
        return self._simulator

    @property
    def replica_id(self) -> int:
        return self._replica_id
    @property
    def stage_id(self) -> int:
        return self._stage_id
    @property
    def cache_engine(self):
        return self._cache_engine
    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    @property
    def gpu_compute_device(self):
        return self._gpu_compute_device
    
    @property
    def kv_size_calculator(self):
        return self._kv_size_calculator

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False


    def pre_stage_process(self, batch: Batch):
        batch.restore_batch_kv(self._stage_id)

    def batch_retrieve_kv_cache(self, batch: Batch, cur_time: float) -> float:
        hit_token_length = []
        for bidx, request in enumerate(batch.requests):
            # NOTE: Now no serving engine cache && always blocking.
            # TODO: Configure blocking/non-blocking.
            next_process_length = request.num_processed_tokens + batch.num_tokens[bidx]
            # print(f"request {request.id} with total length {len(request.tokens)}, after this batch, it will process {next_process_length}")
            if not request.is_prefill_complete:
                seen_prompt_len = next_process_length
                assert seen_prompt_len > 0, f"seen_prompt_len: {seen_prompt_len}"
                # print(f"\nrequest {request.id} is not prefill complete, retrieve query len is {seen_prompt_len}")
                hit_len, timepoint, quality = self._cache_engine.retrieve(cur_time, request.tokens[:seen_prompt_len], 0, True)
                cur_time = timepoint
                request.update_min_quality(quality)
                # TODO: serving engine can have an internal cache, 
                # we should look into that first, and provide a mask.
                hit_token_length.append(hit_len)
            else:
                hit_token_length.append(request.num_processed_tokens)
        batch.modify_batch_kv(hit_token_length)
        return cur_time

    def on_schedule(self, cur_time: float) -> Tuple[Batch, BatchStage, ExecutionTime, float]:
        if self._is_busy or not self._batch_queue:
            return None, None, None, cur_time
        # print(f"Replica {self._replica_id} Stage {self._stage_id} is scheduling batch {self._batch_queue[0].id}")
        self._is_busy = True
        batch = self._batch_queue.pop(0)
        self.pre_stage_process(batch)
        cur_time = self.batch_retrieve_kv_cache(batch, cur_time)
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )
        self._acc_exec_time += total_execution_time
        return batch, batch_stage, execution_time, cur_time
