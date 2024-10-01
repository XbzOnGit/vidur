from typing import Tuple
import atexit
from vidur.entities import Batch, BatchStage, ExecutionTime, KVStorageController
from vidur.execution_time_predictor import BaseExecutionTimePredictor

# This is one node.
class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        kv_cache_controller: KVStorageController,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor
        self._kv_cache_controller = kv_cache_controller
        # It is not so meaningful to asscociate the swap out time with the batch stage.
        # Cos it depends on evcition which is not related to this one.

        # Eviction time in total stored in storage.
        self._batch_queue = []
        self._is_busy = False

        self._prepare_time = 0.0
        self._compute_time = 0.0
        self._store_time = 0.0


        atexit.register(self.dump_stats)
    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False

    def dump_stats(self):
        print(f"Replica {self._replica_id}, stage {self._stage_id}"
              f" prepare time: {self._prepare_time}, compute time: {self._compute_time}, store_time: {self._store_time}\n")

    # The replica scheduler should be more conservative than this.
    # Cos it does not know about lower layers and will call restart on requests.
    # So if managing properly, the batch replica scheduler gives should 
    # have all of its previous KV cache inside memory.
    def on_schedule(self, timestamp) -> Tuple[Batch, BatchStage, ExecutionTime, float, float, list]:
        if self._is_busy or not self._batch_queue:
            return None, None, None, None, None, None
        self._is_busy = True
        batch = self._batch_queue.pop(0)
        ready_exec_timeinfo = (timestamp, 0.0)
        if self._kv_cache_controller is not None:
            ready_exec_timeinfo, new_full_blocks_list =  self._kv_cache_controller.lookup_then_fetch(batch, timestamp)
        start_first_exec_time, load_per_layer_time = ready_exec_timeinfo

        # Note that the fetch overhead here means the overhead that got stuck on cache fetch.
        # It can be zero because already in GPU or can be zero due to pipeline.
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        end_execution_time = None
        synced_write_end_time = None
        store_time = None
        if self._kv_cache_controller is not None:
            end_execution_time, synced_write_end_time = \
                self._kv_cache_controller.execution_and_store(execution_time, new_full_blocks_list, 
                                                                 start_first_exec_time, load_per_layer_time, timestamp)
            assert synced_write_end_time >= end_execution_time, f"{synced_write_end_time} < {end_execution_time}"
            store_time = synced_write_end_time - end_execution_time
            end_execution_time = synced_write_end_time
        else:
            end_execution_time = start_first_exec_time + execution_time.total_time
            new_full_blocks_list = []
        total_execution_time = end_execution_time - timestamp
        model_execution_time = execution_time.model_time
        self._compute_time += execution_time.total_time
        self._prepare_time += total_execution_time - execution_time.model_time
        if synced_write_end_time is not None:
            self._compute_time -= store_time
            self._store_time += store_time
        # batch_stage.execution_time = total_execution_time = end_execution_time - timestamp
        # So in the end, batch stage end arrivals at end_execution_time - timestamp + timestamp
        # But float point error can cause trouble here, the next assertion might fail.
        # assert total_execution_time + timestamp == end_execution_time, f"{total_execution_time} + {timestamp} != {end_execution_time}"
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )
        return batch, batch_stage, execution_time, start_first_exec_time, end_execution_time, new_full_blocks_list
