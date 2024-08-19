from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime, Storage, StorageController
from vidur.execution_time_predictor import BaseExecutionTimePredictor


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        kv_cache_controller: StorageController
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

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False

    def modify_request_states(self, batch: Batch, timestamp):
        # Note that this also modifies kv_cache_controller and returns an overhead.
        # Also the batch.
        fetch_overhead = 0.0
        # request's kv_hit_length, num_processed_tokens, prefill_complete or not.
        # batch's num_tokens.
        # The above should be modified back before the next stage.
        # add_kv_fetch should be kept.
        # print(batch)
        kv_hit_length_list = []
        num_processed_tokens_list = []
        should_reset_prefill_complete_list = []
        batch_num_tokens_list = []
        if self._kv_cache_controller is not None:
            req_bidx = 0
            for request in batch.requests:
                kv_hit_length_list.append(request.kv_cache_hit_length)
                num_processed_tokens_list.append(request.num_processed_tokens)
                reset_prefill = False
                batch_num_tokens_list.append(batch.num_tokens[req_bidx])
                if not request.is_prefill_complete:
                    assert request.num_processed_tokens < request.num_prefill_tokens
                    lookup_result, overhead, _ = self._kv_cache_controller.lookup(request.tokens, True, request.num_processed_tokens)
                    fetch_overhead += overhead
                    if lookup_result is not None:
                        assert len(lookup_result) > request.num_processed_tokens
                        diff_len = len(lookup_result) - request.num_processed_tokens
                        request.set_kv_cache_hit_length(len(lookup_result))
                        request.set_num_processed_tokens(len(lookup_result))
                        if request.num_processed_tokens >= request.num_prefill_tokens:
                            request.set_prefill_complete(timestamp)
                            reset_prefill = True
                            diff_len = batch.num_tokens[req_bidx] - 1 # Make it 1.
                        # print(f"difflen: {diff_len}")
                        batch.num_tokens[req_bidx] -= diff_len
                    request.add_kv_fetch_time(overhead)
                should_reset_prefill_complete_list.append(reset_prefill)
                req_bidx += 1
        return kv_hit_length_list, num_processed_tokens_list, should_reset_prefill_complete_list, batch_num_tokens_list, fetch_overhead


    def on_schedule(self, timestamp) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            return None, None, None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        # Now just synchoronously fetch and insert.
        # Do fetch, execute, insert here.
        kv_hit_length_list, num_processed_tokens_list, should_reset_prefill_complete_list, batch_num_tokens_list, fetch_overhead = self.modify_request_states(batch, timestamp)
        # print(batch)
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        # insert_overhead should be kept to add across stages.
        # But kv_inserted should be reset to make next stage insert those tokens.
        should_reset_kv_inserted_list = []
        insert_overhead = 0.0
        if self._kv_cache_controller is not None:
            for request in batch.requests:
                if not request.kv_inserted:
                    overhead = self._kv_cache_controller.insert(request.tokens)
                    request.add_kv_insert_time(overhead)
                    request.set_kv_inserted(True)
                    insert_overhead += overhead
                    should_reset_kv_inserted_list.append(True)
                else:
                    should_reset_prefill_complete_list.append(False)
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            fetch_overhead,
            insert_overhead,
            batch.requests,
            batch.num_tokens,
        )
        batch.set_restore_between_stages(
            kv_hit_length_list,
            num_processed_tokens_list,
            should_reset_prefill_complete_list,
            batch_num_tokens_list,
            should_reset_kv_inserted_list,
        )
        return batch, batch_stage, execution_time, fetch_overhead, insert_overhead
