from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime, KVStorageController
from vidur.execution_time_predictor import BaseExecutionTimePredictor


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

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False


    # The replica scheduler should be more conservative than this.
    # Cos it does not know about lower layers and will call restart on requests.
    # So if managing properly, the batch replica scheduler gives should 
    # have all of its previous KV cache inside memory.
    def on_schedule(self, timestamp) -> Tuple[Batch, BatchStage, ExecutionTime, float, float, list]:
        if self._is_busy or not self._batch_queue:
            return None, None, None, None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        # Now just synchoronously fetch and insert.
        # Do fetch, execute, insert here.
        ready_exec_timeinfo = (timestamp, 0.0)
        if self._kv_cache_controller is not None:
            ready_exec_timeinfo, new_full_blocks_list =  self._kv_cache_controller.on_batch_stage(batch, timestamp)
        start_first_exec_time, load_per_layer_time = ready_exec_timeinfo

        # Note that the fetch overhead here means the overhead that got stuck on cache fetch.
        # It can be zero because already in GPU or can be zero due to pipeline.
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        if self._kv_cache_controller is not None:
            per_layer_execution_time = execution_time.total_time / self._kv_cache_controller.num_layers
            end_execution_time = None
            end_last_exec_time = timestamp
            end_last_preload_time = start_first_exec_time
            end_exec_of_first_layer = None
            cpu_make_space_per_layer_time = None
            end_last_cpu_make_space_layer_time = None
            
            assert timestamp <= start_first_exec_time, f"{timestamp} > {start_first_exec_time}"
            async_write_list = []
            if self._kv_cache_controller._gpu_write_through_cpu:
                # If not, do not write to CPU here.
                # FIXME: Does this naturally pin the blocks??!!
                new_list = self._kv_cache_controller.filter_write_to_CPU_and_preaccess(new_full_blocks_list, timestamp)
                needed_block_number = len(new_list)
                end_cpu_make_space_time, end_cpu_make_space_fir_time, cpu_make_space_per_layer_time = \
                self._kv_cache_controller.make_space_for_CPU(needed_block_number, timestamp)
                end_last_cpu_make_space_layer_time = timestamp # Get per layer time that CPU memory is available.
                # Make CPU has this much space to write to, can trigger eviction to disks.
            for _ in range(self._kv_cache_controller.num_layers):
                start_this_exec_time = max(end_last_exec_time, end_last_preload_time)
                end_this_exec_time = start_this_exec_time + per_layer_execution_time
                if end_exec_of_first_layer is None:
                    end_exec_of_first_layer = end_this_exec_time
                # Launch async write.
                if self._kv_cache_controller._gpu_write_through_cpu:
                    # Has filtered to not present in CPU in write_through inside this function.
                    write_timepoint = max(end_last_cpu_make_space_layer_time, end_this_exec_time)
                    # Should be execed && have enough CPU space.
                    # Note that it is 0 --> 1 write, so layer_no is 0.
                    end_aw, end_faw, end_per_aw = \
                    self._kv_cache_controller.use_channel(0, needed_block_number, 1, write_timepoint, 1)
                    assert end_aw == end_faw, f"{end_aw} != {end_faw}"
                    async_write_list.append(end_aw)
                    # Assume make space is continuous.
                    end_last_cpu_make_space_layer_time += cpu_make_space_per_layer_time
                end_last_exec_time = end_this_exec_time
                # Assume that preload is continuous.
                end_last_preload_time += load_per_layer_time
            end_execution_time = end_last_exec_time
            if len(new_full_blocks_list) > 0:
                self._kv_cache_controller.switch_active_fullblocks_into_cache(new_full_blocks_list, 
                                                                        end_execution_time, end_exec_of_first_layer)
                if self._kv_cache_controller._gpu_write_through_cpu:
                    assert len(async_write_list) == self._kv_cache_controller.num_layers
                    # Set present time in CPU.
                    self._kv_cache_controller.set_full_block_present_in_after_async_write(new_full_blocks_list, 
                                                                                        async_write_list[-1],
                                                                                        async_write_list[0])
        else:
            end_execution_time = start_first_exec_time + execution_time.total_time
            new_full_blocks_list = []
        # TODO: Check total execution time here.
        total_execution_time = end_execution_time - timestamp
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
        '''
        print(f"Replica stage {batch_stage.id} total execution time returned by predictor: ", execution_time.total_time)
        print(f"Replica stage {batch_stage.id} total execution time calculated: ", total_execution_time)
        print(f"Batch token number is {batch.num_tokens}")
        # NOTE: Some of execution time is in ms, so convert to seconds.
        print(f"prefill time on one batch_stage {batch_stage.id}: {execution_time.attention_prefill_execution_time * 1e-3}")
        print(f"decode time on one batch_stage {batch_stage.id}: {execution_time.attention_decode_execution_time * 1e-3}\n")
        print(f"attention_layer_pre_proj_execution_time: {execution_time._attention_layer_pre_proj_execution_time * 1e-3}")
        print(f"attention_layer_post_proj_execution_time: {execution_time._attention_layer_post_proj_execution_time * 1e-3}")
        print(f"attention_rope_execution_time: {execution_time._attention_rope_execution_time * 1e-3}")
        print(f"attention_kv_cache_save_execution_time: {execution_time._attention_kv_cache_save_execution_time * 1e-3}")
        print(f"attention_decode_execution_time: {execution_time._attention_decode_execution_time * 1e-3}")
        print(f"attention_prefill_execution_time: {execution_time._attention_prefill_execution_time * 1e-3}")
        print(f"tensor_parallel_communication_time: {execution_time._tensor_parallel_communication_time * 1e-3}")
        print(f"attn_norm_time: {execution_time._attn_norm_time * 1e-3}")
        print(f"mlp_layer_up_proj_execution_time: {execution_time._mlp_layer_up_proj_execution_time * 1e-3}")
        print(f"mlp_layer_down_proj_execution_time: {execution_time._mlp_layer_down_proj_execution_time * 1e-3}")
        print(f"mlp_layer_act_execution_time: {execution_time._mlp_layer_act_execution_time * 1e-3}")
        print(f"tensor_parallel_communication_time: {execution_time._tensor_parallel_communication_time * 1e-3}")
        print(f"mlp_norm_time: {execution_time._mlp_norm_time * 1e-3}")
        print(f"get_attention_layer_execution_time: {execution_time._get_attention_layer_execution_time() * 1e-3}")
        print(f"get_mlp_layer_execution_time: {execution_time._get_mlp_layer_execution_time() * 1e-3}")
        print(f"add_time: {execution_time._add_time * 1e-3}")
        print(f"block_execution_time: {execution_time._get_block_execution_time() * 1e-3}")
        print(f"block_execution_time * num_layers_per_pipeline_stage: {execution_time._get_block_execution_time() * 1e-3 * execution_time._num_layers_per_pipeline_stage}")
        print(f"model_time: {execution_time.model_time}, cpu_overhead: {execution_time.total_time - execution_time.model_time}\n\n\n")
        '''
        return batch, batch_stage, execution_time, start_first_exec_time, end_execution_time, new_full_blocks_list
