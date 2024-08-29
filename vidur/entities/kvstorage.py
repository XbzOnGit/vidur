from typing import List, Union, Tuple, Set
import random
import atexit
import math
from collections import deque
from vidur.entities.kv_block_trie import KVBlockTrie, KVBlock, KVBlockTrieNode
from vidur.entities.base_entity import BaseEntity
from vidur.entities.communications import Channel
from vidur.entities.batch import Batch
from vidur.logger import init_logger

logger = init_logger(__name__)


'''
1. Feed a batch stage with batch info into thsi controller.
Then it can manage the blocks of KV cache, after hit/miss/fetch/evict and allocate.
Also hack its token numbers but keep what's hacked restore information inside the batch.
Always restore before the batch gets to be processed, cos for this batch_stage, the KV operations haven't been done.
2. On batch stage end, it can insert full blocks and if completed, evict not-full active blocks.
'''
# All memory, including those in lower memory, are managed in blocks.
class KVStorageController(BaseEntity):
    def __init__(self, block_size, layer_pipeline: bool, num_layers_per_node: int, read_pipeline_buffer: bool, 
                 gpu_write_through_cpu: bool):
        # Now always fetch the longest path.
        self._id = KVStorageController.generate_id()
        self._kv_block_trie = KVBlockTrie(layer_pipeline, block_size, num_layers_per_node) # Only this storage, and everything on this.
        self._block_size = block_size
        self._active_blocks = {} # From reqid to number of active blocks.
        # Note that do not need to record the block here, cos its content can be gotten from request.tokens.


        # Note that this should record all requests in progress(RUNNING), even if no active blocks(0).
        # This can help refcnt to pin the blocks in GPU.
        # Only those calculated, the reused ones will not be changed, need not have new blocks.
        
        
        # The 'color' of a node is taken as the highest one in its location set.
        # Colors and locations are kept in KVBlockTrie, and also keep a frontier(leaf) for every color inside.
        # Should be able to be returned here, for the sake of eviction and fetch.
        self._layer_pipeline = layer_pipeline
        self._num_layers = num_layers_per_node # This is per node(per controller).
        self._read_pipeline_buffer = read_pipeline_buffer
        self._gpu_write_through_cpu = gpu_write_through_cpu
        self._storage_layer_cnt = 0
    
    # Note that the evict_policy and evict_op here, sometimes should evict in advance to make space for the next stage.
    # This should be done in scheduler, then after this, acquire in KVBlockTrie to avoid forced eviction that will 
    # be synced delay.
    def append_layer(self, num_blocks: int, read_thput, write_thput, 
                     evict_policy: str, evict_op: str, threshould_blocks: int, space_per_token_per_layer: int):
        self._kv_block_trie.append_layer(num_blocks, read_thput, write_thput, evict_policy, evict_op, threshould_blocks, 
                                         space_per_token_per_layer)
        self._storage_layer_cnt += 1
    
    def on_batch_stage(self, batch_to_hack: Batch, timestamp) -> Tuple[Tuple[float, float], List[Tuple[int, KVBlockTrieNode]]]:
        # Return the timepoint ready to execute and the per layer preload time.
        # Step 1. From num_processed to num_processed + num_tokens
        # Only care about those num_processed tokens % block_size == 0.
        # Now should RESET num_processed_tokens, cos it stands for KV cache.


        # Cut block by block and lookup, then allocate active blocks for the rest.
        # Step 2. Add overhead, hack requests and restore information.
        # Step 3. Return the timepoint ready to execute to relica stage scheduler.

        # It should execute and issue writes layer by layer.
        # batch_stage_end should restore.


        # TODO: Add support for disk, now it's not the same with CachedAttention.

        hit_traces = []
        preload_set: Set[KVBlockTrieNode] = set()
        synced_fetch_from_disk_to_memory_set: Set[KVBlockTrieNode] = set()
        preload_list: List[KVBlockTrieNode] = []
        synced_fetch_from_disk_to_memory_list: List[KVBlockTrieNode] = []
        number_of_blocks_to_active_diff = 0

        new_full_blocks_list: List[Tuple[int, KVBlockTrieNode]] = []
        end_first_layer_of_already_in_gpu = None
        bidx = 0
        for request in batch_to_hack.requests:
            if request.num_processed_tokens % self._block_size != 0:
                assert request.id in self._active_blocks
            number_of_tokens_to_lookup = max(request.num_processed_tokens, request.num_prefill_tokens)
            number_of_blocks_to_lookup = number_of_tokens_to_lookup // self._block_size
            kvblock_list = request.full_kvblocks[:number_of_blocks_to_lookup]
            # Lru list has been updated in lookup.
            # Record hit.
            hit_trace: List[KVBlockTrieNode] = self._kv_block_trie.lookup(kvblock_list, timestamp)
            hit_traces.append(hit_trace)
            # Should forever be enough space on GPU for replica scheduler to guarantee.
            new_req = request.id not in self._active_blocks
            for hit in hit_trace[1:]:
                # Just preload first.
                if new_req:
                    # Do not discard it.
                    # The only call to add_ref.
                    hit.add_ref()
                color = hit.color
                if color > 1:
                    # Do not need to preload from disk.
                    # But still add_ref.
                    assert color == 2
                    if hit not in synced_fetch_from_disk_to_memory_set:
                        synced_fetch_from_disk_to_memory_list.append(hit)
                        synced_fetch_from_disk_to_memory_set.add(hit)
                    if hit not in preload_set:
                        preload_list.append(hit)
                        preload_set.add(hit)
                elif color == 1:
                    # Add preload.
                    if hit not in preload_set:
                        preload_list.append(hit)
                        preload_set.add(hit)
                else:
                    assert color == 0
                    complete_time, complete_firlay_time = hit.get_ready_time(0)
                    if end_first_layer_of_already_in_gpu is None or complete_firlay_time > end_first_layer_of_already_in_gpu:
                        end_first_layer_of_already_in_gpu = complete_firlay_time
                    # This is because the preload of this batch should be together
                    # And preload of last batch should be done before this batch cos it is before execution of last batch.
            hit_length = len(hit_trace) - 1
            assert request.num_processed_tokens // self._block_size <= hit_length, f"{request.num_processed_tokens} // {self._block_size} > {hit_length}, id is {request.id}"
            # NOTE: It can be one more token, if prefill ends.
            curent_tokens_after_this = request.num_processed_tokens + batch_to_hack.num_tokens[bidx]
            if curent_tokens_after_this == request.num_prefill_tokens:
                curent_tokens_after_this += 1
            current_active_block = math.ceil((curent_tokens_after_this - 
                                    hit_length * self._block_size) / self._block_size)
            current_full_blocks = curent_tokens_after_this // self._block_size
            previous_have_blocks = hit_length
            previous_full_blocks = request.num_processed_tokens // self._block_size
            if previous_have_blocks > previous_full_blocks:
                # The later ones are effective hit.
                for i in range(previous_full_blocks, previous_have_blocks):
                    index = i + 1
                    hit_node = hit_trace[index]
                    self._kv_block_trie.add_hit(hit_node.color)
            last_virtual_full_block: KVBlockTrieNode = hit_trace[len(hit_trace) - 1]
            last_depth = last_virtual_full_block.depth
            if current_full_blocks > previous_have_blocks:
                for i in range(previous_have_blocks, current_full_blocks):
                    last_depth += 1
                    assert (i + 1) * self._block_size <= len(request.tokens), f"{((i + 1) * self._block_size)} > {len(request.tokens)}"
                    current_virtual_full_block = KVBlockTrieNode(last_virtual_full_block, 
                                                                 tuple(request.tokens[i * self._block_size: (i + 1) * self._block_size]),
                                                                 self._kv_block_trie, last_depth)
                    new_full_blocks_list.append((request.id, current_virtual_full_block))
                    last_virtual_full_block = current_virtual_full_block
            # print(f"Request {request.id} has {current_active_block} active blocks after counting current.")
            if request.id not in self._active_blocks:
                self._active_blocks[request.id] = current_active_block
                number_of_blocks_to_active_diff += current_active_block
            else:
                # _active_block only decreases on restart or completion of one block(and switch into cache).
                assert current_active_block >= self._active_blocks[request.id]
                number_of_blocks_to_active_diff += current_active_block - self._active_blocks[request.id]
                self._active_blocks[request.id] = current_active_block
            bidx += 1

        # Lookup and timestamp update in trie && active blocks record in controller.
        # Also add reference counter.

        # for hit_trace in hit_traces:
        #    for lno in range(self._kv_block_trie.num_storage_layers):
        #        self._kv_block_trie.evict_list_try_push_in_reverse_order(hit_trace, lno)
        assert len(preload_list) == len(preload_set)
        assert len(synced_fetch_from_disk_to_memory_list) == len(synced_fetch_from_disk_to_memory_set)
        number_of_blocks_to_preload = len(preload_list)
        number_of_blocks_to_synced_to_memory = len(synced_fetch_from_disk_to_memory_list)
        # print(f"Number of blocks to preload: {number_of_blocks_to_preload}, number of blocks to synced to memory: {number_of_blocks_to_synced_to_memory}")
        if number_of_blocks_to_synced_to_memory > 0:
            # Now fetch from disk to memory.
            # We know that it must be NOT in memory, so make space.
            # Loading into memory.
            # print("1")
            evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(1, 
                                                                                                            number_of_blocks_to_synced_to_memory, 
                                                                                                            timestamp, 
                                                                                                            False, self._read_pipeline_buffer)
            disk_read_channel = self._kv_block_trie.get_channel(1)[0]
            fetch_end_time, fetch_firlayer_time, per_layer_fetch_time = disk_read_channel.transmit(number_of_blocks_to_synced_to_memory *
                                                                                                   self._block_size, 
                                                                                                   evict_end_time, 
                                                                                                   self._num_layers)
            for node in synced_fetch_from_disk_to_memory_list:
                assert node.color == 2
                # LRU list has been updated on lookup.
                self._kv_block_trie.add_insert(1)
                node.fetch_to_higher_location(fetch_end_time, fetch_firlayer_time)
            # Now add the timepoint to fetch from memory to CPU.
            # NOTE: Change timestamp here, cos it is a synced time flow.
            timestamp = fetch_end_time

            
        # NOTE: The current assumption is always evict higher layers first.
        # So not possible to leave space in the middle.
        # So if write through && swap-out-once, make space here should return 0.0 overhead.

        # FIXME: Now launch eviction for active blocks together with preload blocks.
        # It can be latter, allowing preload to start earlier.
        
        
        assert number_of_blocks_to_active_diff >= 0
        # print(f"Number of blocks to active diff: {number_of_blocks_to_active_diff}\n\n")
        # Space for active blocks have been made above.
        # Active blocks marked in controller.
        # Now mark in KVBlockTrie.

        # Make space on the highest level.
        # avb = self._kv_block_trie._num_blocks[0] - self._kv_block_trie._used_blocks[0]
        # print("\n\n")
        # print(avb)
        # print(number_of_blocks_to_active_diff)
        # print(f"Used block: {self._kv_block_trie._used_blocks[0]}")
        if self._kv_block_trie.available_blocks(0) < number_of_blocks_to_active_diff:
            # print("Called")
            # Wait for it.
            # Recording of active blocks in controller have been updated.
            # print("2")
            original_blocks = self._kv_block_trie.available_blocks(0)
            msend, msfir, msper = self._kv_block_trie.synced_acquire_space(0, number_of_blocks_to_active_diff, timestamp, False, 
                                                                        False)
            # print(f"{original_blocks} -> {self._kv_block_trie.available_blocks(0)}")
            timestamp = msend
        else:
            # print("3")
            msend, mkfir, msper = self._kv_block_trie.synced_acquire_space(0, number_of_blocks_to_active_diff, timestamp, True, False)
            assert msper == 0.0
            assert msend == mkfir
            assert msend == timestamp
        
        # print("\n\n")
        # Space and time for active blocks.

        time_to_start_fetch = None
        read_channel = self._kv_block_trie.get_channel(0)[0]
        assert timestamp >= read_channel.last_time_in_use
        # print("One timestamp: ", timestamp)
        if self._layer_pipeline:
            if self._read_pipeline_buffer:
                # Now only support write through with preload read buffer.
                assert self._gpu_write_through_cpu
                # Then no actual eviction write is needed.
                # Cos always written before.
                # We know it must be NOT in GPU memory, so make space.
                # no write because should always be present in 1.
                # print("4")
                self._kv_block_trie.synced_acquire_space(0, number_of_blocks_to_preload, 
                                                      timestamp, True, True)
                # This is loading into 0, so use that buffer.
                # Should be the time max(read_channel.last_time_in_use, request scheduled time, space contention time)
                # FIXME: Modify scheduler to mark a scheduled time in advance in job queues.
                # FIXME: It is better to launch preload there.
                # FIXME: If preload is launched there, time_to_start_fetch is not important and should be skipped.
                # FIXME: Use another timestamp, attached in scheduler.
                time_to_start_fetch = max(read_channel.last_time_in_use, batch_to_hack.scheduled_at)
                # Note that need enough space at time_to_start_fetch.
            else:
                # Then fetch at that time.
                # After execution point, now.
                # print("5")
                evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(0, 
                                                                                            number_of_blocks_to_preload,
                                                                                            timestamp, False, False)
                # FIXME: Now make this case simpler, just wait until end.
                time_to_start_fetch = evict_end_time
        else:
            # print("6")
            evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(0,
                                                                                            number_of_blocks_to_preload,
                                                                                            timestamp, False, False)
            # Wait for evict to complete when no pipeline.
            time_to_start_fetch = evict_end_time

        end_time = time_to_start_fetch
        end_firlayer_time = time_to_start_fetch
        per_layer_preload_time = 0.0
        
        
        # Even if pipeline is enabled, last execution should end before the next one, thus preload should end before next execution.
        # This is because we always fetch all layers in one go.
        if number_of_blocks_to_preload > 0:
            end_time, end_firlayer_time, per_layer_preload_time = read_channel.transmit(number_of_blocks_to_preload * self._block_size, 
                                                                                    time_to_start_fetch, self._num_layers)
        # print("Preload time: ", end_time)
        
        # Now update trie for preload information.
        # FIXME: Now assume fetch will NOT stop due to eviction.
        # So when pipeline with read_preload_buffer met with eviction write back, always wait for eviction to complete for now.
        ready_exec_per_layer = (end_firlayer_time, per_layer_preload_time) if self._layer_pipeline else (end_time ,0.0)
        for node in preload_list:
            assert node.color == 1
            self._kv_block_trie.add_insert(0)
            node.fetch_to_higher_location(end_time, end_firlayer_time)
            
        # Have to wait for the whole batch to fetch, so set the batch fetch time on the node.
        # Now hack the batch and store the restore information.
        kv_hit_length_list = []
        num_processed_tokens_list = []
        should_reset_prefill_complete_list = []
        batch_num_tokens_list = []
        req_bidx = 0
        # Now really change it.
        for request in batch_to_hack.requests:
            kv_hit_length_list.append(request.kv_cache_hit_length)
            num_processed_tokens_list.append(request.num_processed_tokens)
            batch_num_tokens_list.append(batch_to_hack.num_tokens[req_bidx])
            if not request.is_prefill_complete:
                hit_trace = hit_traces[req_bidx]
                assert all(hit.color == 0 for hit in hit_trace[1:])
                hit_block_cnt = len(hit_trace) - 1
                hit_token_length = hit_block_cnt * self._block_size
                if hit_token_length > request.num_processed_tokens:
                    # Effective hit.
                    # If ==, can be just after a switch into cache.
                    # If <, some tokens KV cache can be inside active blocks.
                    diff_len = hit_token_length - request.num_processed_tokens
                    request.set_kv_cache_hit_length(hit_token_length)
                    request.set_num_processed_tokens(hit_token_length)
                    if request.num_processed_tokens >= request.num_prefill_tokens:
                        request.set_prefill_complete(end_time) # End after preload.
                        should_reset_prefill_complete_list.append(True)
                        diff_len = batch_to_hack.num_tokens[req_bidx] - 1
                    else:
                        should_reset_prefill_complete_list.append(False)
                    batch_to_hack.num_tokens[req_bidx] -= diff_len
            req_bidx += 1
        # Filter new_full_blocks_list to only take those really need.
        batch_to_hack.set_restore_between_stages(kv_hit_length_list, num_processed_tokens_list, should_reset_prefill_complete_list, batch_num_tokens_list, new_full_blocks_list)
        return ready_exec_per_layer, new_full_blocks_list



    # Called on restart or completion of a request.
    def remove_request_from_active_blocks(self, reqid):
        if reqid in self._active_blocks:
            cnt = self._active_blocks[reqid]
            del self._active_blocks[reqid]
            self._kv_block_trie.release_active_block_for_highest_level(cnt)
        else:
            raise ValueError(f"Request {reqid} not in active blocks.")
        
    def request_callback_on_restart_and_complete(self, request):
        # Remove ref cnt.
        # And remove from active blocks.
        if request.num_processed_tokens % self._block_size != 0:
            assert request.id in self._active_blocks
        num_processed_blocks = request.num_processed_tokens // self._block_size
        kvblock_list = request.full_kvblocks[:num_processed_blocks]
        # Should not update timestamp.
        hit_trace: List[KVBlockTrieNode] = self._kv_block_trie.lookup(kvblock_list, -1.0)
        for hit in hit_trace[1:]:
            hit.remove_ref()
        self.remove_request_from_active_blocks(request.id)

    @property
    def num_layers(self):
        return self._num_layers
    
    def write_through_async_to_CPU(self, new_full_blocks_list :List[Tuple[int, KVBlockTrieNode]], 
                                   timestamp, num_layer: int):
        # Here we should ONLY launch write.
        # And return time point that this layer is ready in CPU memory.

        # Can trigger eviction in CPU memory.
        # Filter new_full_blocks set.
        new_list = []
        for new_block in new_full_blocks_list:
            reqid, the_node = new_block
            parent = the_node.parent
            assert reqid in self._active_blocks
            assert len(the_node.tokens) == self._block_size, f"{len(the_node.tokens)} != {self._block_size}"
            if the_node.tokens not in parent.children:
                new_list.append(new_block)
            else:
                node: KVBlockTrieNode = parent.children[the_node.tokens]
                if not node.check_if_color_present(1):
                    new_list.append(new_block)

        needed_block_number = len(new_list)
        # This is writing to layer 1, so do not use the buffer.
        # print("7")
        end_time, end_fir_time, per_layer_time = self._kv_block_trie.synced_acquire_space(1, 
                                                needed_block_number, timestamp, False, False)
        # That buffer is used for fetch, so false for use_shadow_buf flag.
        write_channel = self._kv_block_trie.get_channel(0)[1]
        end_write, end_firlay_write, per_layer_write_time = \
        write_channel.transmit(needed_block_number * self._block_size, end_time, num_layer)
        return end_write, end_firlay_write, per_layer_write_time

    
    # Space has even been acquired before.
    def _insert_with_prepared_node_into_layer0(self, the_node: KVBlockTrieNode, end_time, end_firlayer_time, refcnt_inc: bool):
        parent_node = the_node.parent
        if the_node.tokens in parent_node.children:
            # Always fetch all the blocks, so if present, must be in GPU.
            # No need to update timestamp, already counted on hit.
            # assert parent_node.children[the_node.tokens].check_if_color_present(0), f"id: {parent_node.children[the_node.tokens].id}\n{parent_node.children[the_node.tokens]._storage_layer_info}"
            # Note that the node can be from active blocks instead of a fetch, so it is reasonable to not be inside layer 0.
            the_node = parent_node.children[the_node.tokens]
            # Anyway, it should be inside layer 0 now.
            original_color = the_node.color
            # If is 0, nothing to do.
            if original_color > 0:
                if original_color == 1:
                    # Just switch the active block inside.
                    # Need to acquire one block of space, cos active blocks have been released.
                    # Cannot fetch this block(no hit) cos do not know what will be generated in decoding.
                    # Then in active blocks, find the same block in cache, but not in GPU.
                    # This will push it into layer 0, with that active block as space.
                    # Later write through is naturally omitted.
                    self._kv_block_trie.insert_into_gpu_from_active_block_with_original_in_cpu(the_node, 
                                                                                               end_time, end_firlayer_time)
                else:
                    assert original_color == 2
                    raise NotImplementedError("Not implemented yet.")
        else:
            # A new block.
            the_node = self._kv_block_trie.insert_with_prepared_new_node(the_node, 0, end_time, end_firlayer_time)
        if refcnt_inc:
            the_node.add_ref()
        return the_node



    # Switch into cache with a list of timepoints
    def switch_active_fullblocks_into_cache(self, new_full_blocks_list: List[Tuple[int, KVBlockTrieNode]],
                                            end_time, end_firlayer_time):
        # print("Switching into cache.")
        # Timepoints are end of execution.
        # print("SWITCH INTO CACHE")
        release_cnt = 0
        for reqid, the_node in new_full_blocks_list:
            assert reqid in self._active_blocks
            assert self._active_blocks[reqid] > 0, f"{self._active_blocks[reqid]}, reqid: {reqid}"
            self._active_blocks[reqid] -= 1
            release_cnt += 1
        blocks_before_release = self._kv_block_trie.available_blocks(0)
        self._kv_block_trie.release_active_block_for_highest_level(release_cnt)
        real_insert = 0
        original_blocks = self._kv_block_trie.available_blocks(0)
        assert original_blocks >= release_cnt
        # Space should be enough.
        # Must insert in order.
        replace_list = []
        for reqid, the_node in new_full_blocks_list:
            # Timestamp updated inside.
            # Space also acquired inside.
            # Should have been fetched.
            new_node = self._insert_with_prepared_node_into_layer0(the_node, end_time, end_firlayer_time, True)
            if new_node == the_node:
                real_insert += 1
            replace_list.append((reqid, new_node))
        # print(f"Switch into cache: {real_insert} out of {len(new_full_blocks_list)}.")
        now_blocks = self._kv_block_trie.available_blocks(0)
        # print(f"Blocks: {blocks_before_release} -> {original_blocks} -> {now_blocks}\n")
        idx = 0
        for tp in replace_list:
            new_full_blocks_list[idx] = tp
            idx += 1
        

    def set_full_block_present_in_after_async_write(self, new_full_blocks_list: List[Tuple[int, KVBlockTrieNode]], 
                                                    end_time, end_firlayer_time):
        for reqid, the_node in new_full_blocks_list:
            assert reqid in self._active_blocks
            assert the_node.color == 0 # Must be present in GPU.
            if not the_node.check_if_color_present(1):
                self._kv_block_trie.add_insert(1)
                the_node.push_to_lower_location(end_time, end_firlayer_time)