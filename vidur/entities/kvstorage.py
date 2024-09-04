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
                 gpu_write_through_cpu: bool, disk_cpu_prefetch: bool):
        # Now always fetch the longest path.
        self._id = KVStorageController.generate_id()
        self._kv_block_trie = KVBlockTrie(layer_pipeline, block_size, num_layers_per_node, disk_cpu_prefetch) # Only this storage, and everything on this.
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
        self._disk_cpu_prefetch = disk_cpu_prefetch
        self._storage_layer_cnt = 0
        self._delay_of_waiting_for_prefetch_before_preload = 0.0
        self._wait_for_active_blocks_sync_time = 0.0
        self._wait_for_preload_space_synced_due_to_more_than_buf = 0.0
        self._time_saved_due_to_overlap_prefetch_and_preload = 0.0
        self._preload_start_before_schedule_time = 0.0
        self._read_buffer_available = []
        atexit.register(self.dump_stats)

    def set_read_buffer_available(self, layer_no, timepoint):
        assert timepoint >= self._read_buffer_available[layer_no]
        self._read_buffer_available[layer_no] = timepoint
    
    # Note that the evict_policy and evict_op here, sometimes should evict in advance to make space for the next stage.
    # This should be done in scheduler, then after this, acquire in KVBlockTrie to avoid forced eviction that will 
    # be synced delay.
    def append_layer(self, num_blocks: int, read_thput, write_thput, 
                     evict_policy: str, evict_op: str, threshould_blocks: int, space_per_token_per_layer: int):
        self._kv_block_trie.append_layer(num_blocks, read_thput, write_thput, evict_policy, evict_op, threshould_blocks, 
                                         space_per_token_per_layer)
        self._storage_layer_cnt += 1
        self._read_buffer_available.append(0.0)

    def dump_stats(self):
        print()
        print(f"delay_of_waiting_for_prefetch_before_preload: {self._delay_of_waiting_for_prefetch_before_preload}")
        print(f"wait_for_active_blocks_sync_time: {self._wait_for_active_blocks_sync_time}")
        print(f"wait_for_preload_space_synced_due_to_more_than_buf: {self._wait_for_preload_space_synced_due_to_more_than_buf}")
        print(f"time_saved_due_to_overlap_prefetch_and_preload: {self._time_saved_due_to_overlap_prefetch_and_preload}")
        print(f"preload_start_before_schedule_time: {self._preload_start_before_schedule_time}")
        print("\n")
    
    def on_batch_stage(self, batch_to_hack: Batch, timestamp) -> Tuple[Tuple[float, float], List[Tuple[int, KVBlockTrieNode]]]:
        # print(f"\n\nBatch {batch_to_hack.id} starts to be hacked.")
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
        # print(f"On batch_stage timestamp: {timestamp}")
        hit_traces = []
        preload_set: Set[KVBlockTrieNode] = set()
        synced_fetch_from_disk_to_memory_set: Set[KVBlockTrieNode] = set()
        preload_list: List[KVBlockTrieNode] = []
        synced_fetch_from_disk_to_memory_list: List[KVBlockTrieNode] = []
        number_of_blocks_to_active_diff = 0

        new_full_blocks_list: List[Tuple[int, KVBlockTrieNode]] = []
        end_first_layer_of_already_in_gpu = None
        bidx = 0
        max_arrival_time = max([request.arrived_at for request in batch_to_hack.requests])
        for request in batch_to_hack.requests:
            # print(f"\nRequest {request.id} starts to be hacked.")
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
            for hit in hit_trace[1:]:
                # Just preload first.
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
                hit.set_do_not_evict(True)
                if hit.is_in_evict_heap:
                    # Pin them.
                    # print(f"Remove {hit.id} from evict of layer {color}, it is leaf: {hit.is_leaf}")
                    # if len(hit.children) > 0:
                    #     for child in hit.children.values():
                    #         print(f"Child {child.id} has color {child.color}")
                    # else:
                    #     print("No children.")
                    hit.remove_from_evict_heap()
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
                # Effective hit.
                # The later ones are effective hit.
                for i in range(previous_full_blocks, previous_have_blocks):
                    index = i + 1
                    hit_node = hit_trace[index]
                    hit_node.add_ref() # Effective hit means it should not be completely evicted until request is done/restarted.
                    self._kv_block_trie.add_hit(hit_node.color)
            last_virtual_full_block: KVBlockTrieNode = hit_trace[len(hit_trace) - 1]
            last_depth = last_virtual_full_block.depth
            assert all(hit.refcnt > 0 for hit in hit_trace[1:])
            # So will not be evicted from disk.
            if current_full_blocks > previous_have_blocks:
                # Effective computational insert.
                for i in range(previous_have_blocks, current_full_blocks):
                    last_depth += 1
                    assert (i + 1) * self._block_size <= len(request.tokens), f"{((i + 1) * self._block_size)} > {len(request.tokens)}"
                    current_virtual_full_block = KVBlockTrieNode(last_virtual_full_block, 
                                                                 tuple(request.tokens[i * self._block_size: (i + 1) * self._block_size]),
                                                                 self._kv_block_trie, last_depth)
                    current_virtual_full_block.add_ref()
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
        synced_to_mem_end_time = timestamp
        synced_to_mem_fir_time = timestamp
        synced_to_mem_per_layer_time = 0.0
        if not self._disk_cpu_prefetch:
            synced_to_mem_end_time, synced_to_mem_fir_time, synced_to_mem_per_layer_time = self.synced_fetch_from_disk_to_memory(synced_fetch_from_disk_to_memory_list, timestamp)
        # Update timestamp cos it is synced, just as if time is passed there.
        else:
            # print(f"max arrival time is {max_arrival_time}, while timestamp is {timestamp}")
            synced_to_mem_end_time, synced_to_mem_fir_time, synced_to_mem_per_layer_time = self.oracle_prefetch_from_disk_to_memory(synced_fetch_from_disk_to_memory_list, max_arrival_time, 
                                                                 timestamp)

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
        msend = timestamp
        msfir = timestamp
        msper = 0.0
        if self._kv_block_trie.available_blocks(0) < number_of_blocks_to_active_diff:
            # print("Called")
            # Wait for it.
            # Recording of active blocks in controller have been updated.
            # print("2")
            original_blocks = self._kv_block_trie.available_blocks(0)
            msend, msfir, msper = self._kv_block_trie.synced_acquire_space(0, number_of_blocks_to_active_diff, timestamp, False, 
                                                                        False)
            # print(f"{original_blocks} -> {self._kv_block_trie.available_blocks(0)}")
            assert msend >= timestamp
            # FIXME: Should I pipeline this??!!
            self._wait_for_active_blocks_sync_time += msend - timestamp
        else:
            # print("3")
            # Async way.
            msend, mkfir, msper = self._kv_block_trie.synced_acquire_space(0, number_of_blocks_to_active_diff, timestamp, True, False)
            assert msper == 0.0
            assert msend == mkfir
            assert msend == timestamp
        
        # print("\n\n")
        # Space and time for active blocks.
        ready_exec_per_layer, end_time = self.preload_into_GPU(batch_to_hack, preload_list, timestamp, 
                                                               synced_to_mem_end_time, synced_to_mem_fir_time,
                                                               synced_to_mem_per_layer_time, msend, msfir, msper)
            
        # Have to wait for the whole batch to fetch, so set the batch fetch time on the node.
        # Now hack the batch and store the restore information.
        kv_hit_length_list = []
        num_processed_tokens_list = []
        should_reset_prefill_complete_list = []
        batch_num_tokens_list = []
        req_bidx = 0
        # Now really change it.

        # Now everything should be in GPU.
        # And no more eviction in this batch.
        # Before switching into cache, only the last one in hit trace possible to be evictable.
        for h_tr in hit_traces:
            h_len = len(h_tr) - 1
            for h in h_tr[1:]:
                assert h.color == 0
                # Note that there can be the same block in different hit traces.
                # So do not assert do_not_evict here.
                h.set_do_not_evict(False)
            if h_len > 0:
                last_hit: KVBlockTrieNode = h_tr[h_len]
                assert not last_hit.is_in_evict_heap
                last_hit.callback_on_possible_leaf_change()
                # if last_hit.is_in_evict_heap:
                #     print(f"Add {last_hit.id} to evict of layer {last_hit.color} after possible leaf change.")
                

        # Hack && restore ALL related indirect numbers of 
        # num_processed_tokens and prefill_complete and prefill_complete_time, num_tokens of batch.
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
    
    
    # Space has even been acquired before.
    def _insert_with_prepared_node_into_layer0(self, the_node: KVBlockTrieNode, end_time, end_firlayer_time):
        parent_node = the_node.parent
        # extra_node_for_async_write_through = False
        if the_node.tokens in parent_node.children:
            # Always fetch all the blocks, so if present, must be in GPU.
            # No need to update timestamp, already counted on hit.
            # assert parent_node.children[the_node.tokens].check_if_color_present(0), f"id: {parent_node.children[the_node.tokens].id}\n{parent_node.children[the_node.tokens]._storage_layer_info}"
            # Note that the node can be from active blocks instead of a fetch, so it is reasonable to not be inside layer 0.
            the_node = parent_node.children[the_node.tokens]
            # NOTE: Although have called add_ref in lookup && new full blocks, 
            # that new block might not be inserted, but hit a previous block.
            # so add_ref should switch.
            the_node.add_ref()
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

                    # This will not introduce overhead.
                    self._kv_block_trie.insert_into_gpu_from_active_block_with_original_in_cpu(the_node, 
                                                                                               end_time, end_firlayer_time)
                else:
                    assert original_color == 2
                    # FIXME: This means with disk, must have gpu to cpu write through.
                    assert self._gpu_write_through_cpu
                    # This can rely on later write through.
                    # extra_node_for_async_write_through = True
                    # self._kv_block_trie.insert_into_gpu_from_active_block_original_in_disk(the_node, 
                    #                                                                        end_time, end_firlayer_time)
                    self._kv_block_trie.insert_into_gpu_from_active_block_original_in_disk_allow_tft(the_node, 
                                                                                                        end_time, 
                                                                                                        end_firlayer_time)
        else:
            # A new block.
            the_node = self._kv_block_trie.insert_with_prepared_new_node(the_node, 0, end_time, end_firlayer_time)
            assert the_node.refcnt == 1
        return the_node


    def filter_write_to_CPU_and_preaccess(self, new_full_blocks_list :List[Tuple[int, KVBlockTrieNode]], timestamp):
        new_list: List[Tuple[int, KVBlockTrieNode]] = []
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
        for new_block in new_list:
            reqid, the_node = new_block
            # Preaccess only when inside heap, to pin it.
            if the_node.is_in_evict_heap:
                # If inside evict heap, need to pin it.
                the_node.timestamp_update(timestamp)
        return new_list
    
    def make_space_for_CPU(self, number_of_blocks_to_write: int, timestamp):
        return self._kv_block_trie.synced_acquire_space(1, number_of_blocks_to_write, timestamp, False, False)
    
    def use_channel(self, storage_layer, block_number, op_type, timestamp, num_layer):
        return self._kv_block_trie.get_channel(storage_layer)[op_type].transmit(block_number * self._block_size, timestamp, num_layer)

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
            new_node = self._insert_with_prepared_node_into_layer0(the_node, end_time, end_firlayer_time)
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

    '''
    # NOTE: If no hit on CPU at all, but very big cache size(like big_swap)
    # (layer_pipeline, read_pipeline_buffer, gpu_write_through_cpu):
    # No CPU memory can be the same with (False, False, True).
    # (True, True, True) in this case can give a close but different number??!! WHY??!!
    # (False, False, False) can be slower, cos sometimes it triggers synced evict from GPU to CPU.
    '''
    # Return (ready_exec_first_layer, per_layer_preload_time), end_time
    def preload_into_GPU(self, batch: Batch, preload_list: List[KVBlockTrieNode], timestamp, synced_to_mem_end_time, 
                         synced_to_mem_fir_time, synced_to_mem_per_layer_time, msend, msfir, msper):
        # max(max_arrival_time, last_channel_in_use, read_buffer_available).
        read_channel = self._kv_block_trie.get_channel(0)[0]
        last_channel_in_use_time = read_channel.last_time_in_use
        assert timestamp >= last_channel_in_use_time
        # Last execution must end after the last load to GPU.
        # Then this on schedule can be called.
        read_buffer_available = self._read_buffer_available[0]
        max_arrival_time = max([req.arrived_at for req in batch.requests])
        time_to_preload_start_lower_bound = max(max_arrival_time, last_channel_in_use_time, read_buffer_available)
        # print(f"max_arrival_time: {max_arrival_time}, last_channel_in_use: {last_channel_in_use_time}, read_buf_available: {read_buffer_available}")
        # print(f"Preload start time: {time_to_preload_start}")
        # Guarantee that read buffer is available.
        # Now make space.
        number_of_blocks_to_preload = len(preload_list)
        if number_of_blocks_to_preload == 0:
            # Note that this means synced from disk to mem must also be 0.
            assert synced_to_mem_end_time == timestamp
            assert synced_to_mem_fir_time == timestamp
            assert synced_to_mem_per_layer_time == 0.0
            # Just wait for active blocks.
            if msend > timestamp:
                # FIXME: Currently this is not pipelined.
                # But it can be.
                # print(f"Wait for active blocks time: {msend - timestamp}")
                return (msend, 0.0), msend
            return (timestamp, 0.0), timestamp
        
        # Check if any node here hit on CPU is not present, just prefetched from disk.
        # If so, need to wait for some time.
        # Wait for present on CPU is done by synced_mem_end_time.
        
        synced_to_mem_start = synced_to_mem_fir_time - synced_to_mem_per_layer_time

        # Still need to make space for the final timepoint to prefetch.
        '''
        NOTE:
        Before batch_stage is called, prefetch is launched.
        The sceond batch_stage is called, start to make space for active blocks and rbuffer.
        Calculate when to preload by considering when blocks are synced into memory.
        time_to_preload_start + per_layer_preload_time * (K-1) should be >= synced_to_mem_start + synced_to_mem_per_layer_time * K.
        for K from 1 to num_layers.
        time_to_preload_start >= synced_to_mem_start + synced_to_mem_per_layer_time * K - per_layer_preload_time * (K - 1).
        = synced_to_mem_start + K * (synced_to_mem_per_layer_time - per_layer_preload_time) + per_layer_preload_time.
        '''
        time_to_preload_start = None
        per_layer_preload_time = read_channel.get_per_layer_time(number_of_blocks_to_preload * self._block_size)
        constraint_due_to_into_cpu_pipelined = None
        if per_layer_preload_time <= synced_to_mem_per_layer_time:
            # Do not assert this, it can be not true when preload blocks is more than synced into memory.
            # assert per_layer_preload_time <= synced_to_mem_per_layer_time, f"{per_layer_preload_time} > {synced_to_mem_per_layer_time}"
            constraint_due_to_into_cpu_pipelined = synced_to_mem_start + \
                    self.num_layers * (synced_to_mem_per_layer_time - per_layer_preload_time) + per_layer_preload_time
        else:
            constraint_due_to_into_cpu_pipelined = synced_to_mem_start + synced_to_mem_per_layer_time
        # Then synced_to_mem_start + K * (synced_to_mem_per_layer_time - per_layer_preload_time) + per_layer_preload_time.
        # increase with K, the strongest constraint is on K = num_layers.
        # or on K = 1.
        if self._layer_pipeline:
            if self._read_pipeline_buffer:
                assert self._gpu_write_through_cpu
                read_buffer_blocks = self._kv_block_trie.read_buffer_blocks(0)
                make_space_rbuf_end_time = timestamp
                make_space_rbuf_firlayer_time = timestamp
                make_space_rbuf_per_layer_time = 0.0
                async_preload_number = number_of_blocks_to_preload
                if number_of_blocks_to_preload > read_buffer_blocks:
                    synced_block_num = number_of_blocks_to_preload - read_buffer_blocks
                    make_space_rbuf_end_time, make_space_rbuf_firlayer_time, \
                    make_space_rbuf_per_layer_time = self._kv_block_trie.synced_acquire_space(0, synced_block_num, 
                                                                                            timestamp, False, False)
                    async_preload_number = read_buffer_blocks
                    self._wait_for_preload_space_synced_due_to_more_than_buf += make_space_rbuf_end_time - timestamp
                
                swap_end_time, swap_firlayer_time, swap_per_layer_time = \
                self._kv_block_trie.synced_acquire_space(0, async_preload_number, 
                                                         make_space_rbuf_end_time, True, True)
                assert make_space_rbuf_end_time >= timestamp
                # TODO: Here it FORCES make_space_rbuf_end_time to be after timestamp.
                self.set_read_buffer_available(0, swap_end_time)
                # Only wait until synced space is acquired.
                # swap_end is marked, but do not wait here.
                # Max of CPU present in a pipelined way, preload buf ready.
                # FIXME: Now wait for synced preload space to be completely ready.
                if number_of_blocks_to_preload > read_buffer_blocks:
                    time_to_preload_start = max(make_space_rbuf_end_time, constraint_due_to_into_cpu_pipelined, time_to_preload_start_lower_bound)
                else:
                    # No synced make space.
                    # Just when CPU is ready, and lower bound ready.
                    time_to_preload_start = max(constraint_due_to_into_cpu_pipelined, time_to_preload_start_lower_bound)
                # print(f"make_space_rbuf_end_time: {make_space_rbuf_end_time}")
                # print(f"constraint_due_to_into_cpu_pipelined: {constraint_due_to_into_cpu_pipelined}")
                # print(f"time_to_preload_start_lower_bound: {time_to_preload_start_lower_bound}")
                if_not_overlap = max(make_space_rbuf_end_time, synced_to_mem_end_time)
                assert if_not_overlap >= time_to_preload_start
                if_not_end_fir_layer = if_not_overlap + per_layer_preload_time
                if if_not_end_fir_layer <= timestamp:
                    assert time_to_preload_start + per_layer_preload_time <= timestamp
                else:
                    # Only possible to save time when the first execution starting time is later than call of on_schedule.
                    start_of_first_exec_after_overlap = max(time_to_preload_start + per_layer_preload_time, timestamp)
                    start_of_first_exec_if_not_overlap = max(if_not_end_fir_layer, timestamp)
                    # Because at least after on schedule is called.
                    assert start_of_first_exec_after_overlap <= start_of_first_exec_if_not_overlap
                    # print(f"Time saved {start_of_first_exec_if_not_overlap - start_of_first_exec_after_overlap}")
                    self._time_saved_due_to_overlap_prefetch_and_preload += start_of_first_exec_if_not_overlap - start_of_first_exec_after_overlap
                # print(f"Time to preload start: {time_to_preload_start}")
                # print(f"timestamp: {timestamp}")
                # print(f"Start first exec: {max(time_to_preload_start + per_layer_preload_time, timestamp)}")
            else:
                # Until space ready && CPU completely in memory.
                evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(0, 
                                                                                            number_of_blocks_to_preload,
                                                                                            timestamp, False, False)
                # FIXME: Now just wait until enough space.
                time_to_preload_start = max(evict_end_time, synced_to_mem_end_time, time_to_preload_start_lower_bound)
        else:
            evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(0,
                                                                                            number_of_blocks_to_preload,
                                                                                            timestamp, False, False)
            # Wait for evict to complete when no pipeline.
            time_to_preload_start = max(evict_end_time, synced_to_mem_end_time, time_to_preload_start_lower_bound)
        
        # max(max_arrival_time, last_channel_in_use, read_buffer_available, cpu_present_time, synced_make_space_time).
        end_time = time_to_preload_start
        end_firlayer_time = time_to_preload_start
        per_layer_preload_time = 0.0

        self._preload_start_before_schedule_time += timestamp - time_to_preload_start
        # If > 0, save some time, else even wait for it.
        assert number_of_blocks_to_preload > 0
        end_time, end_firlayer_time, per_layer_preload_time = \
        read_channel.transmit(number_of_blocks_to_preload * self._block_size, 
                              time_to_preload_start, self._num_layers)
        ready_exec_per_layer = (end_firlayer_time, per_layer_preload_time) if self._layer_pipeline else (end_time, 0.0)
        for node in preload_list:
            assert node.color == 1
            self._kv_block_trie.add_insert(0)
            node.fetch_to_higher_location(end_time, end_firlayer_time)
        return ready_exec_per_layer, end_time
    

    def synced_fetch_from_disk_to_memory(self, synced_fetch_from_disk_to_memory_list: List[KVBlockTrieNode], timestamp: float):
        number_of_blocks_to_synced_to_memory = len(synced_fetch_from_disk_to_memory_list)
        if number_of_blocks_to_synced_to_memory == 0:
            return timestamp, timestamp, 0.0
        evict_end_time, evict_firlayer_time, evict_per_layer = self._kv_block_trie.synced_acquire_space(1, 
                                                                                                        number_of_blocks_to_synced_to_memory, 
                                                                                                        timestamp, False, False)
        wait_for_present_on_disk_end_time = timestamp
        for node in synced_fetch_from_disk_to_memory_list:
            assert node.color == 2
            ed_time, ed_fir_time = node.get_ready_time(2)
            if ed_time > wait_for_present_on_disk_end_time:
                wait_for_present_on_disk_end_time = ed_time
        start_to_fetch_time = max(wait_for_present_on_disk_end_time, evict_end_time)
        disk_read_channel = self._kv_block_trie.get_channel(1)[0]
        fetch_end_time, fetch_firlayer_time, per_layer_fetch_time = disk_read_channel.transmit(number_of_blocks_to_synced_to_memory *
                                                                                               self._block_size, 
                                                                                               start_to_fetch_time, 
                                                                                               self._num_layers)
        for node in synced_fetch_from_disk_to_memory_list:
            assert node.color == 2
            self._kv_block_trie.add_insert(1)
            node.fetch_to_higher_location(fetch_end_time, fetch_firlayer_time)
        return fetch_end_time, fetch_end_time, 0.0


    def oracle_prefetch_from_disk_to_memory(self, fetch_list: List[KVBlockTrieNode], max_arrival_time ,timestamp):
        number_of_blocks_to_fetch = len(fetch_list)
        if number_of_blocks_to_fetch == 0:
            return timestamp, timestamp, 0.0
        # Only prefetch the size that fits.
        read_buf_size = self._kv_block_trie.read_buffer_blocks(1)
        disk_read_channel: Channel = self._kv_block_trie.get_channel(1)[0]
        assert number_of_blocks_to_fetch <= read_buf_size
        # Make read buf BIG enough, larger than KV cache available blocks in GPU.
        # So according to original scheduler, should be in this range.


        # Note that this might not be the best to use max_arrival_time.
        # buffer av for space constraint.
        # max_arrival is logical constraint.
        # last_in_use for channel constraint.

        # To make every batch just fetch once from disk to make it simpler, 
        # now batch fetch together. But add time differently.
        wait_for_present_on_disk_end_time = 0.0
        for node in fetch_list:
            assert node.color == 2
            ed_time, ed_fir_time = node.get_ready_time(2)
            if ed_time > wait_for_present_on_disk_end_time:
                wait_for_present_on_disk_end_time = ed_time
        # Wait for present before prefetch.
        oracle_time_to_start_prefetch = max(max_arrival_time, self._read_buffer_available[1], 
                                            disk_read_channel.last_time_in_use, wait_for_present_on_disk_end_time)
        # Launch together, but divide the time.
        swap_end_time, swap_firlayer_time, swap_per_layer = self._kv_block_trie.synced_acquire_space(1,
                                                                                number_of_blocks_to_fetch,
                                                                                oracle_time_to_start_prefetch,
                                                                                True,
                                                                                True, True)
        self._read_buffer_available[1] = swap_end_time

        fetch_end_time, fetch_firlayer_time, per_layer_fetch_time = disk_read_channel.transmit(\
            number_of_blocks_to_fetch * self._block_size, oracle_time_to_start_prefetch, self._num_layers)
        final_end_time = max(fetch_end_time, timestamp)
        self._delay_of_waiting_for_prefetch_before_preload += final_end_time - timestamp
        # print(f"End of prefetch is {fetch_end_time}")
        # for swapped space to be ready. 
        for node in fetch_list:
            assert node.color == 2
            self._kv_block_trie.add_insert(1)
            node.fetch_to_higher_location(fetch_end_time, fetch_firlayer_time)
        return fetch_end_time, fetch_firlayer_time, per_layer_fetch_time