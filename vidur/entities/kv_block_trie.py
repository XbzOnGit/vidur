from typing import List, Tuple
from vidur.entities.communications import Channel
from vidur.utils.heap import Heap
import atexit


# The exposed APIs should return the timestamp that it is possible for the next operations.
# Like a complete timestamp and a layer timestamp, and a per layer time interval.
# Then controller can return organize the time to start execution and end execution based on this.
# Issue a series of writes if async write.

class KVBlock:
    def __init__(self, block_size):
        self.block_size = block_size
        self.tokens = []
    def add_token(self, token):
        self.tokens.append(token)
    def set_tokens(self, tokens: List):
        self.tokens = tokens
    def is_full(self):
        return len(self.tokens) == self.block_size
    def get_size(self):
        return len(self.tokens)

general_kv_block_size = None

# 1. Always make refcnt > 0 for RUNNING requests full blocks of kV cache.
# It should even go into lower layers, remember to add_ref when inserting blocks, even if duplicated.
# 2. For active blocks, for those more than one batch's time, at most one block for one RUNNING request.
# This is acceptable to be always in GPU memory.
# 3. refcnt > 0 --> can evict to lower layers, but do not be discarded.
# 1 + 2 + 3 can guarantee that a request not restarted can always have num_processed_tokens all stored in KV cache
# (it will not degrade).
# 4. On restart, all active blocks and blocks refcnt along the path are degraded.

# 5. When that layer is not the lowest, any leaf is in candidates.
# When the layer is in lowest, only refcnt == 0 is in candidates.


kv_trie_node_id = 0
class KVBlockTrieNode:
    def __init__(self, parent, tokens_parent_to_here: Tuple[str], kvtrie, depth):
        global kv_trie_node_id
        self.id = kv_trie_node_id
        kv_trie_node_id += 1

        self._do_not_evict = False

        self.record_of_evictlist_ops = []
        self.parent = parent
        self.children = {}
        self.depth = depth
        # Currently always remove from highest level first.
        # And color is also the highest.
        # A heap should work.
        # The edge is a block, from parent to this node.
        self._refcnt = 0 # Also the edge from parent to this node.
        
        self._position_in_evict_heap = -1
        # (lru_timestamp, -length)
        # So for the same timestamp, longer length is smaller, so upper 
        # in min_heap --> evicted first.
        self._evict_timestamp = (-1, -self.depth)

        # FIXME: Currently at most 3 layers.
        # (is_present, timestamp, first_layer_timestamp)
        self._storage_layer_info = [(False, -1.0, -1.0), (False, -1.0, -1.0), (False, -1.0, -1.0)]
        self._children_color_cnt = [0, 0, 0]

        self.pgdsf_frequency = 0
        self.pgdsf_total_cost = 0
        self.pgdsf_computenum = 0
        self.pgdsf_avgcost = 0
        self.pgdsf_priority = 0
        self.last_pgdsf_priority_update_layer = -1


        self._cachedattention_in_disk_eviction_window = -1
        self._cachedattention_in_cpu_eviction_window = -1

        self._position_in_reorder_kv_heap = -1


        self._tokens_parent_to_here = tokens_parent_to_here
        self._kvtrie: KVBlockTrie = kvtrie

    @property
    def color(self):
        idx = 0
        for layer_info in self._storage_layer_info:
            if layer_info[0]:
                return idx
            idx += 1
        return -1

    @property
    def do_not_evict(self):
        return self._do_not_evict
    
    def set_do_not_evict(self, value):
        self._do_not_evict = value

    def get_ready_time(self, layer_no: int):
        assert layer_no >= 0
        assert layer_no < self._kvtrie.num_storage_layers
        assert self._storage_layer_info[layer_no][0]
        return self._storage_layer_info[layer_no][1], self._storage_layer_info[layer_no][2]
    @property
    def cachedattention_in_disk_eviction_window(self):
        return self._cachedattention_in_disk_eviction_window
    
    @property
    def cachedattention_in_cpu_eviction_window(self):
        return self._cachedattention_in_cpu_eviction_window
    
    def set_cachedattention_in_disk_eviction_window(self, value):
        self._cachedattention_in_disk_eviction_window = value
    
    def set_cachedattention_in_cpu_eviction_window(self, value):
        self._cachedattention_in_cpu_eviction_window = value

    @property
    def position_in_evict_heap(self):
        return self._position_in_evict_heap
    @property
    def storage_layer_info(self):
        return self._storage_layer_info
    
    @property
    def is_leaf(self):
        # Color of itself && color of children.
        if self._kvtrie.allow_reorder_kv_blocks:
            # Any node is leaf, allow eviction.
            return True
        color = self.color
        # [0, color]
        i = 0
        while i < color:
            assert self._children_color_cnt[i] == 0, f"{self.id} {i}\nself: {self._storage_layer_info}\nchild: {self._children_color_cnt}"
            i += 1
        return self._children_color_cnt[color] == 0
    
    @property
    def is_evict_candidate(self):
        is_leaf = self.is_leaf
        if not is_leaf:
            return False
        else:
            color = self.color
            assert color < self._kvtrie.num_storage_layers
            if color == self._kvtrie.num_storage_layers - 1:
                return self._refcnt == 0
            return True
    
    @property
    def is_in_evict_heap(self):
        return self._position_in_evict_heap >= 0
    
    @property
    def evict_timestamp(self):
        return self._evict_timestamp

    def swap_in_evict_heap(self, another_node):
        this_position = self._position_in_evict_heap
        another_position = another_node._position_in_evict_heap
        the_array = self._kvtrie.evict_candidates[self.color]
        original_len = len(the_array)
        the_array[this_position] = another_node
        the_array[another_position] = self
        self._position_in_evict_heap = another_position
        another_node._position_in_evict_heap = this_position
        now_len = len(the_array)
        assert now_len == original_len

    def sift_up_evict_heap(self, unconditional: bool):
        the_array = self._kvtrie.evict_candidates[self.color]
        original_len = len(the_array)
        position = self._position_in_evict_heap
        while position > 0:
            parent_position = (position - 1) // 2
            parent = the_array[parent_position]
            if not unconditional and parent.evict_timestamp <= self.evict_timestamp:
                break
            self.swap_in_evict_heap(parent)
            position = parent_position
            assert self._position_in_evict_heap == position
            assert the_array[position] == self
        now_len = len(the_array)
        assert now_len == original_len
            
    def sift_down_evict_heap(self, unconditional: bool):
        the_array = self._kvtrie.evict_candidates[self.color]
        original_len = len(the_array)
        position = self._position_in_evict_heap
        while position < len(the_array):
            left_child_position = 2 * position + 1
            right_child_position = 2 * position + 2
            if left_child_position >= len(the_array):
                # Already leaf.
                break
            left_child = the_array[left_child_position]
            right_child = None
            if right_child_position < len(the_array):
                right_child = the_array[right_child_position]
            assert left_child is not None
            if right_child is not None and right_child.evict_timestamp < left_child.evict_timestamp:
                min_child = right_child
                min_child_position = right_child_position
            else:
                min_child = left_child
                min_child_position = left_child_position
            if not unconditional and min_child.evict_timestamp >= self.evict_timestamp:
                break
            # If unconditional, swap until leaf.
            self.swap_in_evict_heap(min_child)
            position = min_child_position
            assert self._position_in_evict_heap == position
            assert the_array[position] == self
        now_len = len(the_array)
        assert now_len == original_len
    
    def pop_self_root_in_evict_heap(self):
        the_array = self._kvtrie.evict_candidates[self.color]
        original_len = len(the_array)
        assert len(the_array) > 0
        assert the_array[0] == self
        assert self._position_in_evict_heap == 0
        self._position_in_evict_heap = -1
        if len(the_array) == 1:
            the_array.pop()
        else:
            last_node = the_array.pop()
            last_node._position_in_evict_heap = 0
            the_array[0] = last_node
            last_node.sift_down_evict_heap(False)
        assert len(the_array) == original_len - 1

    
    def remove_from_evict_heap(self):
        # print(f"Remove block {self.id} from layer {self.color} evict heap.")
        color = self.color
        self._kvtrie.add_evict_heap_size(color, -1)
        the_array = self._kvtrie.evict_candidates[color]
        orginal_len = len(the_array)
        assert self._position_in_evict_heap >= 0
        assert the_array[self._position_in_evict_heap] == self
        self.sift_up_evict_heap(True)
        self.pop_self_root_in_evict_heap()
        # print(f"layer{color}: {orginal_len} --> {len(the_array)}")
        assert len(the_array) == orginal_len - 1
        

    def add_self_to_evict_heap(self):
        # print("Add to evict heap called.\n\n")
        # These timestamps are timestamp_update first, then looked up when adding into heap.
        # if self.id == 1581:
        #     print(f"Add to evict heap on node {self.id} with color {self.color}, with timestamp {self.evict_timestamp} BEFORE")
        if self._kvtrie.scheduler_aware_eviction:
            timestamp = None
            if self.evict_timestamp[1] < 0:
                timestamp = self.evict_timestamp[0]
            else:
                timestamp = self.evict_timestamp[1]
            assert timestamp >= 0
            if self.color == 0:
                self._evict_timestamp = (timestamp, -self.depth)
            else:
                if self.color == 1:
                    self._evict_timestamp = (self._cachedattention_in_cpu_eviction_window, timestamp)
                else:
                    assert self.color == 2
                    self._evict_timestamp = (self._cachedattention_in_disk_eviction_window, timestamp)
        assert not self.do_not_evict
        if not self._kvtrie.scheduler_aware_eviction or self.color == 0:
            assert self.evict_timestamp[0] >= 0
            assert self.evict_timestamp[1] < 0
        # assert self.evict_timestamp[0] >= 0, f"{self.evict_timestamp}, id: {self.id}, layer: {self.color}"
        # It can be < 0, if not in eviction window.
        # assert self.evict_timestamp[1] < 0
        # It can be >= 0, if in layer 1 and 2.
        color = self.color
        if self._kvtrie.is_pgdsf_eviction[color]:
            self.set_pgdsf_priority(color) # Possible to switch to another clock.
            self._evict_timestamp = (self.pgdsf_priority, -self.depth)
        self._kvtrie.add_evict_heap_size(color, 1)
        # print(f"Add block {self.id} to layer {color} evict heap.")
        
        the_array = self._kvtrie.evict_candidates[color]
        original_len = len(the_array)
        assert self._position_in_evict_heap == -1
        self._position_in_evict_heap = len(the_array)
        the_array.append(self)
        self.sift_up_evict_heap(False)
        # print(f"layer{color}: {original_len} --> {len(the_array)}")
        assert len(the_array) == original_len + 1
        
    
    # Called on color changed && children color change.
    # Timestamp already updated.
    def callback_on_possible_leaf_change(self):
        if self == self._kvtrie.root:
            return
        is_leaf = self.is_leaf
        color = self.color
        if self.is_in_evict_heap:
            if not is_leaf:
                # print(f"{self.id} Remove from heap on leaf change to non-leaf.")
                self.remove_from_evict_heap()
        else:
            if is_leaf:
                if self.do_not_evict:
                    return
                if color == self._kvtrie.num_storage_layers - 1:
                    if self._refcnt == 0:
                        # print(f"{self.id} Add to heap on leaf refcnt == 0.")
                        self.add_self_to_evict_heap()
                else:
                    # print(f"{self.id} Add to heap on is leaf.")
                    self.add_self_to_evict_heap()

    def modify_reorder_kv_heap_on_color_change(self):
        if self._kvtrie.block_tokens_to_nodes is not None:
            if self._position_in_reorder_kv_heap >= 0:
                the_heap_of_reorder: Heap = self._kvtrie.block_tokens_to_nodes[self.tokens]
                the_heap_of_reorder.modify_heap(self)

    def pop_self_from_reorder_kv_heap_on_evict_to_discard(self):
        if self._kvtrie.block_tokens_to_nodes is not None:
            assert self._position_in_reorder_kv_heap >= 0
            the_heap_of_reorder: Heap = self._kvtrie.block_tokens_to_nodes[self.tokens]
            the_heap_of_reorder.delete(self)
            # Even clear the heap.
            if the_heap_of_reorder.size() == 0:
                del self._kvtrie.block_tokens_to_nodes[self.tokens]

    def assert_inside_reorder_kv_heap(self):
        if self._kvtrie.block_tokens_to_nodes is not None:
            assert self._position_in_reorder_kv_heap >= 0
    
    def check_if_inside_reorder_kv_heap(self):
        if self._kvtrie.block_tokens_to_nodes is not None:
            return self._position_in_reorder_kv_heap >= 0
        return False

    def insert_into_reorder_kv_heap(self):
        if self._kvtrie.block_tokens_to_nodes is not None:
            if self.tokens not in self._kvtrie.block_tokens_to_nodes:
                self._kvtrie.block_tokens_to_nodes[self.tokens] = Heap("color", "_position_in_reorder_kv_heap")
            the_heap_of_reorder: Heap = self._kvtrie.block_tokens_to_nodes[self.tokens]
            the_heap_of_reorder.insert(self)

    

    # Interal states on evictions, color, leaves.
    def callback_on_fetch(self, from_no: int, to_no: int, timestamp, layer_timestamp):
        assert from_no == to_no + 1
        assert self._storage_layer_info[from_no][0]
        assert not self._storage_layer_info[to_no][0]
        assert self.color == from_no # Should only be called when needing a fetch.
        assert self.color >= 0
        self.assert_inside_reorder_kv_heap() # Do not insert
        # Anyway, need to remove from evict list of from_no layer cos color changed.
        # print(f"{self.id} Remove from heap on fetch from {from_no} to {to_no}.")
        # Color changed, but perhaps it is not in heap before.
        if self.is_in_evict_heap:
            self.remove_from_evict_heap()
        # print(f"timestamp of node {self.id} updated with {timestamp} for callback_on_fetch.")
        self.timestamp_update(timestamp)
        # Do not delete storage layer info of from_no, it is still there.
        # Update color.
        self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
        # NOTE: Color can change.
        self.modify_reorder_kv_heap_on_color_change()
        # print(f"{self.id}: {self._storage_layer_info}")
        # print(f"Parent ID is {self.parent.id}")
        # print(f"Parent color is {self.parent.color}")
        # Color changed from_no --> to_no(decreased).
        self.parent._children_color_cnt[from_no] -= 1
        assert self.parent._children_color_cnt[from_no] >= 0
        assert self.parent._children_color_cnt[to_no] >= 0
        self.parent._children_color_cnt[to_no] += 1
        # Can make parent no longer a leaf.
        self.parent.callback_on_possible_leaf_change()
        # Can make itself a leaf, add to list.
        # It should be the latest one to access, which is the tail.
        # This is reasonable because when we fetch it, it should be important.
        self.callback_on_possible_leaf_change()
        # Once the batch does not exceed GPU memory, 
        # and when eviction is triggered on GPU, 
        # previous ones fetched to GPU are naturally pinned.
        # Cos there must be node not accessed, and must be leaf.
        # That leaf LRU is earlier than what we accessed before.


        # TODO: Further improvement: Know the batch not request, 
        # we can pre-access those already in GPU once, to make them 
        # not evicted then loaded when preparing the batch.
    
    # [True, False, True]
    def callback_on_switch_to_layer0_tft(self, timestamp, layer_timestamp):
        assert not self._storage_layer_info[0][0]
        assert not self._storage_layer_info[1][0]
        assert self._storage_layer_info[2][0]
        if self.is_in_evict_heap:
            # print(f"{self.id} Remove from heap on switch to layer 0 tft.")
            self.remove_from_evict_heap()
        # print(f"timestamp of node {self.id} updated with {timestamp} for callback_on_switch_to_layer0_tft.")
        self.timestamp_update(timestamp)
        self._storage_layer_info[0] = (True, timestamp, layer_timestamp)
        # NOTE: Color can change.
        if not self.check_if_inside_reorder_kv_heap():
            self.insert_into_reorder_kv_heap()
        else:
            self.modify_reorder_kv_heap_on_color_change()
        self.parent._children_color_cnt[2] -= 1
        assert self.parent._children_color_cnt[2] >= 0
        assert self.parent._children_color_cnt[0] >= 0
        self.parent._children_color_cnt[0] += 1
        self.parent.callback_on_possible_leaf_change()
        self.callback_on_possible_leaf_change()

    # NOTE: When a node is pushed to lower layer in write-through, no color change at all.
    # So no leaf change and candidate changes.
    def callback_on_push_lower(self, from_no: int, to_no: int, timestamp, layer_timestamp):
        assert from_no == to_no - 1
        assert self._storage_layer_info[from_no][0]
        # No color change.
        if self._storage_layer_info[to_no][0]:
            assert self._storage_layer_info[to_no][1] <= timestamp
            assert self._storage_layer_info[to_no][2] <= layer_timestamp
        else:
            color_before = self.color
            self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
            color_after = self.color
            assert color_before == color_after
            # No color change.
            # So no leaf change.
    
    def timestamp_update(self, new_access_time):
        color = self.color
        assert color >= 0, f"{self.id} {color}"
        assert self._storage_layer_info[color][0]
        if self._kvtrie.no_real_timestamp_update[color]:
            return
        another_form = False
        original_timestamp = self._evict_timestamp
        if self._kvtrie.scheduler_aware_eviction:
            if color == 1:
                self._evict_timestamp = (self._cachedattention_in_cpu_eviction_window, new_access_time)
                another_form = True
            elif color == 2:
                self._evict_timestamp = (self._cachedattention_in_disk_eviction_window, new_access_time)
                another_form = True
        if not another_form:
            old_access_time = self._evict_timestamp[0]
            self._evict_timestamp = (new_access_time, -self.depth)
            if self.is_in_evict_heap:
                assert old_access_time <= new_access_time, f"{self.id} {old_access_time} > {new_access_time}, layer {color}"
                self.sift_down_evict_heap(False)
        else:
            if self.is_in_evict_heap:
                if self.evict_timestamp < original_timestamp:
                    self.sift_up_evict_heap(False)
                else:
                    self.sift_down_evict_heap(False)
        # if self.id == 1581:
        #     print(f"Update timestamp on node {self.id} with color {self.color}, with new timestamp {new_access_time}, in the end: {self.evict_timestamp}")
                

    def set_pgdsf_priority(self, layer_no):
        self.pgdsf_priority = self._kvtrie.pgdsf_clock[layer_no] + self.pgdsf_avgcost * self.pgdsf_frequency
        self.last_pgdsf_priority_update_layer = layer_no



    def pgdsf_update(self, exec_time_of_request, beta, is_cached, layer_no):
        # print(f"Update {self.id} with exec_time_of_request: {exec_time_of_request}, beta: {beta}, is_cached: {is_cached}, layer_no: {layer_no}")
        assert self._kvtrie.is_pgdsf_eviction[layer_no]
        self.pgdsf_frequency += 1
        if not is_cached:
            assert beta > 0
            self.pgdsf_total_cost += exec_time_of_request / beta
            self.pgdsf_computenum += 1
            self.pgdsf_avgcost = self.pgdsf_total_cost / self.pgdsf_computenum
        assert self.pgdsf_computenum > 0
        if self.is_in_evict_heap:
            assert self.pgdsf_priority == self.evict_timestamp[0]
        # Use the corresponding layer's color, that clock.
        self.set_pgdsf_priority(layer_no)
        # print(f"Update on node {self.id} with layer {layer_no}, is_cached {is_cached}, color {self.color}, with clock {self._kvtrie.pgdsf_clock[layer_no]}, frequency {self.pgdsf_frequency}, avgcost {self.pgdsf_avgcost}, priority {self.pgdsf_priority}")
        if self.is_in_evict_heap:
            if self.pgdsf_priority < self.evict_timestamp[0]:
                self._evict_timestamp = (self.pgdsf_priority, -self.depth)
                self.sift_up_evict_heap(False)
            else:
                self._evict_timestamp = (self.pgdsf_priority, -self.depth)
                self.sift_down_evict_heap(False)
        else:
            self._evict_timestamp = (self.pgdsf_priority, -self.depth)
    
        
    def pgdsf_transfer(self, from_temp_new_node):
        assert from_temp_new_node.pgdsf_frequency == 1
        assert from_temp_new_node.pgdsf_computenum == 1
        self.pgdsf_frequency += 1
        self.pgdsf_total_cost += from_temp_new_node.pgdsf_total_cost
        self.pgdsf_computenum += 1
        self.pgdsf_avgcost = self.pgdsf_total_cost / self.pgdsf_computenum
        if self.is_in_evict_heap:
            assert self.pgdsf_priority == self.evict_timestamp[0]
        color = self.color
        assert color >= 0
        assert self._kvtrie.is_pgdsf_eviction[color]
        self.set_pgdsf_priority(color)
        # print(f"transfer to node {self.id} with color {self.color}, with clock {self._kvtrie.pgdsf_clock[self.color]}, frequency {self.pgdsf_frequency}, avgcost {self.pgdsf_avgcost}, priority {self.pgdsf_priority}, from node {from_temp_new_node.id}")
        if self.is_in_evict_heap:
            if self.pgdsf_priority < self.evict_timestamp[0]:
                self._evict_timestamp = (self.pgdsf_priority, -self.depth)
                self.sift_up_evict_heap(False)
            else:
                self._evict_timestamp = (self.pgdsf_priority, -self.depth)
                self.sift_down_evict_heap(False)
        else:
            self._evict_timestamp = (self.pgdsf_priority, -self.depth)



    def callback_on_evict(self, from_no: int, to_no: int, timestamp, layer_timestamp):
        assert from_no == to_no - 1
        assert self._storage_layer_info[from_no][0]
        assert self.color == from_no
        # Because always evict higher layers of the same copy.
        # Guranteed by the fact that it is only in evict list of its color.
        # Anyway, need to remove from evict list of from_no layer cos color changed.
        # print(f"{self.id} Remove from heap on evict from {from_no} to {to_no}.")
        assert not self.is_in_evict_heap # Should have been removed from list in evict_selection.
        # self.remove_from_evict_heap()
        # Update color.
        self._storage_layer_info[from_no] = (False, -1.0, -1.0)
        # Swap out once always true.
        if self._storage_layer_info[to_no][0]:
            assert self._storage_layer_info[to_no][1] <= timestamp
            assert self._storage_layer_info[to_no][2] <= layer_timestamp
        else:
            self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
        if not self.check_if_inside_reorder_kv_heap():
            self.insert_into_reorder_kv_heap()
        # NOTE: Color can change.
        self.modify_reorder_kv_heap_on_color_change()
        # Even if already inside, still color change.
        # Color changed from_no --> to_no(increased).
        self.parent._children_color_cnt[from_no] -= 1
        assert self.parent._children_color_cnt[from_no] >= 0
        assert self.parent._children_color_cnt[to_no] >= 0
        self.parent._children_color_cnt[to_no] += 1
        # Can make parent a leaf, add to evict_list.
        self.parent.callback_on_possible_leaf_change()
        # Can make itself no longer a leaf, remove from evict_list.
        self.callback_on_possible_leaf_change()
    
    def set_storage_layer_info_timestamps(self, layer_no, timestamp, layer_timestamp):
        # assert self._storage_layer_info[layer_no][0], f"{self.id} {layer_no}"
        # assert self._storage_layer_info[layer_no][1] == float("inf")
        # assert self._storage_layer_info[layer_no][2] == float("inf")
        self._storage_layer_info[layer_no] = (True, timestamp, layer_timestamp)
        if not self.check_if_inside_reorder_kv_heap():
            self.insert_into_reorder_kv_heap()
        # NOTE: Color can change.
        self.modify_reorder_kv_heap_on_color_change()

    
    def callback_on_insert_into_gpu(self, timestamp, layer_timestamp):
        # Only called when not in GPU before.
        # A new node.
        assert self.color == -1
        assert not self._storage_layer_info[0][0]
        # Should be new in trie.
        assert len(self.children) == 0
        assert not self.is_in_evict_heap # A new node, updated later in callback_on_possible_leaf_change.
        self._storage_layer_info[0] = (True, timestamp, layer_timestamp)
        if not self.check_if_inside_reorder_kv_heap():
            self.insert_into_reorder_kv_heap()
        # NOTE: Color can change.
        self.modify_reorder_kv_heap_on_color_change()
        # Update timestamps.
        # print(f"timestamp of node {self.id} updated with {timestamp} for callback_on_insert_into_gpu.")
        self.timestamp_update(timestamp)
        # print(f"Insert into GPU: {self.id}, evict_timestamp: {self.evict_timestamp}")
        # Color changed -1 --> 0.
        assert self.parent is not None
        assert self.parent._children_color_cnt[0] >= 0
        self.parent._children_color_cnt[0] += 1
        self.parent.callback_on_possible_leaf_change()
        self.callback_on_possible_leaf_change()


        
    def add_child_node(self, child_node):
        assert child_node.tokens not in self.children
        assert len(child_node.children) == 0
        self.children[child_node.tokens] = child_node


    def insert_a_new_child_node(self, child_node):
        assert child_node.tokens not in self.children
        assert len(child_node.children) == 0
        self.children[child_node.tokens] = child_node
        assert child_node.parent == self
        child_color = child_node.color
        assert child_color >= 0
        assert self._children_color_cnt[child_color] >= 0
        self._children_color_cnt[child_color] += 1
        self.callback_on_possible_leaf_change()
        child_node.callback_on_possible_leaf_change()
        

    
    def fetch_to_higher_location(self, complete_timestamp, complete_layer_timestamp):
        color = self.color
        assert color > 0, f"color: {color}" # Only fetch to higher if not there.
        higher_level_no = color - 1
        # print(f"Fetch {self.id} from {color} to {higher_level_no}.")
        # print(f"Before fetch: {self._storage_layer_info}\n")
        self.callback_on_fetch(color, higher_level_no, complete_timestamp, complete_layer_timestamp)
        return higher_level_no

    def push_to_lower_location(self, complete_timestamp, complete_layer_timestamp):
        color = self.color
        assert color >= 0
        lower_level_no = color + 1
        already_inside = False
        if self._storage_layer_info[lower_level_no][0]:
            already_inside = True
        assert not already_inside
        self.callback_on_push_lower(color, lower_level_no, complete_timestamp, complete_layer_timestamp)
        return lower_level_no
    
    def evict_to_lower_location(self, complete_timestamp, complete_layer_timestamp):
        assert not self.is_in_evict_heap # Should have been removed from list in evict_selection.
        color = self.color
        assert color >= 0
        lower_level_no = color + 1
        assert lower_level_no < self._kvtrie.num_storage_layers
        already_inside = False
        if self._storage_layer_info[lower_level_no][0]:
            already_inside = True
        self.callback_on_evict(color, lower_level_no, complete_timestamp, complete_layer_timestamp)
        return already_inside, lower_level_no
    
    def evict_to_discard(self):
        color = self.color
        assert color == self._kvtrie.num_storage_layers - 1
        assert self._refcnt == 0
        assert not self.is_in_evict_heap # Should have been removed from list in evict_selection.
        self._storage_layer_info[color] = (False, -1.0, -1.0)
        # NOTE: color will change to -1, just pop from heap.
        assert self.color == -1
        self.assert_inside_reorder_kv_heap()
        self.pop_self_from_reorder_kv_heap_on_evict_to_discard()
        self.parent._children_color_cnt[color] -= 1
        assert self.parent._children_color_cnt[color] >= 0
        self.parent.callback_on_possible_leaf_change()
        # For self, have been removed from evict list.
        # Delete node outside from parent children.
        # print(f"After evict to discard, color of {self.id} is {self.color}")
   
    def check_if_color_present(self, location_id):
        return self._storage_layer_info[location_id][0]

    def add_ref(self):
        assert self._refcnt >= 0
        if self._refcnt == 0:
            last_layer_no = self._kvtrie.num_storage_layers - 1
            assert last_layer_no >= 0
            if self.color == last_layer_no:
                # Only possible to be in one evict heap(colored one).
                if self.is_in_evict_heap:
                    # No leaf changes, no color changes.
                    # print(f"{self.id} Remove from heap on refcnt increase.")
                    self.remove_from_evict_heap()
        self._refcnt += 1
        # print(f"{self.id} add_ref refcnt from {self._refcnt - 1} to {self._refcnt}")
    
    def remove_ref(self):
        self._refcnt -= 1
        # print(f"{self.id} remove_ref refcnt from {self._refcnt + 1} to {self._refcnt}")
        assert self._refcnt >= 0, f"{self.id} with refcnt {self._refcnt}"
        if self._refcnt == 0:
            # print(f"{self} refcnt drops to 0.")
            # If refcnt drops to 0, allow to evict from last layer.
            last_layer_no = self._kvtrie.num_storage_layers - 1
            assert last_layer_no >= 0
            if self.color == last_layer_no:
                assert not self.is_in_evict_heap
                self.callback_on_possible_leaf_change()

    @property
    def refcnt(self):
        return self._refcnt
    
    @property
    def tokens(self):
        return self._tokens_parent_to_here
    

    

# Random and FIFO not easy to implement now.

# Model channels outside the trie, this is only information about it.
class KVBlockTrie:
    def __init__(self, layer_pipeline: bool, block_size, num_layers: int, 
                 disk_cpu_prefetch: bool, scheduler_aware_eviction: bool, allow_reorder_kv_blocks: bool):
        self.root = KVBlockTrieNode(None, tuple(), self, 0)
        print(f"scheduler-aware prefetch: {disk_cpu_prefetch}")
        print(f"scheduler-aware eviction: {scheduler_aware_eviction}")
        # Configurations
        self._block_size = block_size
        global general_kv_block_size
        if general_kv_block_size is not None:
            assert general_kv_block_size == block_size
        self._allow_reorder_kv_blocks = allow_reorder_kv_blocks
        self._evict_heap_size = []
        general_kv_block_size = block_size
        self._num_layers = num_layers
        self._num_blocks = []
        self._num_threshold_blocks = [] # When exceeding this, async evict.
        self._channel_from_this_to_lower = []
        self.eviction_selection = []
        self._insert_op = [] # Used for write through.
        self._evict_op = [] # Discard, write back, write through.
        # If write through, no need to evict, so callback same as Discard.

        self._insert_cnt = []
        self._evict_cnt = []
        self._hit_cnt = []
        self._active_blocks = []

        # Runtime info.
        self._used_blocks = []
        # Write through && swap out once can make eviction on insertion 0 time for this layer.
        # Runtime info.
        # The candidates are leaves - refcnt > 0.
        # It should change
        # 1. leaves change.
        # 2. refcnt change.
        # 3. order changes(like access).
        # Access can be lookup or insertion, anyway the change takes effect when it is back into candidates.
        # It must be out of it when it is referenced.
        # Leaves out are evicting into lower/discarded, so out of candidates, no need to update list order.
        # Leaves in are inserted or fetched, inside candidates again, just insert to the front(fifo and lru both).
        # refcnt change, for lru should insert into front when it drops back, while fifo should remember some place.
        # (I can implement a LRU first).
        # Access always changes refcnt.

        # This just needs to be interpretable by the evict selection function && other callbacks 
        # that will be called on access.
        # Now for LRU, just the head and tail of the trienode list.
        self._evict_candidates = []
        

        self._layer_pipeline = layer_pipeline
        self._disk_cpu_prefetch = disk_cpu_prefetch
        self._scheduler_aware_eviction = scheduler_aware_eviction
        self._cachedattention_newest_mark = -1

        self._no_real_timestamp_update = [False, False, False]
        # scheduler eviction still has update, but in another form.
        self._is_pgdsf_eviction = [False, False, False]
        self._pgdsf_clock = []

        self.block_tokens_to_nodes = None
        if self._allow_reorder_kv_blocks:
            self.block_tokens_to_nodes = {}

        self._blocks_extended_by_reorder = 0

        atexit.register(self.dump_stats)

    def set_cachedattention_newest_mark(self, value):
        self._cachedattention_newest_mark = value
    
    @property
    def cachedattention_newest_mark(self):
        return self._cachedattention_newest_mark
    
    @property
    def scheduler_aware_eviction(self):
        return self._scheduler_aware_eviction

    def dump_stats(self):
        print(f"Blocks extended by reorder: {self._blocks_extended_by_reorder}")
        for i in range(len(self._num_blocks)):
            print(f"Layer {i}:")
            print(f"Insert: {self._insert_cnt[i]}, Evict: {self._evict_cnt[i]}, Hit: {self._hit_cnt[i]}")

    def _print_trie_node(self, node: KVBlockTrieNode):
        print(f"Node: {node.id}")
        print(f"Refcnt: {node.refcnt}")
        print(f"Colors: {node._colors}")
        if node.parent is not None:
            print(f"parent: {node.parent.id}")
            if node.parent.id != node.id - 1:
                print(f"node {node.id} with parent {node.parent.id} NOT -1 form!!")
        else:
            print("parent: None")
        print("--------------------------------------------")
        for child in node.children.values():
            self._print_trie_node(child)
    def _print_trie(self):
        self._print_trie_node(self.root)

    def _print_leaves_node(self, color ,node: KVBlockTrieNode):
        if node is not self.root:
            if node.is_leaf and node.color == color:
                print(f"Leaf: {node.id} of layer {color}")
        for child in node.children.values():
            self._print_leaves_node(color, child)

    def _print_leaves(self, color):
        self._print_leaves_node(color, self.root)

    def _print_color_node(self, color, node: KVBlockTrieNode):
        if node is not self.root:
            if node.color == color:
                print(f"Node: {node.id} of layer {color}")
        for child in node.children.values():
            self._print_color_node(color, child)

    def _print_color(self, color):
        self._print_color_node(color, self.root)

    def _print_present_node(self, color, node: KVBlockTrieNode):
        if node is not self.root:
            if node.check_if_color_present(color):
                print(f"Node: {node.id} of layer {color}, {node.storage_layer_info[0][0], node.storage_layer_info[1][0], node.storage_layer_info[2][0]}")
        for child in node.children.values():
            self._print_present_node(color, child)

    def _print_present(self, color):
        self._print_present_node(color, self.root)

    def _get_size_node(self, color, node: KVBlockTrieNode):
        retval = 0
        if node is not self.root:
            if node.check_if_color_present(color):
                retval += 1
        for child in node.children.values():
            retval += self._get_size_node(color, child)
        return retval

    def _get_size(self, color):
        calculated_size = self._get_size_node(color, self.root)
        if color > 0:
            assert self._used_blocks[color] == calculated_size, f"{self._used_blocks[color]} != {calculated_size} on layer {color}"
        else:
            assert self._used_blocks[color] == calculated_size + self._active_blocks[0], f"{self._used_blocks[color]} != {calculated_size} + {self._active_blocks[0]} = {calculated_size + self._active_blocks[0]} on layer {color}"
        return calculated_size
    
    def check_size_consistency(self):
        assert False, f"Should not be called."
        for i in range(self.num_storage_layers):
            self._get_size(i)

    def delete_node(self, node: KVBlockTrieNode):
        assert node.color == -1 # Call evict_discard just before it.
        assert node.refcnt == 0
        assert not node.is_in_evict_heap # Removed on selection.
        assert len(node.children) == 0, f"{node.id} -len{len(node.children)}-> {next(iter(node.children.values())).id}"
        del node.parent.children[node.tokens]
        # Update of parent in evict_to_discard.

    def add_evict_heap_size(self, color ,diffsize):
        # print(f"Layer {color} heap size diff: {diffsize}")
        # print(f"{self._evict_heap_size[color]} --> {self._evict_heap_size[color] + diffsize}")
        self._evict_heap_size[color] += diffsize
        
    # If into this function consistently, should exit consistently.
    # Because color and size both updated inside.
    def _evict_blocks(self, layer_no, evict_number, timestamp, no_write: bool) -> Tuple[float, float, float]:
        # self.check_size_consistency()
        assert evict_number >= 0
        if evict_number == 0:
            return timestamp, timestamp, 0.0
        # Synced evict.
        # Select blocks.
        # Modify trie to record eviction.
        # free space.
        # Return end_time, first_layer_end_time, per_layer_time_interval.
        assert layer_no >= 0
        assert layer_no < self.num_storage_layers
        evict_selection_function = self.eviction_selection[layer_no]
        # print(f"Layer {layer_no} evicting {evict_number} blocks.")
        # Should have beend removed from list in evict_selection.
        total_layer_cnt = self.num_storage_layers
        assert total_layer_cnt == len(self._num_blocks)
        write_to_next_layer: List[KVBlockTrieNode] = []
        evicted_nodes = []
        # callback on evict: Heap update && color info && leaf info of parent && evict candidates of parent and self.
        max_end_original_present_next_layer_time = 0.0
        max_fir_original_present_next_layer_time = 0.0
        max_original_present_next_layer_interval = 0.0
        for _ in range(evict_number):
            # Should update evict set inside the loop, or when evict_number if large, it will run out of candidates.
            self.add_evict(layer_no)
            evict_node: KVBlockTrieNode = evict_selection_function(layer_no)
            # print(f"Evicting node {evict_node.id} with color {evict_node.color}")
            assert evict_node is not None
            assert not evict_node.do_not_evict
            assert not evict_node.is_in_evict_heap
            evicted_nodes.append(evict_node)
            assert evict_node.parent is not None
            original_color = evict_node.color
            assert original_color == layer_no
            if original_color == total_layer_cnt - 1:
                evict_node.evict_to_discard()
                self.delete_node(evict_node)
            else:
                if not evict_node.check_if_color_present(layer_no + 1):
                    assert not no_write
                    # Should write.
                    # The evicted block is not in the next layer before.
                    write_to_next_layer.append(evict_node)
                    # Do not evict again now.
                    # Cos will mark it later.
                    evict_node.set_do_not_evict(True)
                    if evict_node.is_in_evict_heap:
                        evict_node.remove_from_evict_heap()
                else:
                    end_next, fir_next = evict_node.get_ready_time(layer_no + 1)
                    max_end_original_present_next_layer_time = max(max_end_original_present_next_layer_time, end_next)
                    max_fir_original_present_next_layer_time = max(max_fir_original_present_next_layer_time, fir_next)
                    assert self._num_layers > 0
                    next_interval = (end_next - fir_next) / (self._num_layers - 1) if self._num_layers > 1 else 0.0
                    max_original_present_next_layer_interval = max(max_original_present_next_layer_interval, next_interval)
                # Call this even if already inside, to delete from current layer.
                # Safe to call with inf, inf even if not a write to next layer.
                already_inside, wno = evict_node.evict_to_lower_location(float("inf"), float("inf"))
                assert wno == layer_no + 1
            # This is for updating eviction candidates.
            # Update ending time later.

            # Present in next layer otherwise, do not need to write.

        blocks_needed_for_next_layer = len(write_to_next_layer)
        evict_end_time = 0.0
        evict_first_layer_end_time = 0.0
        evict_per_layer_time_interval = 0.0
        write_end_time = 0.0
        write_first_layer_end_time = 0.0
        write_per_layer_time_interval = 0.0
        if blocks_needed_for_next_layer > 0:
            if blocks_needed_for_next_layer > self.available_blocks_with_buffer(layer_no + 1):
                more_space_in_next_layer = blocks_needed_for_next_layer - self.available_blocks_with_buffer(layer_no + 1)
                evict_end_time, evict_first_layer_end_time, evict_per_layer_time_interval = \
                self._evict_blocks(layer_no + 1, more_space_in_next_layer, timestamp, no_write)
            else:
                evict_end_time = timestamp
                evict_first_layer_end_time = timestamp
                evict_per_layer_time_interval = 0.0
        
            # Then count the time for the actual write.
            write_channel: Channel = self._channel_from_this_to_lower[layer_no][1]
            write_end_time, write_first_layer_end_time, write_per_layer_time_interval = \
            write_channel.transmit(blocks_needed_for_next_layer * self._block_size, evict_end_time, self._num_layers)
            # Then use that space in lower layer.
            # NOTE: used_blocks changed.
            self._used_blocks[layer_no + 1] += blocks_needed_for_next_layer
            assert self._used_blocks[layer_no + 1] <= self._num_blocks[layer_no + 1]
        else:
            write_end_time = timestamp
            write_first_layer_end_time = timestamp
            write_per_layer_time_interval = 0.0

        if layer_no < self.num_storage_layers - 1:
            self.add_insert(layer_no + 1, len(write_to_next_layer))
        # Insert to next layer.
        # Note that callback_on_evict only 
        # Every evicted but not deleted node has color change.

        # Update the ready time.
        for node in write_to_next_layer:
            # If not, discarded or already in before.
            node.set_storage_layer_info_timestamps(layer_no + 1, write_end_time, write_first_layer_end_time)
            node.set_do_not_evict(False)
            node.callback_on_possible_leaf_change()
        
        # Mark freed space.
        # NOTE: used_blocks changed.
        self._used_blocks[layer_no] -= evict_number
        # Color should have been evicted.
        # self.check_size_consistency()
        # for i in range(self.num_storage_layers):
        #     self._get_size(i)
        space_av_end_time = max(write_end_time, max_end_original_present_next_layer_time)
        space_av_first_layer_end_time = max(write_first_layer_end_time, max_fir_original_present_next_layer_time)
        space_av_per_layer_time_interval = max(write_per_layer_time_interval, max_original_present_next_layer_interval)
        return space_av_end_time, space_av_first_layer_end_time, space_av_per_layer_time_interval
    
    def available_blocks(self, layer_no: int):
        return self._num_threshold_blocks[layer_no] - self._used_blocks[layer_no]
    
    def available_blocks_with_buffer(self, layer_no: int):
        return self._num_blocks[layer_no] - self._used_blocks[layer_no]
    
    def read_buffer_blocks(self, layer_no: int):
        return self._num_blocks[layer_no] - self._num_threshold_blocks[layer_no]

    def mark_active_block_number(self, layer_no: int, diffsize: int):
        assert diffsize >= 0
        self._active_blocks[layer_no] += diffsize

    # block_number free blocks.
    # Also mark the space as used in this function.
    def synced_acquire_space(self, layer_no, block_number: int, timestamp, no_synced_write: bool, use_buf: bool, 
                             allow_lower_write:bool = False) -> Tuple[float, float, float]:
        # self.check_size_consistency()
        availble_blocks_without_buffer = self.available_blocks(layer_no)
        # print(f"Layer {layer_no} available blocks: {availble_blocks_without_buffer}")
        # print(f"Layer {layer_no} acquires {block_number} blocks.")
        # Print the function name of the traceback above this function.
        # print(f"Function name: {inspect.currentframe().f_back.f_code.co_name}")
        assert availble_blocks_without_buffer >= 0, f"layer: {layer_no}, {availble_blocks_without_buffer} < 0"
        if availble_blocks_without_buffer >= block_number:
            # Use that space.
            # NOTE: used_blocks changed.
            self._used_blocks[layer_no] += block_number
            assert self._used_blocks[layer_no] <= self._num_blocks[layer_no], f"One: {self._used_blocks[layer_no]} > {self._num_blocks[layer_no]}"
            assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
            return timestamp, timestamp, 0.0
        else:
            # Need to evict.
            should_make_space = block_number - availble_blocks_without_buffer
            if use_buf:
                assert no_synced_write, f"{layer_no} {block_number} {timestamp} {no_synced_write} {use_buf}"
                # A seperate call to make use of read buffer.
                # This buffer is used for loading into layer_no.
                read_buffer_blocks = self.read_buffer_blocks(layer_no)
                assert read_buffer_blocks >= 0
                assert read_buffer_blocks >= should_make_space, f"{read_buffer_blocks} < {should_make_space}"
                # Make another call outside for remaining ones.
                # No synced time.
                # NOTE: used_blocks changed.
                assert self._used_blocks[layer_no] + block_number <= self._num_blocks[layer_no]
                # assert self._used_blocks[layer_no] <= self._num_blocks[layer_no], f"Two: {self._used_blocks[layer_no]} > {self._num_blocks[layer_no]}"
                # And evict to make space for read buffer.
                # FIXME: Assume that the read buffer is available before the end execution of last batch.
                # FIXME: Assume that write through is used, so evict time to get back read buffer is 0.
                # The purpose here is to mark the space as free, so that read buffer is back.
                # Also swap the space for read buffer.

                # FIXME: Does this guarantee those loaded in here not evicted immediately?
                if not allow_lower_write:
                    evict_end, evict_fir, evict_per = self._evict_blocks(layer_no, should_make_space, timestamp, no_synced_write)
                    # FIXME: What makes it non-zero??!!
                    # assert evict_per == 0
                    # assert evict_end == timestamp
                    # assert evict_fir == timestamp
                else:
                    evict_end, evict_fir, evict_per = self._evict_blocks(layer_no, should_make_space, timestamp, False)
                assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
                # self._get_size(layer_no)
                # Put it here, then consistent when entering the _evict_blocks.

                self._used_blocks[layer_no] += block_number
                return evict_end, evict_fir, evict_per
            else:
                evict_end, evict_fir, evict_per = self._evict_blocks(layer_no, should_make_space, timestamp, no_synced_write)
                # NOTE: used_blocks changed.
                self._used_blocks[layer_no] += block_number
                assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
                if not self._used_blocks[layer_no] <= self._num_blocks[layer_no]:
                    print(f"block_number: {block_number}, should_make_space: {should_make_space}, availble_blocks_without_buffer: {availble_blocks_without_buffer}")
                assert self._used_blocks[layer_no] <= self._num_blocks[layer_no], f"Three: Adding {should_make_space}, then {self._used_blocks[layer_no]} > {self._num_blocks[layer_no]}"
                return evict_end, evict_fir, evict_per



    

    def release_active_block_for_highest_level(self, number):
        # print(f"Release {number} blocks for active blocks.")
        assert len(self._used_blocks) > 0
        assert len(self._num_blocks) > 0
        assert number >= 0
        assert self._num_threshold_blocks[0] >= self._used_blocks[0]
        # Only possible to exceed threshold in synced_acquire_space temporarily.
        # Assume write through and fast enough write speed to make space available.
        assert self._num_blocks[0] >= self._used_blocks[0]
        assert number <= self._used_blocks[0], f"{number} > {self._used_blocks[0]}"
        # NOTE: used_blocks changed.
        self._used_blocks[0] -= number
        self._active_blocks[0] -= number
        assert self._used_blocks[0] >= 0
        assert self._active_blocks[0] >= 0
        return
    
    def direct_free_memory(self, layer_no, number):
        assert len(self._used_blocks) > 0
        assert len(self._num_blocks) > 0
        assert number >= 0
        assert self._num_threshold_blocks[layer_no] >= self._used_blocks[layer_no]
        assert self._num_blocks[layer_no] >= self._used_blocks[layer_no]
        assert number <= self._used_blocks[layer_no], f"{number} > {self._used_blocks[layer_no]}"
        # NOTE: used_blocks changed.
        self._used_blocks[layer_no] -= number
        assert self._used_blocks[layer_no] >= 0
        return
    
    
    def lookup(self, query: List[KVBlock], timestamp):
        # print(f"Lookup called with {len(query)} blocks.")
        previous_hit_blocks = set()
        retval = [self.root]
        current_node = self.root
        for block in query:
            next_node: KVBlockTrieNode = current_node.children.get(tuple(block.tokens), None)
            if next_node is None:
                break
            retval.append(next_node)
            assert next_node not in previous_hit_blocks
            previous_hit_blocks.add(next_node)
            # Update evict list.
            # FIXME: Now only LRU.
            # NOTE: Only move if already inside.
            # NOTE: Just update in this order, should maintain leaf sets.
            if timestamp >= 0.0:
                # print(f"timestamp of node {next_node.id} updated with {timestamp} for lookup.")
                next_node.timestamp_update(timestamp)
            current_node = next_node
        # len(retval) - 1 should be the number of full blocks found.
        # Then extend it if allow_reorder_kv_blocks is True.
        if self.allow_reorder_kv_blocks:
            original_hit_len = len(retval) - 1
            for i in range(original_hit_len, len(query)):
                # print(f"Reorder lookup: {i}")
                the_block = tuple(query[i].tokens)
                if the_block not in self.block_tokens_to_nodes:
                    break
                else:
                    min_heap: Heap = self.block_tokens_to_nodes[the_block]
                    min_node = min_heap.get_min()
                    if min_node in previous_hit_blocks:
                        # print(f"\n\n\nWARNING: {min_node.id} already in previous_hit_blocks.\n\n\n\n")
                        # NOTE: A workaround to avoid the space counting mess because of the same block
                        # appearing multiple times in hit_trace.
                        break
                    else:
                        previous_hit_blocks.add(min_node)
                    assert min_node is not None # Or the heap should not exsist.
                    # NOTE: Previously only add_ref for 
                    retval.append(min_node)
                    self._blocks_extended_by_reorder += 1
        return retval

    def insert_with_prepared_new_node(self, the_node: KVBlockTrieNode, layer_no: int, timestamp, layer_timestamp):
        # Mark space outside.
        # Even if already in, update list.
        self.add_insert(layer_no)
        last_node: KVBlockTrieNode = the_node.parent
        assert the_node.tokens not in last_node.children
        assert the_node.color == -1
        last_node.add_child_node(the_node)
        assert not the_node.is_in_evict_heap
        # Acquire space.
        # NOTE: used_blocks changed.
        self._used_blocks[layer_no] += 1
        assert self._used_blocks[layer_no] <= self._num_blocks[layer_no]
        assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
        # Will be inside the evict set if possible.
        # Color and timestamp also updated here.
        the_node.callback_on_insert_into_gpu(timestamp, layer_timestamp)
        return the_node
    
    def insert_into_gpu_from_active_block_with_original_in_cpu(self, the_node: KVBlockTrieNode, 
                                                               timestamp, layer_timestamp):
        # Timestamp to tell when this ready(after execution).
        self.add_insert(0)
        # Fetch to higher, but without cost.
        # Space should be acquired, cos active blocks released before.
        # Just fetch this node, not parent.
        the_node.fetch_to_higher_location(timestamp, layer_timestamp)
        # NOTE: used_blocks changed.
        self._used_blocks[0] += 1
        assert self._used_blocks[0] <= self._num_blocks[0]
        assert self._used_blocks[0] <= self._num_threshold_blocks[0]
        return the_node
    # Allow [True, False, True]
    def insert_into_gpu_from_active_block_original_in_disk_allow_tft(self, the_node: KVBlockTrieNode, timestamp, 
                                                                     layer_timestamp):
        self.add_insert(0)
        the_node.callback_on_switch_to_layer0_tft(timestamp, layer_timestamp)
        self._used_blocks[0] += 1
        assert self._used_blocks[0] <= self._num_blocks[0]
        assert self._used_blocks[0] <= self._num_threshold_blocks[0]
        return the_node

    def insert_one_node(self, child_node):
        color = child_node.color
        assert color >= 0
        # Space has been acquired outside.
        # self._used_blocks[color] += 1
        assert self._used_blocks[color] <= self._num_threshold_blocks[color]
        child_node.parent.insert_a_new_child_node(child_node)
    
    def evict_selection_lru(self, layer_no: int):
        # Choose one block, which is one edge.
        # Node is representing the edge.
        assert len(self.evict_candidates) > layer_no
        if len(self.evict_candidates[layer_no]) == 0:
            # Print the leaves of the Trie.
            '''            print("Leaves of the Trie:")
            self._print_leaves(layer_no)
            print("Color of the Trie:")
            self._print_color(layer_no)
            print("Leaves of 0 color")
            self._print_leaves(0)
            print("Color of 0 color")
            self._print_color(0)
            print("Leaves of 2 color")
            self._print_leaves(2)
            print("Color of 2 color")
            self._print_color(2)
            '''
            self._print_present(layer_no)
            self._get_size(layer_no)
        assert len(self.evict_candidates[layer_no]) > 0, f"Layer {layer_no} has no evict candidates. heap_size: {self._evict_heap_size}\nnum_blocks: {self._num_blocks}\nused_blocks: {self._used_blocks}\nthreshold_blocks: {self._num_threshold_blocks}"
        evicted_node: KVBlockTrieNode = self.evict_candidates[layer_no][0]
        if self.num_storage_layers == 1:
            assert len(evicted_node.children) == 0
            # print(f"Evict timestamp {evicted_node.evict_timestamp}")
        assert evicted_node is not None
        assert evicted_node.position_in_evict_heap == 0
        original_size = len(self.evict_candidates[layer_no])
        # print(f"{evicted_node.id} Remove from heap on selection.")
        evicted_node.remove_from_evict_heap()
        assert len(self.evict_candidates[layer_no]) == original_size - 1
        return evicted_node
    
    def eviction_selection_pgdsf(self, layer_no: int):
        evicted_node: KVBlockTrieNode = self.evict_selection_lru(layer_no)
        assert evicted_node.last_pgdsf_priority_update_layer == layer_no
        if evicted_node.evict_timestamp[0] > self._pgdsf_clock[layer_no]:
            original_clock = self._pgdsf_clock[layer_no]
            self._pgdsf_clock[layer_no] = evicted_node.evict_timestamp[0]
            # print(f"Layer {layer_no} clock updated from {original_clock} to {self._pgdsf_clock[layer_no]}")
        return evicted_node
    
    def add_insert(self, layer_no: int, add_number: int = 1):
        self._insert_cnt[layer_no] += add_number
    def add_evict(self, layer_no: int, add_number: int = 1):
        self._evict_cnt[layer_no] += add_number
    def add_hit(self, layer_no: int, add_number: int = 1):
        self._hit_cnt[layer_no] += add_number

    def append_layer(self, num_blocks, read_thput, 
                     write_thput, evict_policy: str, evict_op: str, threshold_blocks: int, space_per_token_per_layer):
        # print(f"num_blocks type: {type(num_blocks)}")
        self._num_blocks.append(num_blocks)
        self._used_blocks.append(0)
        self._num_threshold_blocks.append(threshold_blocks)
        self._evict_heap_size.append(0)
        self._active_blocks.append(0)
        # print(f"layer: {len(self._num_blocks) - 1}, num_blocks: {num_blocks}, threshold_blocks: {threshold_blocks}")
        self._channel_from_this_to_lower.append((Channel(read_thput, space_per_token_per_layer), Channel(write_thput, space_per_token_per_layer)))
        if evict_policy.upper() == "LRU":
            self.eviction_selection.append(self.evict_selection_lru)
        elif evict_policy.upper() == "PGDSF":
            self.eviction_selection.append(self.eviction_selection_pgdsf)
            self._no_real_timestamp_update[len(self._num_blocks) - 1] = True
            self._is_pgdsf_eviction[len(self._num_blocks) - 1] = True
            self._pgdsf_clock.append(0)
        elif evict_policy.upper() == "MRU":
            raise NotImplementedError("MRU not implemented.")
        else:
            raise ValueError(f"Unknown eviction policy: {evict_policy}")
        self._evict_candidates.append([]) # A heap.
        self._evict_op.append(evict_op)
        self._insert_cnt.append(0)
        self._evict_cnt.append(0)
        self._hit_cnt.append(0)

    
    def get_channel(self, layer_no) -> Tuple[Channel, Channel]:
        return self._channel_from_this_to_lower[layer_no]

    @property
    def num_storage_layers(self):
        return len(self._num_blocks)
    @property
    def num_blocks(self):
        return self._num_blocks
    @property
    def used_blocks(self):
        return self._used_blocks
    
    @property
    def evict_candidates(self):
        return self._evict_candidates
    
    @property
    def no_real_timestamp_update(self):
        return self._no_real_timestamp_update
    
    @property
    def is_pgdsf_eviction(self):
        return self._is_pgdsf_eviction
    
    @property
    def pgdsf_clock(self):
        return self._pgdsf_clock
    
    @property
    def allow_reorder_kv_blocks(self):
        return self._allow_reorder_kv_blocks

def get_general_kv_block_size():
    global general_kv_block_size
    return general_kv_block_size