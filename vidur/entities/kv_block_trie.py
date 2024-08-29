from typing import List, Tuple
import heapq
from vidur.entities.communications import Channel
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

# TODO: Add some way to pin it to some layer(more than refcnt which prevents discarding).

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
        self._storage_layer_info = [(False, -1.0, -1.0), (False, -1.0, -1.0), (False, -1.0, -1.0)]
        self._children_color_cnt = [0, 0, 0]

        self._tokens_parent_to_here = tokens_parent_to_here
        self._kvtrie = kvtrie

    @property
    def color(self):
        idx = 0
        for layer_info in self._storage_layer_info:
            if layer_info[0]:
                return idx
            idx += 1
        return -1

    def get_ready_time(self, layer_no: int):
        assert layer_no >= 0
        assert layer_no < self._kvtrie.num_storage_layers
        assert self._storage_layer_info[layer_no][0]
        return self._storage_layer_info[layer_no][1], self._storage_layer_info[layer_no][2]

    @property
    def position_in_evict_heap(self):
        return self._position_in_evict_heap
    
    @property
    def is_leaf(self):
        # Color of itself && color of children.
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
        assert self.evict_timestamp[0] >= 0, f"{self.evict_timestamp}, id: {self.id}"
        assert self.evict_timestamp[1] < 0
        color = self.color
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
                if color == self._kvtrie.num_storage_layers - 1:
                    if self._refcnt == 0:
                        # print(f"{self.id} Add to heap on leaf refcnt == 0.")
                        self.add_self_to_evict_heap()
                else:
                    # print(f"{self.id} Add to heap on is leaf.")
                    self.add_self_to_evict_heap()

    # Interal states on evictions, color, leaves.
    def callback_on_fetch(self, from_no: int, to_no: int, timestamp, layer_timestamp):
        assert from_no == to_no + 1
        assert self._storage_layer_info[from_no][0]
        assert not self._storage_layer_info[to_no][0]
        assert self.color == from_no # Should only be called when needing a fetch.
        # Anyway, need to remove from evict list of from_no layer cos color changed.
        # print(f"{self.id} Remove from heap on fetch from {from_no} to {to_no}.")
        # Color changed, but perhaps it is not in heap before.
        if self.is_in_evict_heap:
            self.remove_from_evict_heap()
        self.timestamp_update(timestamp)
        # Do not delete storage layer info of from_no, it is still there.
        # Update color.
        self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
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
            self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
            # No color change.
            # So no leaf change.
    
    def timestamp_update(self, new_access_time):
        color = self.color
        assert color >= 0, f"{self.id} {color}"
        assert self._storage_layer_info[color][0]
        old_access_time = self._evict_timestamp[0]
        self._evict_timestamp = (new_access_time, -self.depth)
        if self.is_in_evict_heap:
            assert old_access_time <= new_access_time
            self.sift_down_evict_heap(False)
        
        
        


    def callback_on_evict(self, from_no: int, to_no: int, timestamp, layer_timestamp):
        assert from_no == to_no - 1
        assert self._storage_layer_info[from_no][0]
        assert self.color == from_no
        # Because always evict higher layers of the same copy.
        # Guranteed by the fact that it is only in evict list of its color.
        # Anyway, need to remove from evict list of from_no layer cos color changed.
        # print(f"{self.id} Remove from heap on evict from {from_no} to {to_no}.")
        assert not self.is_in_evict_heap # Should have been removed from list ssin evict_selection.
        # self.remove_from_evict_heap()
        # Update color.
        self._storage_layer_info[from_no] = (False, -1.0, -1.0)
        # Swap out once always true.
        if self._storage_layer_info[to_no][0]:
            assert self._storage_layer_info[to_no][1] <= timestamp
            assert self._storage_layer_info[to_no][2] <= layer_timestamp
        else:
            self._storage_layer_info[to_no] = (True, timestamp, layer_timestamp)
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
        assert self._storage_layer_info[layer_no][0]
        assert self._storage_layer_info[layer_no][1] == float("inf")
        assert self._storage_layer_info[layer_no][2] == float("inf")
        self._storage_layer_info[layer_no] = (True, timestamp, layer_timestamp)
    
    def callback_on_insert_into_gpu(self, timestamp, layer_timestamp):
        # Only called when not in GPU before.
        # A new node.
        assert self.color == -1
        assert not self._storage_layer_info[0][0]
        # Should be new in trie.
        assert len(self.children) == 0
        assert not self.is_in_evict_heap # A new node, updated later in callback_on_possible_leaf_change.
        self._storage_layer_info[0] = (True, timestamp, layer_timestamp)
        # Update timestamps.
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
    
    def remove_ref(self):
        self._refcnt -= 1
        assert self._refcnt >= 0
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
    def __init__(self, layer_pipeline: bool, block_size, num_layers: int):
        self.root = KVBlockTrieNode(None, tuple(), self, 0)
        # Configurations
        self._block_size = block_size
        global general_kv_block_size
        if general_kv_block_size is not None:
            assert general_kv_block_size == block_size
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

        atexit.register(self.dump_stats)

    

    def dump_stats(self):
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

    def delete_node(self, node: KVBlockTrieNode):
        assert node.color == -1 # Call evict_discard just before it.
        assert node.refcnt == 0
        assert not node.is_in_evict_heap # Removed on selection.
        assert len(node.children) == 0, f"{node.id} -len{len(node.children)}-> {next(iter(node.children.values())).id}"
        del node.parent.children[node.tokens]
        # Update of parent in evict_to_discard.
        
    
    def _evict_blocks(self, layer_no, evict_number, timestamp, no_write: bool) -> Tuple[float, float, float]:
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
        for _ in range(evict_number):
            # Should update evict set inside the loop, or when evict_number if large, it will run out of candidates.
            self.add_evict(layer_no)
            evict_node: KVBlockTrieNode = evict_selection_function(layer_no)
            assert evict_node is not None
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
                # Call this even if already inside, to delete from current layer.
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
            if blocks_needed_for_next_layer > (self._num_blocks[layer_no + 1] - self._used_blocks[layer_no + 1]):
                more_space_in_next_layer = blocks_needed_for_next_layer - (self._num_blocks[layer_no + 1] - self._used_blocks[layer_no + 1])
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
        
        # Mark freed space.
        self._used_blocks[layer_no] -= evict_number
        return write_end_time, write_first_layer_end_time, write_per_layer_time_interval
    
    def available_blocks(self, layer_no: int):
        return self._num_threshold_blocks[layer_no] - self._used_blocks[layer_no]
    
    def available_blocks_with_buffer(self, layer_no: int):
        return self._num_blocks[layer_no] - self._used_blocks[layer_no]
    
    def read_buffer_blocks(self, layer_no: int):
        return self._num_blocks[layer_no] - self._num_threshold_blocks[layer_no]

    # block_number free blocks.
    # Also mark the space as used in this function.
    def synced_acquire_space(self, layer_no, block_number: int, timestamp, no_write: bool, use_buf: bool) -> Tuple[float, float, float]:
        availble_blocks_without_buffer = self.available_blocks(layer_no)
        # print(f"Layer {layer_no} available blocks: {availble_blocks_without_buffer}")
        # print(f"Layer {layer_no} acquires {block_number} blocks.")
        # Print the function name of the traceback above this function.
        # print(f"Function name: {inspect.currentframe().f_back.f_code.co_name}")
        assert availble_blocks_without_buffer >= 0
        if availble_blocks_without_buffer >= block_number:
            # Use that space.
            self._used_blocks[layer_no] += block_number
            assert self._used_blocks[layer_no] <= self._num_blocks[layer_no], f"One: {self._used_blocks[layer_no]} > {self._num_blocks[layer_no]}"
            assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
            return timestamp, timestamp, 0.0
        else:
            # Need to evict.
            should_make_space = block_number - availble_blocks_without_buffer
            synced_evict_make_space = should_make_space
            if use_buf:
                assert no_write, f"{layer_no} {block_number} {timestamp} {no_write} {use_buf}"
                # A seperate call to make use of read buffer.
                # This buffer is used for loading into layer_no.
                read_buffer_blocks = self.read_buffer_blocks(layer_no)
                assert read_buffer_blocks >= 0
                # Should always be available when use_buf is True.
                assert read_buffer_blocks >= should_make_space
                # Make another call outside for remaining ones.
                # No synced time.
                self._used_blocks[layer_no] += block_number
                assert self._used_blocks[layer_no] <= self._num_blocks[layer_no], f"Two: {self._used_blocks[layer_no]} > {self._num_blocks[layer_no]}"
                # And evict to make space for read buffer.
                # FIXME: Assume that the read buffer is available before the end execution of last batch.
                # FIXME: Assume that write through is used, so evict time to get back read buffer is 0.
                # The purpose here is to mark the space as free, so that read buffer is back.
                # Swap the space for read buffer.
                evict_end, evict_fir, evict_per = self._evict_blocks(layer_no, should_make_space, timestamp, no_write)
                assert evict_per == 0
                assert evict_end == timestamp
                assert evict_fir == timestamp
                assert self._used_blocks[layer_no] <= self._num_threshold_blocks[layer_no]
                return evict_end, evict_fir, evict_per
            else:
                evict_end, evict_fir, evict_per = self._evict_blocks(layer_no, should_make_space, timestamp, no_write)
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
        self._used_blocks[0] -= number
        return
    
    
    def lookup(self, query: List[KVBlock], timestamp):
        # print(f"Lookup called with {len(query)} blocks.")
        retval = [self.root]
        current_node = self.root
        for block in query:
            next_node: KVBlockTrieNode = current_node.children.get(tuple(block.tokens), None)
            if next_node is None:
                break
            retval.append(next_node)
            # Update evict list.
            # FIXME: Now only LRU.
            # NOTE: Only move if already inside.
            # NOTE: Just update in this order, should maintain leaf sets.
            if timestamp >= 0.0:
                next_node.timestamp_update(timestamp)
            current_node = next_node
        # len(retval) - 1 should be the number of full blocks found.
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
        self._used_blocks[0] += 1
        assert self._used_blocks[0] <= self._num_blocks[0]
        assert self._used_blocks[0] <= self._num_threshold_blocks[0]
        return the_node
    
    '''
    def insert_into_gpu_from_active_block_original_in_disk(self, the_node: KVBlockTrieNode,timestamp, layer_timestamp):
        self.add_insert(0)
        self.add_insert(1)
        last_node: KVBlockTrieNode = the_node.parent
        assert the_node.tokens in last_node.children
        assert the_node.color == 2 # Originally in disk.
        # Return the node to make it write to CPU later.
        last_node.fetch_to_higher_location(timestamp, layer_timestamp)
        last_node.fetch_to_higher_location(timestamp, layer_timestamp)
    '''


    
    def evict_selection_lru(self, layer_no: int):
        # Choose one block, which is one edge.
        # Node is representing the edge.
        assert len(self.evict_candidates) > layer_no
        assert len(self.evict_candidates[layer_no]) > 0, f"Layer {layer_no} has no evict candidates."
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
        # print(f"layer: {len(self._num_blocks) - 1}, num_blocks: {num_blocks}, threshold_blocks: {threshold_blocks}")
        self._channel_from_this_to_lower.append((Channel(read_thput, space_per_token_per_layer), Channel(write_thput, space_per_token_per_layer)))
        # FIXME: Now only LRU.
        assert evict_policy.upper() == "LRU"
        if evict_policy.upper() == "LRU":
            self.eviction_selection.append(self.evict_selection_lru)
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

def get_general_kv_block_size():
    global general_kv_block_size
    return general_kv_block_size