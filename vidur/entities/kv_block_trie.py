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
    def __init__(self, parent, tokens_parent_to_here: Tuple[str], kvtrie):
        global kv_trie_node_id
        self.id = kv_trie_node_id
        kv_trie_node_id += 1

        self.record_of_evictlist_ops = []
        self.parent = parent
        self.children = {}
        # Currently always remove from highest level first.
        # And color is also the highest.
        # A heap should work.
        self._colors = [] # color + timestamp
        # The edge is a block, from parent to this node.
        self._refcnt = 0 # Also the edge from parent to this node.
        # FIXME: Currently at most 3 layers.
        self.evict_list_prev = [None, None, None]
        self.evict_list_next = [None, None, None]
        self._children_color_cnt = [0, 0, 0]

        self._tokens_parent_to_here = tokens_parent_to_here
        self._kvtrie = kvtrie

    def mark_initial_location(self, initial_location_id: int, initial_timestamp, initial_layer_timestamp):
        assert len(self._colors) == 0
        self._colors.append((initial_location_id, initial_timestamp, initial_layer_timestamp))
    
    def add_location(self, location: int, timestamp, layer_timestamp):
        heapq.heappush(self._colors, (location, timestamp, layer_timestamp))
    
    
    def add_child_node(self, child_node, layer_no, timestamp, layer_timestamp):
        assert child_node.tokens not in self.children
        assert len(child_node._colors) == 0
        self.children[child_node.tokens] = child_node
        child_node.mark_initial_location(layer_no, timestamp, layer_timestamp)
        
        
    def already_has_block(self, block: KVBlock):
        return block.tokens in self.children
    
    def fetch_to_higher_location(self, complete_timestamp, complete_layer_timestamp):
        assert len(self._colors) > 0
        assert self._colors[0][0] > 0
        higher_level_no = self._colors[0][0] - 1
        heapq.heappush(self._colors, (higher_level_no, complete_timestamp, complete_layer_timestamp))
        return higher_level_no
    def push_to_lower_location(self, complete_timestamp, complete_layer_timestamp):
        assert len(self._colors) > 0
        lower_level_no = self._colors[0][0] + 1
        already_inside = False
        if len(self._colors) > 1 and self._colors[1][0] == lower_level_no:
            already_inside = True
        assert not already_inside
        heapq.heappush(self._colors, (lower_level_no, complete_timestamp, complete_layer_timestamp))
        return lower_level_no
    def evict_to_lower_location(self, complete_timestamp, complete_layer_timestamp):
        assert len(self._colors) > 0
        lower_level_no = self._colors[0][0] + 1
        already_inside = False
        if len(self._colors) > 1 and self._colors[1][0] == lower_level_no:
            already_inside = True
        heapq.heappop(self._colors)
        if not already_inside:
            heapq.heappush(self._colors, (lower_level_no, complete_timestamp, complete_layer_timestamp))
        return already_inside, lower_level_no
    def get_color(self):
        return self._colors[0]
    def check_if_color_present(self, location_id):
        for color in self._colors:
            if color[0] == location_id:
                return color
        return None
    def add_ref(self):
        assert self._refcnt >= 0
        if self._refcnt == 0:
            last_layer_no = self._kvtrie.num_storage_layers - 1
            assert last_layer_no >= 0
            inside = False
            for color in self._colors:
                if color[0] == last_layer_no:
                    inside = True
                    break
            if inside:
                #assert self.evict_list_next[last_layer_no] is not None or self.evict_list_prev[last_layer_no] is not None, \
                #f"{self.id}\n{self.record_of_evictlist_ops}"
                # It can be both None, if it is the only one in the list.
                self._kvtrie.remove_from_evict_list(last_layer_no, self)
            
        self._refcnt += 1
    def remove_ref(self):
        self._refcnt -= 1
        assert self._refcnt >= 0
        if self._refcnt == 0:
            # print(f"{self} refcnt drops to 0.")
            # If refcnt drops to 0, allow to evict from last layer.
            last_layer_no = self._kvtrie.num_storage_layers - 1
            assert last_layer_no >= 0
            inside = False
            for color in self._colors:
                if color[0] == last_layer_no:
                    inside = True
                    break
            if inside:
                assert self.evict_list_next[last_layer_no] is None
                assert self.evict_list_prev[last_layer_no] is None
                self._kvtrie.push_to_evict_list_tail(last_layer_no, self)
                # It can be both None, if it is the only one in the list.
                # assert self.evict_list_next[last_layer_no] is not None or self.evict_list_prev[last_layer_no] is not None

    # Trie insert.
    # Trie delete.
    # Color upgrade.
    # Color degrade.

    # Should update children color count && evict list(leaf condition).
    '''
    def leaf_update(self, op_type: int, layerno_before: int):
        if op_type == 0:
            assert layerno_before == 0
            parent_node = self.parent
            assert parent_node is not None
            parent_node._children_color_cnt[0] += 1
            if parent_node._children_color_cnt[0] == 1:
                # Remove from evict list because it is not a leaf anymore.
                # But should be leaf before.
                assert parent_node.evict_list_next[0] is not None or parent_node.evict_list_prev[0] is not None
                self._kvtrie.remove_from_evict_list(0, parent_node)
        elif op_type == 1:
            assert layerno_before == self._kvtrie.num_storage_layers - 1
            parent_node = self.parent
            assert parent_node is not None
            parent_node._children_color_cnt[layerno_before] -= 1
            assert parent_node._children_color_cnt[layerno_before] >= 0
            if parent_node._children_color_cnt[layerno_before] == 0:
                # Make parent a leaf.
    '''

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
        self.root = KVBlockTrieNode(None, tuple(), self)
        # Configurations
        self._block_size = block_size
        self._num_layers = num_layers
        self._num_blocks = []
        self._num_shadow_buffer_blocks = []
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
        assert node.get_color()[0] == self.num_storage_layers - 1
        assert node.refcnt == 0
        assert node.evict_list_prev[self.num_storage_layers - 1] is None
        assert node.evict_list_next[self.num_storage_layers - 1] is None
        if len(node.children) != 0:
            print(f"{node.id} -len{len(node.children)}-> {next(iter(node.children.values())).id}")
            # self._print_trie()
        assert len(node.children) == 0, f"{node.id} -len{len(node.children)}-> {next(iter(node.children.values())).id}"
        del node.parent.children[node.tokens]
    
    def _evict_blocks(self, layer_no, evict_number, timestamp, no_write: bool) -> Tuple[float, float, float]:
        assert evict_number >= 0
        if evict_number == 0:
            return timestamp, timestamp, 0.0
        # Synced evict.
        # print(f"Evicting {evict_number} blocks from layer {layer_no}")
        # Select blocks.
        # Modify trie to record eviction.
        # free space.
        # Return end_time, first_layer_end_time, per_layer_time_interval.
        evict_selection = self.eviction_selection[layer_no]
        # Should have beend removed from list in evict_selection.
        total_layer_cnt = len(self._evict_candidates)
        write_to_next_layer = []
        for _ in range(evict_number):
            self.add_evict(layer_no)
            evict_node: KVBlockTrieNode = evict_selection(layer_no)
            if layer_no == total_layer_cnt - 1:
                # Must be discard.
                assert evict_node.refcnt == 0
                # Discard.
                # Remove from evict list, it must be inside.
                self.delete_node(evict_node)
            else:
                # Just to lower layer.
                # Check if a write is needed.
                if evict_node.check_if_color_present(layer_no + 1) is None:
                    if no_write:
                        next_next_layer_no = layer_no + 2
                        if next_next_layer_no < total_layer_cnt:
                            assert evict_node.check_if_color_present(next_next_layer_no) is not None
                        else:
                            raise ValueError("No write but no color in lower layers.")
                    else:
                        # Should write.
                        # The evicted block is not in the next layer before.
                        write_to_next_layer.append(evict_node)

                            
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

        # else no write && no eviction.
        # Mark the eviction in trie.
        for evict_node in write_to_next_layer:
            self.add_insert(layer_no + 1)
            already_inside, wno = evict_node.evict_to_lower_location(write_end_time, write_first_layer_end_time)
            if evict_node.refcnt == 0:
                self.push_to_evict_list_tail(wno, evict_node)
            assert wno == layer_no + 1
            assert not already_inside
        # Mark freed space.
        self._used_blocks[layer_no] -= evict_number
        return write_end_time, write_first_layer_end_time, write_per_layer_time_interval
        
    # block_number free blocks.
    # Return end_fir
    def synced_make_space(self, layer_no, block_number: int, timestamp, no_write: bool, use_shadow_buf: bool) -> Tuple[float, float, float]:
        if self._num_blocks[layer_no] - self._used_blocks[layer_no] >= block_number:
            return timestamp, timestamp, 0.0
        else:
            # Need to evict.
            should_make_space = block_number - (self._num_blocks[layer_no] - self._used_blocks[layer_no])
            synced_evict_make_space = should_make_space
            if use_shadow_buf:
                # This buffer is used for loading into layer_no.
                synced_evict_make_space = max(0, (should_make_space - self._num_shadow_buffer_blocks[layer_no]))
            # print(f"Evicting {synced_evict_make_space} blocks from layer {layer_no}")
            end_time, end_fir_time, per_time =  self._evict_blocks(layer_no, synced_evict_make_space, timestamp, no_write)
            async_evict_block_num = should_make_space - synced_evict_make_space
            # print(f"Async Evicted {synced_evict_make_space} blocks from layer {layer_no}")
            # print(f"{type(async_evict_block_num)}")
            if async_evict_block_num > 0:
                self._evict_blocks(layer_no, async_evict_block_num, end_time, no_write) # But do not return this.
                # This is to get back the shadow buffer space.
            assert self._num_blocks[layer_no] - self._used_blocks[layer_no] >= block_number
            return end_time, end_fir_time, per_time



    
    # NOTE: Leave space before calling this.
    def acquire_active_block_for_highest_level(self, number):
        # print(f"Acquire {number} blocks for active blocks.")
        assert len(self._used_blocks) > 0
        assert len(self._num_blocks) > 0
        assert number >= 0
        assert self._num_blocks[0] >= self._used_blocks[0]
        assert number <= self._num_blocks[0]
        avaiable_block_num = self._num_blocks[0] - self._used_blocks[0]
        assert avaiable_block_num >= number
        # Then acquire them.
        self._used_blocks[0] += number
        return

    def release_active_block_for_highest_level(self, number):
        # print(f"Release {number} blocks for active blocks.")
        assert len(self._used_blocks) > 0
        assert len(self._num_blocks) > 0
        assert number >= 0
        assert self._num_blocks[0] >= self._used_blocks[0]
        assert number <= self._used_blocks[0]
        self._used_blocks[0] -= number
        return
    
    def remove_from_evict_list(self, layer_no: int, node: KVBlockTrieNode):
        # print(f"{node} removed from evict list {layer_no}")
        node.record_of_evictlist_ops.append((layer_no, "remove"))
        if self._evict_candidates[layer_no][0] == None:
            assert self._evict_candidates[layer_no][1] == None
            raise ValueError("No evict list.")
        assert self._evict_candidates[layer_no][1] is not None
        if self._evict_candidates[layer_no][0] == node and self._evict_candidates[layer_no][1] == node:
            self._evict_candidates[layer_no] = (None, None)
            node.evict_list_prev[layer_no] = None
            node.evict_list_next[layer_no] = None
            return
        prev_head = self._evict_candidates[layer_no][0]
        prev_tail = self._evict_candidates[layer_no][1]
        prev_prev = node.evict_list_prev[layer_no]
        prev_next = node.evict_list_next[layer_no]
        if node.evict_list_prev[layer_no] is not None:
            node.evict_list_prev[layer_no].evict_list_next[layer_no] = node.evict_list_next[layer_no]
        if node.evict_list_next[layer_no] is not None:
            node.evict_list_next[layer_no].evict_list_prev[layer_no] = node.evict_list_prev[layer_no]
        node.evict_list_prev[layer_no] = None
        node.evict_list_next[layer_no] = None
        if prev_head == node:
            assert prev_tail != node
            self._evict_candidates[layer_no] = (prev_next, prev_tail)
        if prev_tail == node:
            assert prev_head != node
            self._evict_candidates[layer_no] = (prev_head, prev_prev)
        assert node.evict_list_prev[layer_no] is None
        assert node.evict_list_next[layer_no] is None
        return
        
    def push_to_evict_list_front(self, layer_no: int, node: KVBlockTrieNode):
        assert node.evict_list_next[layer_no] is None
        assert node.evict_list_prev[layer_no] is None
        node.record_of_evictlist_ops.append((layer_no, "push to front"))
        # print(f"{node} pushed to front of evict list {layer_no}")
        prev_head = self._evict_candidates[layer_no][0]
        prev_tail = self._evict_candidates[layer_no][1]
        if prev_head is None:
            assert prev_tail is None
            self._evict_candidates[layer_no] = (node, node)
        else:
            assert prev_tail is not None
            node.evict_list_next[layer_no] = prev_head
            prev_head.evict_list_prev[layer_no] = node
            self._evict_candidates[layer_no] = (node, prev_tail)

    def push_to_evict_list_tail(self, layer_no: int, node: KVBlockTrieNode):
        assert node.evict_list_next[layer_no] is None
        assert node.evict_list_prev[layer_no] is None
        node.record_of_evictlist_ops.append((layer_no, "push to tail"))
        # print(f"{node} pushed to tail of evict list {layer_no}")
        prev_head = self._evict_candidates[layer_no][0]
        prev_tail = self._evict_candidates[layer_no][1]
        if prev_tail is None:
            assert prev_head is None
            self._evict_candidates[layer_no] = (node, node)
        else:
            assert prev_head is not None
            node.evict_list_prev[layer_no] = prev_tail
            prev_tail.evict_list_next[layer_no] = node
            self._evict_candidates[layer_no] = (prev_head, node)
            
    
    def move_to_evict_list_front_in_all_layers(self, number_of_layers: int, node: KVBlockTrieNode):
        for i in range(number_of_layers):
            if node.evict_list_next[i] is None and node.evict_list_prev[i] is None:
                continue
            self.remove_from_evict_list(i, node)
            self.push_to_evict_list_front(i, node)

    def move_to_evict_list_tail_in_all_layers(self, number_of_layers: int, node: KVBlockTrieNode):
        for i in range(number_of_layers):
            if node.evict_list_next[i] is None and node.evict_list_prev[i] is None:
                # print(f"{node} not in evict list {i}")
                continue
            self.remove_from_evict_list(i, node)
            self.push_to_evict_list_tail(i, node)
    
    def lookup(self, query: List[KVBlock]):
        # print(f"Lookup called with {len(query)} blocks.")
        retval = [self.root]
        current_node = self.root
        for block in query:
            next_node = current_node.children.get(tuple(block.tokens), None)
            if next_node is None:
                break
            retval.append(next_node)
            # Update evict list.
            # FIXME: Now only LRU.
            # NOTE: Only move if already inside.

            # NOTE: Update in reverse order.
            # self.move_to_evict_list_tail_in_all_layers(len(self._evict_candidates), next_node)
            current_node = next_node
        # len(retval) - 1 should be the number of full blocks found.
        reverse_idx = len(retval) - 1
        while reverse_idx > 0:
            node = retval[reverse_idx]
            self.move_to_evict_list_front_in_all_layers(len(self._evict_candidates), node)
            reverse_idx -= 1
        return retval
    
    # If already inside, move it.
    # If not, push it.
    def evict_list_try_push_in_reverse_order(self, trace: List[KVBlockTrieNode], layer_no: int):
        assert self.eviction_selection[layer_no] == self.evict_selection_lru
        reverse_idx = len(trace) - 1
        while reverse_idx > 0:
            node = trace[reverse_idx]
            if node == self.root:
                assert reverse_idx == 0
                break
            if node.evict_list_next[layer_no] is not None or node.evict_list_prev[layer_no] is not None:
                self.remove_from_evict_list(layer_no, node)
            self.push_to_evict_list_tail(layer_no, node)
            reverse_idx -= 1


    # NOTE: Evict list outside this function.
    def insert_with_prepared_node_and_space(self, the_node: KVBlockTrieNode, layer_no: int, timestamp, layer_timestamp):
        # Mark space outside.
        # Even if already in, update list.
        self.add_insert(layer_no)
        last_node: KVBlockTrieNode = the_node.parent
        if the_node.tokens in last_node.children:
            the_node = last_node.children[the_node.tokens]
            assert not the_node.check_if_color_present(layer_no)
            assert the_node.evict_list_prev[layer_no] is None
            assert the_node.evict_list_next[layer_no] is None
            # NOTE: Now update outside.
            self.push_to_evict_list_tail(layer_no, the_node)
            self.move_to_evict_list_tail_in_all_layers(len(self._evict_candidates), the_node)
            self._used_blocks[layer_no] += 1 # Still should count, cos in another layer.
            assert self._used_blocks[layer_no] <= self._num_blocks[layer_no]
            the_node.add_location(layer_no, timestamp, layer_timestamp)
            return the_node # Return the one inserted before.
        else:
            assert len(the_node._colors) == 0
            #the_node.mark_initial_location(layer_no, timestamp, layer_timestamp)
            # Insert into trie.
            #last_node.children[the_node.tokens] = the_node
            last_node.add_child_node(the_node, layer_no, timestamp, layer_timestamp)
            assert the_node.evict_list_prev[layer_no] is None
            assert the_node.evict_list_next[layer_no] is None
            self.push_to_evict_list_tail(layer_no, the_node)
            self.move_to_evict_list_tail_in_all_layers(len(self._evict_candidates), the_node)
            self._used_blocks[layer_no] += 1
            assert self._used_blocks[layer_no] <= self._num_blocks[layer_no]
            return the_node

    
    def evict_selection_lru(self, layer_no: int):
        # Choose one block, which is one edge.
        # Node is representing the edge.
        assert len(self._evict_candidates) > layer_no
        head, tail = self._evict_candidates[layer_no]
        # print(f"Selecting eviction on layer {layer_no}")
        # current = head
        # while current is not None:
        #     print(f"{current.id}, {current.record_of_evictlist_ops}")
        #     current = current.evict_list_next[layer_no]

        assert head is not None, f"{self._evict_candidates[layer_no]}, {layer_no}"
        assert tail is not None, f"{self._evict_candidates[layer_no]}, {layer_no}"
        assert head.evict_list_prev[layer_no] is None, f"{self._evict_candidates[layer_no]}, {layer_no}"
        assert tail.evict_list_next[layer_no] is None, f"{self._evict_candidates[layer_no]}, {layer_no}"
        self.remove_from_evict_list(layer_no, head)
        # if head.id == 1:
        #     print(f"LRU selected {head.id}, {head.record_of_evictlist_ops}")
        return head
    '''
    def evict_selection_mru(self, layer_no: int):
        assert len(self._evict_candidates) > layer_no
        head, tail = self._evict_candidates[layer_no]
        assert head is not None
        assert tail is not None
        assert head.evict_list_prev[layer_no] is None
        assert tail.evict_list_next[layer_no] is None
        self.remove_from_evict_list(layer_no, tail)
        return tail
    '''
    
    def add_insert(self, layer_no: int):
        self._insert_cnt[layer_no] += 1
    def add_evict(self, layer_no: int):
        self._evict_cnt[layer_no] += 1
    def add_hit(self, layer_no: int):
        self._hit_cnt[layer_no] += 1

    def append_layer(self, num_blocks, read_thput, 
                     write_thput, evict_policy: str, evict_op: str, shadow_buffer_fraction: float, space_per_token_per_layer):
        shadow_buffer_num_blocks = int(num_blocks * shadow_buffer_fraction)
        # print(f"num_blocks type: {type(num_blocks)}")
        self._num_blocks.append(num_blocks - shadow_buffer_num_blocks)
        self._num_shadow_buffer_blocks.append(shadow_buffer_num_blocks)
        self._used_blocks.append(0)
        print(f"layer: {len(self._num_blocks) - 1}, num_blocks: {num_blocks}, shadow_buffer_num_blocks: {shadow_buffer_num_blocks}")
        self._channel_from_this_to_lower.append((Channel(read_thput, space_per_token_per_layer), Channel(write_thput, space_per_token_per_layer)))
        # FIXME: Now only LRU.
        assert evict_policy.upper() == "LRU"
        if evict_policy.upper() == "LRU":
            self.eviction_selection.append(self.evict_selection_lru)
        elif evict_policy.upper() == "MRU":
            raise NotImplementedError("MRU not implemented.")
        else:
            raise ValueError(f"Unknown eviction policy: {evict_policy}")
        self._evict_candidates.append((None, None))
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