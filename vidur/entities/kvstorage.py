from typing import List, Union, Tuple
import random
import atexit
from collections import deque

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)

class TrieNode:
    def __init__(self, parent, storage_id) -> None:
        self.children = {}
        self.is_end = False
        self.parent = parent
        self.storage_id = storage_id

def _find_any_end_from_node(current: TrieNode) -> Tuple[List[str], TrieNode]:
    suffix_list = []
    while not current.is_end:
        # Get the first key, value from children.
        # But do not delete it.
        token, child = next(iter(current.children.items()))
        suffix_list.append(token)
        current = child
    return suffix_list, current

# key, aux0(node), aux1(lru_timestamp), aux2(evict_stroage)
def Trie_insert_lru_callback(key: List[str], node, lru_timestamp, lru_from_token_tuple_to_timestamp) -> None:
    # Insert callback is called only when no hit or not a full hit.
    # So key iself must be the longest.
    # key == full_list
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    # print(f"insert_lru_callback: {node.storage_id}")
    # print(len(lru_from_token_tuple_to_timestamp))
    lru_from_token_tuple_to_timestamp[tuple(key)] = lru_timestamp
    # print(len(lru_from_token_tuple_to_timestamp))
    # print("\n\n")

def Trie_lookup_lru_callback(key: List[str], node, lru_timestamp, lru_from_token_tuple_to_timestamp) -> None:
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    # print(f"lookup_lru_callback: {node.storage_id}")
    # print(len(lru_from_token_tuple_to_timestamp))
    lru_from_token_tuple_to_timestamp[tuple(full_list)] = lru_timestamp
    # print(len(lru_from_token_tuple_to_timestamp))
    # print("\n\n")


def Trie_delete_lru_callback(key: List[str], lru_from_token_tuple_to_timestamp, aux1, aux2) -> None:
    # print(f"delete_lru_callback: {aux1.storage_id}")
    # print(len(lru_from_token_tuple_to_timestamp))
    del lru_from_token_tuple_to_timestamp[tuple(key)]
    # print(len(lru_from_token_tuple_to_timestamp))
    # print("\n\n")


# Only insert and delete changes FIFO.
def Trie_insert_fifo_callback(key: List[str], node, lru_timestamp, fifo_queue_tokens) -> None:
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    fifo_queue_tokens.append(key)

def Trie_delete_fifo_callback(key: List[str], fifo_queue_tokens, aux1, aux2) -> None:
    # Called on eviction, so the key should be the first one.
    assert len(fifo_queue_tokens) > 0
    if fifo_queue_tokens[0] == key:
        fifo_queue_tokens.popleft()
    else:
        # pinned.
        assert len(fifo_queue_tokens) > 1
        assert fifo_queue_tokens[1] == key
        push_back_item = fifo_queue_tokens.popleft()
        fifo_queue_tokens.popleft()
        fifo_queue_tokens.appendleft(push_back_item)

def Trie_insert_random_callback(key: List[str], node, lru_timestamp, random_evict_tokens) -> None:
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    random_evict_tokens.add(tuple(key))

def Trie_delete_random_callback(key: List[str], random_evict_tokens, aux1, aux2) -> None:
    assert tuple(key) in random_evict_tokens
    random_evict_tokens.remove(tuple(key))

class Trie:
    def __init__(self, lookup_callback, insert_callback, delete_callback, storage_id) -> None:
        self.root = TrieNode(None, storage_id)
        self.lookup_callback = lookup_callback
        self.insert_callback = insert_callback
        self.delete_callback = delete_callback
        self.storage_id = storage_id
    
    
    def lookup(self, key: List[str], lru_timestamp, evict_storage) -> Union[List[str], None]:
        current = self.root
        found_tokens = []
        for token in key:
            if token not in current.children:
                break
            found_tokens.append(token)
            current = current.children[token]
        if self.lookup_callback is not None and len(found_tokens) > 0:
            # print("lookup_callback")
            self.lookup_callback(found_tokens, current, lru_timestamp, evict_storage)
        if len(found_tokens) == 0:
            return None, None
        suffix_list, _ = _find_any_end_from_node(current)
        full_list = found_tokens + suffix_list
        return found_tokens, full_list

    def insert(self, key: List[str], lru_timestamp, evict_storage):
        assert tuple(key) not in evict_storage
        current = self.root
        for token in key:
            if token not in current.children:
                current.children[token] = TrieNode(current, self.storage_id)
            current = current.children[token]
        current.is_end = True
        if self.insert_callback is not None:
            # print("insert_callback")
            # Called after trie is updated.
            self.insert_callback(key, current, lru_timestamp, evict_storage)
        
    def delete(self, key: List[str], evict_storage):
        assert len(key) > 0
        current = self.root
        for token in key:
            if token not in current.children:
                raise ValueError("Key not found")
            current = current.children[token]
        assert current.is_end
        current.is_end = False
        # Delete all the nodes that are not end and have no children.
        keyidx = len(key) - 1
        while current != self.root:
            if current.is_end or len(current.children) > 0:
                break
            parent = current.parent
            del parent.children[key[keyidx]]
            keyidx -= 1
            current = parent
        if self.delete_callback is not None:
            # print("delete_callback")
            self.delete_callback(key, evict_storage, self.root, None)


def prefix_lookup(key: List[str], trie: Trie, lru_timestamp, evict_storage):
    return trie.lookup(key, lru_timestamp, evict_storage)

def prefix_insert(key: List[str], trie: Trie, lru_timestamp, evict_storage):
    trie.insert(key, lru_timestamp, evict_storage)

def prefix_delete(key: List[str], trie: Trie, evict_storage):
    trie.delete(key, evict_storage)

# evict_storage
def lru_evict_selection(lru_from_token_tuple_to_timestamp, pinned_set) -> List[str]:
    assert len(lru_from_token_tuple_to_timestamp) > 0
    min_timestamp = None
    min_key = None
    for key, timestamp in lru_from_token_tuple_to_timestamp.items():
        if tuple(key) not in pinned_set:
            if min_timestamp is None or timestamp < min_timestamp:
                min_timestamp = timestamp
                min_key = key
    assert min_key is not None
    min_key = list(min_key)
    return min_key

def random_evict_selection(random_evict_tokens, pinned_set) -> List[str]:
    new_set = random_evict_tokens - pinned_set
    assert len(new_set) > 0
    return random.choice(list(new_set))

def FIFO_evict_selection(fifo_queue_tokens, pinned_set) -> List[str]:
    assert len(fifo_queue_tokens) > 0
    if fifo_queue_tokens[0] in pinned_set:
        assert len(fifo_queue_tokens) > 1
        return fifo_queue_tokens[1]
    return fifo_queue_tokens[0]

def discard_evict_op(storage, key: List[str]):
    return 0.0

def write_evict_op(storage, key: List[str]):
    lower_storage = storage.get_lower_storage()
    lookup_result, _ = lower_storage.lookup(key)
    if lookup_result is not None and len(lookup_result) == len(key):
        if storage.get_swap_out_once():
            # No overhead, no swap.
            # timestamp has been updated.
            saved_time = storage.get_kv_size(key) / storage.thput_write_lower()
            storage.add_swap_out_once_saved_time(saved_time)
            return 0.0
        else:
            # Still no eviction on lower storage, no real insertion.
            # Just pretent to write to exactly original place.
            # Timestamp has been updated on lookup.
            return storage.get_kv_size(key) / storage.thput_write_lower()
    kv_size = storage.get_kv_size(key)
    overhead = 0.0
    if lower_storage.get_free_space() < kv_size:
        need_more_free_space = kv_size - lower_storage.get_free_space()
        overhead += lower_storage.evict(need_more_free_space)
    overhead += kv_size / storage.thput_write_lower()
    # print("Storage insert called in write_evict_op")
    assert lower_storage.insert(key) == 0
    return overhead
    


def choose_longer(no_in_orders, keys, aux):
    assert len(keys) > 0
    return_idx = 0
    max_key = keys[0]
    for i in range(1, len(keys)):
        if len(keys[i]) > len(max_key):
            max_key = keys[i]
            return_idx = i
    return return_idx
    

def choose_higher(no_in_orders, keys, aux):
    assert len(no_in_orders) > 0
    min_no = no_in_orders[0]
    return_idx = 0
    for i in range(1, len(no_in_orders)):
        if no_in_orders[i] < min_no:
            min_no = no_in_orders[i]
            return_idx = i
    return return_idx
    

# On replica_stage level.
class StorageController(BaseEntity):
    # TODO: Return the full key that is hit.
    def __init__(self, choice_strategy: str):
        # Now assuming linear.
        self._lookup_order = []
        self._kv_size_per_token = None
        if choice_strategy == "longer":
            self._choose_strategy = choose_longer
        elif choice_strategy == "higher":
            self._choose_strategy = choose_higher
        else:
            raise ValueError("Invalid choice strategy")
    def set_kv_size_per_token(self, kv_size_per_token):
        # This is also per stage.
        self._kv_size_per_token = kv_size_per_token
    def add_storage(self, storage):
        self._lookup_order.append(storage)
    def lookup(self, key: List[str], fetch_if_lower: bool, strictly_larger_than_this=0):
        assert len(self._lookup_order) > 0
        keys = []
        full_keys = []
        no_in_orders = []
        for i in range(len(self._lookup_order)):
            lookup_result, full_key = self._lookup_order[i].lookup(key)
            if lookup_result is not None:
                keys.append(lookup_result)
                full_keys.append(full_key)
                no_in_orders.append(i)
        if len(keys) == 0:
            return None, 0.0, None
        else:
            select_idx = self._choose_strategy(no_in_orders, keys, None)
            select_layer_no = no_in_orders[select_idx]
            if len(keys[select_idx]) <= strictly_larger_than_this:
                return None, 0.0, None
            if select_layer_no == 0:
                # 0 always the one and only GPU memory.
                return keys[select_idx], 0.0, full_keys[select_idx]
            else:
                if not fetch_if_lower:
                    return None, 0.0, None
                else:
                    # From select_layer_no to 0.
                    # Find upper layer eviction and its size.
                    # Call evict on the LOWER LAYER for that size.
                    # Then evict on the upper layer(including insert).
                    # Then insert on the upper layer.
                    overhead = self._fetch_cache(select_layer_no, 0, key)
                    return keys[select_idx], overhead, full_keys[select_idx]
    
    # Do not need pin on computation, now the memory is seperated.
    # It should include a copy.
    
    def _fetch_cache(self, from_layer_no, to_layer_no, key: List[str]):
        assert from_layer_no >= to_layer_no
        if from_layer_no == to_layer_no:
            return 0.0
        assert len(key) > 0
        assert self._kv_size_per_token is not None
        kv_size = self._kv_size_per_token * len(key)
        current_layer_no = from_layer_no
        overhead = 0.0
        while current_layer_no != to_layer_no:
            # current_layer to current_layer - 1
            current_storage = self._lookup_order[current_layer_no]
            put_storage = self._lookup_order[current_layer_no - 1]
            if put_storage.get_free_space() < kv_size:
                current_storage.pin(key)
                more_free_space = kv_size - put_storage.get_free_space()
                overhead += put_storage.evict(more_free_space)
                current_storage.unpin(key)
                # Do not free space in current_storage.
                
            # Now put_storage has enough space.
            # put <== current
            # current is lower.
            overhead += kv_size / put_storage.thput_read_lower() # Read from current layer.
            # print("Storage insert called in _fetch_cache")
            assert put_storage.insert(key) == 0
            current_layer_no -= 1
        return overhead
    
    def insert(self, key: List[str], layer_no=0):
        assert layer_no < len(self._lookup_order)
        # lookup before insert.
        lookup_result, _ = self._lookup_order[layer_no].lookup(key)
        if lookup_result is not None and len(lookup_result) == len(key):
            return 0.0
        return self._lookup_order[layer_no].insert(key)
        




class Storage(BaseEntity):
    def __init__(self, storage_id: int, capacity: int, look_up_type: str, 
                 evict_type: str, evict_op: str, is_highest: bool, space_per_token: int, swap_out_once: bool) -> None:
        self._storage_id = storage_id
        self._capacity = capacity
        self._used_space = 0
        self._is_highest = is_highest # For if it is level that can be used directly.
        self._space_per_token = space_per_token
        self._lookup = None
        self._lookup_storage = None
        self._lookup_insert = None
        self._lookup_delete = None
        self._evict_selection = None
        self._evict_op = None
        self._evict_storage = None
        self._lru_timestamp = 0
        self._insert_cnt = 0
        self._evict_cnt = 0
        self._total_eviction_time = 0
        self._total_fetch_time = 0
        self._swap_out_once = swap_out_once
        self._swap_out_once_saved_time = 0.0
        # Make sure that the same KV cache only once.
        self._pinned_tokens = set()
        if look_up_type == "prefix":
            self._lookup = prefix_lookup
            self._lookup_insert = prefix_insert
            self._lookup_delete = prefix_delete
            if evict_type == "lru":
                self._lookup_storage = Trie(Trie_lookup_lru_callback, Trie_insert_lru_callback, Trie_delete_lru_callback, storage_id)
            elif evict_type == "fifo":
                self._lookup_storage = Trie(None, Trie_insert_fifo_callback, Trie_delete_fifo_callback, storage_id)
            elif evict_type == "random":
                self._lookup_storage = Trie(None, Trie_insert_random_callback, Trie_delete_random_callback, storage_id)
            else:
                self._lookup_storage = Trie(None, None, None)
        if evict_type == "lru":
            self._evict_selection = lru_evict_selection
            self._evict_storage = {}
        elif evict_type == "fifo":
            self._evict_selection = FIFO_evict_selection
            self._evict_storage = deque()
        elif evict_type == "random":
            self._evict_selection = random_evict_selection
            self._evict_storage = set()
        if evict_op == "discard":
            self._evict_op = discard_evict_op
        elif evict_op == "cpu_write":
            self._evict_op = write_evict_op
        assert self._lookup is not None
        assert self._evict_selection is not None
        assert self._evict_op is not None
        atexit.register(self._print_stats)
    
    def _print_stats(self):
        logger.info(f"Storage stats with id {self._storage_id}:")
        logger.info(f"insert_cnt: {self._insert_cnt}")
        logger.info(f"evict_cnt: {self._evict_cnt}")
        logger.info(f"swap_out_saved_time: {self._swap_out_once_saved_time}")
    
    def add_swap_out_once_saved_time(self, saved_time):
        self._swap_out_once_saved_time += saved_time

    def lookup(self, key:List[str]):
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
        self._lru_timestamp += 1
        # print(f"storage_id lookup: {self._storage_id}")
        return self._lookup(key, self._lookup_storage, self._lru_timestamp, self._evict_storage)
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
    
    def insert(self, key:List[str]):
        # print(f"storage_id insert: {self._storage_id}")
        assert tuple(key) not in self._evict_storage, "Key already in evict storage"
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
        self._lru_timestamp += 1
        self._insert_cnt += 1
        # Just lookup first.
        lookup_result, _ = self.lookup(key)
        if lookup_result is not None:
            if len(lookup_result) == len(key):
                return 0
        # Return an extra overhead other than just saving.
        extra_overhead = 0
        needed_size = len(key) * self._space_per_token
        assert needed_size <= self._capacity
        free_space = self.get_free_space()
        need_more_space = needed_size - free_space
        if need_more_space > 0:
            self.evict(need_more_space)
        # print(f"Insertion used space of storage {self._storage_id}: from {self._used_space} to {self._used_space + needed_size}\n\n\n")
        self._used_space += needed_size
        self._lookup_insert(key, self._lookup_storage, self._lru_timestamp, self._evict_storage)
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
        return extra_overhead
    
    def evict(self, evict_size: int):
        # print(f"storage_id evict: {self._storage_id}")
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
        self._lru_timestamp += 1
        overhead = 0.0
        while evict_size > 0:
            # print(f"Select eviction in {self._storage_id}")
            evicted_one = self._evict_selection(self._evict_storage, self._pinned_tokens)
            self._lookup_delete(evicted_one, self._lookup_storage, self._evict_storage)
            # _evict_op is possible to trigger further evictions.
            overhead += self._evict_op(self, evicted_one)
            freed_size = len(evicted_one) * self._space_per_token
            self._used_space -= freed_size
            # print(f"Eviction used space of storage {self._storage_id}: from {self._used_space + freed_size} to {self._used_space}\n\n\n")
            self._evict_cnt += 1
            evict_size -= freed_size
        assert self._used_space >= 0
        # check_evict_space_consistency(self._evict_storage, self._used_space, self._space_per_token, self._lookup_storage)
        return overhead

    def get_swap_out_once(self) -> bool:
        return self._swap_out_once

    def set_lower_storage(self, lower_storage, read_from_thput, write_to_thput):
        self._lower_storage = lower_storage
        self._read_from_lower_thput = read_from_thput
        self._write_to_lower_thput = write_to_thput
    
    def get_lower_storage(self):
        return self._lower_storage

    def thput_read_lower(self):
        return self._read_from_lower_thput
    
    def thput_write_lower(self):
        return self._write_to_lower_thput
    
    def get_kv_size(self, key: List[str]) -> int:
        return len(key) * self._space_per_token
    
    def get_free_space(self):
        assert self._used_space <= self._capacity
        return self._capacity - self._used_space
    
    def pin(self, key: List[str]):
        self._pinned_tokens.add(tuple(key))
    
    def unpin(self, key: List[str]):
        self._pinned_tokens.remove(tuple(key))