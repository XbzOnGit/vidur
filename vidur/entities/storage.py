from typing import List, Union, Tuple
import random
import atexit
from collections import deque

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)

# TODO URGENT, FIXME: Should keep these following variables && functions in a class.
# Cos there might be multiple storage instances.
# For now, only one, so it is fine.

lru_timestamp = 0
lru_from_token_tuple_to_timestamp = {}
fifo_queue_tokens = deque()
random_evict_tokens = set()
insert_cnt = 0
evict_cnt = 0

def print_stats():
    logger.info("Storage stats:")
    logger.info(f"insert_cnt: {insert_cnt}")
    logger.info(f"evict_cnt: {evict_cnt}")

atexit.register(print_stats)


class TrieNode:
    def __init__(self, parent) -> None:
        self.children = {}
        self.is_end = False
        self.parent = parent

def _find_any_end_from_node(current: TrieNode) -> Tuple[List[str], TrieNode]:
    suffix_list = []
    while not current.is_end:
        # Get the first key, value from children.
        # But do not delete it.
        token, child = next(iter(current.children.items()))
        suffix_list.append(token)
        current = child
    return suffix_list, current

def Trie_insert_lru_callback(key: List[str], node) -> None:
    # Insert callback is called only when no hit or not a full hit.
    # So key iself must be the longest.
    # key == full_list
    global lru_timestamp
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    lru_from_token_tuple_to_timestamp[tuple(key)] = lru_timestamp

def Trie_lookup_lru_callback(key: List[str], node) -> None:
    global lru_timestamp
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    lru_from_token_tuple_to_timestamp[tuple(full_list)] = lru_timestamp

    
def Trie_delete_lru_callback(key: List[str], root) -> None:
    del lru_from_token_tuple_to_timestamp[tuple(key)]


# Only insert and delete changes FIFO.
def Trie_insert_fifo_callback(key: List[str], node) -> None:
    global fifo_queue_tokens
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    fifo_queue_tokens.append(key)

def Trie_delete_fifo_callback(key: List[str], root) -> None:
    global fifo_queue_tokens
    # Called on eviction, so the key should be the first one.
    assert len(fifo_queue_tokens) > 0
    assert fifo_queue_tokens[0] == key
    fifo_queue_tokens.popleft()

def Trie_insert_random_callback(key: List[str], node) -> None:
    global random_evict_tokens
    # FIXME: Just for debugging, remove the redundant find and check.
    suffix_list, _ = _find_any_end_from_node(node)
    full_list = key + suffix_list
    assert len(full_list) == len(key)
    random_evict_tokens.add(tuple(key))

def Trie_delete_random_callback(key: List[str], root) -> None:
    global random_evict_tokens
    assert tuple(key) in random_evict_tokens
    random_evict_tokens.remove(tuple(key))

class Trie:
    def __init__(self, lookup_callback, insert_callback, delete_callback) -> None:
        self.root = TrieNode(None)
        self.lookup_callback = lookup_callback
        self.insert_callback = insert_callback
        self.delete_callback = delete_callback
    
    
    def lookup(self, key: List[str]) -> Union[List[str], None]:
        current = self.root
        found_tokens = []
        for token in key:
            if token not in current.children:
                break
            found_tokens.append(token)
            current = current.children[token]
        if self.lookup_callback is not None and len(found_tokens) > 0:
            # print("lookup_callback")
            self.lookup_callback(found_tokens, current)
        if len(found_tokens) == 0:
            return None
        return found_tokens

    def insert(self, key: List[str]):
        current = self.root
        for token in key:
            if token not in current.children:
                current.children[token] = TrieNode(current)
            current = current.children[token]
        current.is_end = True
        if self.insert_callback is not None:
            # print("insert_callback")
            # Called after trie is updated.
            self.insert_callback(key, current)
        
    def delete(self, key: List[str]):
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
            self.delete_callback(key, self.root)


def prefix_lookup(key: List[str], trie: Trie) -> Union[List[str], None]:
    return trie.lookup(key)

def prefix_insert(key: List[str], trie: Trie):
    trie.insert(key)

def prefix_delete(key: List[str], trie: Trie):
    trie.delete(key)

def lru_evict_selection() -> List[str]:
    assert len(lru_from_token_tuple_to_timestamp) > 0
    min_timestamp = None
    min_key = None
    for key, timestamp in lru_from_token_tuple_to_timestamp.items():
        if min_timestamp is None or timestamp < min_timestamp:
            min_timestamp = timestamp
            min_key = key
    min_key = list(min_key)
    return min_key

def random_evict_selection() -> List[str]:
    global lru_from_token_tuple_to_timestamp
    return random.choice(list(random_evict_tokens))

def FIFO_evict_selection() -> List[str]:
    global fifo_queue_tokens
    return fifo_queue_tokens[0]

def discard_evict_op(any):
    # Just nothing, no overhead.
    return 0

class Storage(BaseEntity):
    def __init__(self, storage_id: int, capacity: int, look_up_type: str, 
                 evict_type: str, evict_op: str, is_highest: bool, space_per_token: int) -> None:
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
        if look_up_type == "prefix":
            self._lookup = prefix_lookup
            self._lookup_insert = prefix_insert
            self._lookup_delete = prefix_delete
            if evict_type == "lru":
                self._lookup_storage = Trie(Trie_lookup_lru_callback, Trie_insert_lru_callback, Trie_delete_lru_callback)
            elif evict_type == "fifo":
                self._lookup_storage = Trie(None, Trie_insert_fifo_callback, Trie_delete_fifo_callback)
            elif evict_type == "random":
                self._lookup_storage = Trie(None, Trie_insert_random_callback, Trie_delete_random_callback)
            else:
                self._lookup_storage = Trie(None, None, None)
        if evict_type == "lru":
            self._evict_selection = lru_evict_selection
            self._evict_storage = self._lookup_storage
        elif evict_type == "fifo":
            self._evict_selection = FIFO_evict_selection
            self._evict_storage = self._lookup_storage
        elif evict_type == "random":
            self._evict_selection = random_evict_selection
            self._evict_storage = self._lookup_storage
        if evict_op == "discard":
            self._evict_op = discard_evict_op
        assert self._lookup is not None
        assert self._evict_selection is not None
        assert self._evict_op is not None
    
    def lookup(self, key:List[str]) -> Union[List[str], None]:
        global lru_timestamp
        lru_timestamp += 1
        item = self._lookup(key, self._lookup_storage)
        return item
    
    def insert(self, key:List[str]):
        global lru_timestamp
        global insert_cnt
        global evict_cnt
        lru_timestamp += 1
        insert_cnt += 1
        # Just lookup first.
        lookup_result = self.lookup(key)
        if lookup_result is not None:
            if len(lookup_result) == len(key):
                return 0
        # Return an extra overhead other than just saving.
        extra_overhead = 0
        needed_size = len(key) * self._space_per_token
        assert needed_size <= self._capacity
        while needed_size + self._used_space > self._capacity:
            evicted_one = self._evict_selection()
            self._lookup_delete(evicted_one, self._lookup_storage)
            extra_overhead += self._evict_op(evicted_one)
            freed_size = len(evicted_one) * self._space_per_token
            self._used_space -= freed_size
            evict_cnt += 1
            assert self._used_space >= 0
        self._used_space += needed_size
        # Insert is only called when key can hit but not fully hit.
        # After eviction.
        self._lookup_insert(key, self._lookup_storage)
        return extra_overhead


        
