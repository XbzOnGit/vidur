from vidur.logger import init_logger
from vidur.types import StorageInfoType
from typing import Optional, Dict, Set

logger = init_logger(__name__)


class CompressLevelManager:
    def __init__(self):
        self.to_rate = {}
        self.to_quality = {}
        self.encode_cost = {}
        self.decode_cost = {}
        self._finish_config = False
    def finish_config(self):
        self._finish_config = True
    def add_level(self, level_no: int, rate: float, quality, encode_cost: float, decode_cost: float):
        if self._finish_config:
            return
        # NOTE: Always use level_no == 0 for no compression.
        if level_no == 0:
            assert rate == 1.0, f"Rate should be 1.0 for no compression(level 0), but got {rate}"
            assert encode_cost == 0.0, f"Encode cost should be 0.0 for no compression(level 0), but got {encode_cost}"
            assert decode_cost == 0.0, f"Decode cost should be 0.0 for no compression(level 0), but got {decode_cost}"
        self.to_rate[level_no] = rate
        self.to_quality[level_no] = quality
        self.encode_cost[level_no] = encode_cost
        self.decode_cost[level_no] = decode_cost
        # Second per token.
    def multiply_rate_from_to(self, from_level: int, to_level: int) -> float:
        return self.to_rate[to_level] / self.to_rate[from_level]
    def get_decompress_time(self, token_number: int, compress_level: int) -> float:
        return token_number * self.decode_cost[compress_level]
    def get_compress_time(self, token_number: int, compress_level: int) -> float:
        return token_number * self.encode_cost[compress_level]
    def get_compress_rate(self, compress_level: int) -> float:
        return self.to_rate[compress_level]
    def get_all(self):
        return_list = []
        for level_no in self.to_rate.keys():
            return_list.append((level_no, 
                                self.to_rate[level_no], 
                                self.to_quality[level_no], 
                                self.encode_cost[level_no], 
                                self.decode_cost[level_no]))
        return return_list
    def get_max_quality_drop(self) -> float:
        max_q = max(self.to_quality.values())
        min_q = min(self.to_quality.values())
        return max_q - min_q
    def get_level_set(self) -> set:
        return set(self.to_rate.keys())


compress_level_manager = CompressLevelManager()

def get_compress_level_manager():
    return compress_level_manager


class KVObjectQuery:
    def __init__(self, prefix_hash: str, hash_value: str, 
                 prefix_token_len: int,
                 chunk_token_len: int, size: int):
        self._prefix_hash = prefix_hash
        self._hash_value = hash_value
        self._prefix_token_len = prefix_token_len
        self._chunk_token_len = chunk_token_len
        self._size = size
    @property
    def prefix_token_len(self):
        return self._prefix_token_len
    @property
    def prefix_hash(self):
        return self._prefix_hash
    @property
    def hash_value(self):
        return self._hash_value
    @property
    def chunk_token_len(self):
        return self._chunk_token_len
    @property
    def size(self):
        return self._size

kv_obj_idx = 0
class KVObjectMetadata:
    def __init__(self, prefix_hash: str, hash_value: str, 
                 prefix_token_len: int,
                 chunk_token_len: int, size: int, compression_level: int, 
                 status: StorageInfoType,
                 device_idx,
                 associated_event):
        global kv_obj_idx
        self._id = kv_obj_idx
        kv_obj_idx += 1
        self._prefix_hash = prefix_hash
        self._hash_value = hash_value
        self._prefix_token_len = prefix_token_len
        self._chunk_token_len = chunk_token_len
        self._size = size
        self._compression_level = compression_level
        self._status = status
        self._device_idx = device_idx
        self._associated_event = associated_event
        self._evictor_data = None
        # TODO: Check where this is set.
        self._storage_info = None
        self._hit_cnt = 0
        # if compression_level != 0:
        #     print(f"{self._id} is compressed to {compression_level}")
    @property
    def device_idx(self):
        return self._device_idx
    @property
    def evictor_data(self):
        return self._evictor_data
    @property
    def storage_info(self):
        return self._storage_info
    @property
    def prefix_token_len(self):
        return self._prefix_token_len
    @property
    def prefix_hash(self):
        return self._prefix_hash
    @property
    def hash_value(self):
        return self._hash_value
    @property
    def chunk_token_len(self):
        return self._chunk_token_len
    @property
    def size(self):
        return self._size
    @property
    def compression_level(self):
        return self._compression_level
    @property
    def status(self):
        return self._status
    @property
    def associated_event(self):
        return self._associated_event
    @property
    def hit_cnt(self):
        return self._hit_cnt
    def set_evictor_data(self, evictor_data):
        self._evictor_data = evictor_data
    def set_storage_info(self, storage_info):
        self._storage_info = storage_info
    def update_status(self, status: StorageInfoType):
        self._status = status
    def update_associated_event(self, event):
        self._associated_event = event
    def update_device_idx(self, device_idx):
        self._device_idx = device_idx
    def inc_hit_cnt(self):
        self._hit_cnt += 1



class StorageInfo:
    def __init__(self):
        self.copies: Dict[int, Dict[StorageInfoType, Set[KVObjectMetadata]]] = {}

    def lookup(self, compression_level: Optional[int], status: Optional[StorageInfoType]) -> set:
        ret_set = set()
        # print(f"lookup called with: {compression_level}, {status}")
        # print(f"Full info: {self.copies}\n\n")
        if compression_level is None:
            if status is None:
                for status_dict in self.copies.values():
                    for kv_object_meta_set in status_dict.values():
                        ret_set.update(kv_object_meta_set)
            else:
                for status_dict in self.copies.values():
                    if status in status_dict:
                        ret_set.update(status_dict[status])
        else:
            status_dict = self.copies.get(compression_level, None)
            if status_dict is not None:
                if status is None:
                    for kv_object_meta_set in status_dict.values():
                        ret_set.update(kv_object_meta_set)
                else:
                    if status in status_dict:
                        ret_set.update(status_dict[status])
        return ret_set
    
    def add_copy(self, kv_object_meta: KVObjectMetadata) -> bool:
        # Return how many bytes it will take.
        if kv_object_meta.compression_level in self.copies:
            logger.warning("Two copies with same content on same device.")
            if kv_object_meta.status in self.copies[kv_object_meta.compression_level]:
                logger.warning("Two copies with same content on same device, even with same status.")
                self.copies[kv_object_meta.compression_level][kv_object_meta.status].add(kv_object_meta)
            else:
                self.copies[kv_object_meta.compression_level][kv_object_meta.status] = {kv_object_meta}
        else:
            # print(f"add new copy {kv_object_meta._id}")
            self.copies[kv_object_meta.compression_level] = {kv_object_meta.status: {kv_object_meta}}
        kv_object_meta.set_storage_info(self)
        return True
    
    def remove_copy(self, kv_object_metadata: KVObjectMetadata) -> bool:
        assert kv_object_metadata.compression_level in self.copies
        assert kv_object_metadata.status in self.copies[kv_object_metadata.compression_level]
        assert kv_object_metadata in self.copies[kv_object_metadata.compression_level][kv_object_metadata.status]
        # print(f"Removing copy {kv_object_metadata._id}: ori len {len(self.copies)}")
        self.copies[kv_object_metadata.compression_level][kv_object_metadata.status].remove(kv_object_metadata)
        if len(self.copies[kv_object_metadata.compression_level][kv_object_metadata.status]) == 0:
            del self.copies[kv_object_metadata.compression_level][kv_object_metadata.status]
            if len(self.copies[kv_object_metadata.compression_level]) == 0:
                del self.copies[kv_object_metadata.compression_level]
        # print(f", aft len: {len(self.copies)}")
        # Might use this in update_on_transfer, or mark_ready.
        # kv_object_metadata.set_evictor_data(None)
        kv_object_metadata.set_storage_info(None)
        return True

    def check_exist(self):
        return len(self.copies) > 0
    
    def mark_ready(self, kv_obj_metadata: KVObjectMetadata) -> bool:
        assert kv_obj_metadata.status == StorageInfoType.ARRIVING
        self.remove_copy(kv_obj_metadata)
        kv_obj_metadata.update_status(StorageInfoType.READY)
        assert kv_obj_metadata.associated_event is not None
        kv_obj_metadata.update_associated_event(None)
        self.add_copy(kv_obj_metadata)
        assert kv_obj_metadata.status == StorageInfoType.READY
        # print(f"id: {kv_obj_metadata._id} is ready.")
        return True


class KVTokenIndexItem:
    def __init__(
            self,
            prefix_hash: str,
            current_hash: str,
    ):
        # Index into here, with tokens && device.
        # Can have multiple compression level copies.
        self._prefix_hash = prefix_hash
        self._hash_val = current_hash
        self.storage_info = StorageInfo()
    def add_copy(self, kv_object_metadata: KVObjectMetadata):
        return self.storage_info.add_copy(kv_object_metadata)
    def remove_copy(self, kv_object_metadata: KVObjectMetadata) -> bool:
        return self.storage_info.remove_copy(kv_object_metadata)
    def check_exist(self):
        return self.storage_info.check_exist()
    def lookup(self, compression_level: Optional[int], status: Optional[StorageInfoType]) -> set:
        return self.storage_info.lookup(compression_level, status)
    @property
    def hash_val(self):
        return self._hash_val

# NOTE: General rule is, when you do not have the kv object, call with hash/prompt.
# When you have, call with that.

# Only every TP works are taken as one(TP always acts in lockstep).
# PP has different backends. Also different replicas of models.
# So key is just hash of content of tokens.

# Per replica per stage PER STORAGE DEVICE.
storage_backend_id = 0
class KVStorageBackEnd:
    def __init__(self, cache_engine):
        global storage_backend_id
        self._id = storage_backend_id
        storage_backend_id += 1
        self._hash_to_chunk: Dict[str, KVTokenIndexItem] = {}
        self._cache_engine = cache_engine
        self._kv_size_calculator = self._cache_engine.kv_size_calculator
        self._total_cache_size = 0
        self._put_called_cnt = 0
        self._put_called_full_cnt = 0
        self._put_called_compressed_cnt = 0
        import atexit
        atexit.register(self.print_stats)
    def put(self, kv_object_metadata: KVObjectMetadata) -> bool:
        self._put_called_cnt += 1
        if kv_object_metadata.compression_level != 0:
            self._put_called_compressed_cnt += 1
        else:
            self._put_called_full_cnt += 1
        if kv_object_metadata.hash_value in self._hash_to_chunk:
            return self._hash_to_chunk[kv_object_metadata.hash_value].add_copy(kv_object_metadata)
        else:
            self._total_cache_size += kv_object_metadata.size
            new_item = KVTokenIndexItem(kv_object_metadata.prefix_hash, kv_object_metadata.hash_value)
            new_item.add_copy(kv_object_metadata)
            self._hash_to_chunk[kv_object_metadata.hash_value] = new_item
            return True
    def lookup(self, hash_value: str, compression_level: Optional[int], 
               status: Optional[StorageInfoType]) -> Optional[set]:
        if hash_value in self._hash_to_chunk:
            lookup_result = self._hash_to_chunk[hash_value].storage_info.lookup(compression_level, status)
            if len(lookup_result) > 0:
                return lookup_result
        return None
    def remove(self, kv_object_metadata: KVObjectMetadata) -> bool:
        if kv_object_metadata.hash_value in self._hash_to_chunk:
            remove_flag = self._hash_to_chunk[kv_object_metadata.hash_value].remove_copy(kv_object_metadata)
            if not self._hash_to_chunk[kv_object_metadata.hash_value].check_exist():
                assert remove_flag
                del self._hash_to_chunk[kv_object_metadata.hash_value]
            return remove_flag
        else:
            return False
    
    # Return an estimated cost.
    # Then handle remove and encode and store outside.
    def get_encode_cost(self, kv_object_metadata: KVObjectMetadata, new_compression_level: int) -> float:
        assert kv_object_metadata.hash_value in self._hash_to_chunk
        assert kv_object_metadata.compression_level == 0, "Only support transform from no compression."
        assert new_compression_level != 0, "Only support transform to compression."
        return kv_object_metadata.size * compress_level_manager.encode_cost[new_compression_level]
    
    def get_decode_cost(self, kv_object_metadata: KVObjectMetadata) -> float:
        assert kv_object_metadata.hash_value in self._hash_to_chunk
        assert kv_object_metadata.compression_level != 0, "Only support transform from compression."
        return kv_object_metadata.size * compress_level_manager.decode_cost[kv_object_metadata.compression_level]
        

    def print_stats(self):
        '''
        print(f"\nBackend {self._id}:")
        print(f"Total cache size: {self._total_cache_size // (1024 ** 3)} GB")
        print(f"Total put called: {self._put_called_cnt}")
        print(f"Total put called full: {self._put_called_full_cnt}")
        print(f"Total put called compressed: {self._put_called_compressed_cnt}")
        '''
        pass
        
