from vidur.entities.base_entity import BaseEntity
from vidur.entities.channel import Channel
from typing import List, Optional, Tuple, Dict
from vidur.config import CacheEngineConfig
from vidur.utils.parse_cli import parse_size, parse_thput
from vidur.types import StorageDeviceType, StorageComputeType, StorageInfoType
from vidur.types.evict_op_type import EvictOpType
import struct
import hashlib
from vidur.entities.kvitem import KVStorageBackEnd, KVObjectMetadata, get_compress_level_manager, KVObjectQuery
from vidur.events.transmission_end_event import TransmissionEndEvent
from vidur.entities.evictor import LFUEvictor, LRUEvictor, OurEvictorV1, BaseEvictor, LFUEvictorV2, \
    OursGlobalFrameworkEvictor, FixedOursAlpha, WindowLFUEstimator, OursSuperChunkEvictor
from vidur.entities.compute import ComputationDevice
from vidur.events.compute_end_event import ComputeEndEvent
import atexit

'''
What can be refering to a kv object.
StorageBackEnd.
Evictor will refer to evict_data.
In some End Events, as callbacks.
'''

class CacheLogLevel:
    DEFAULT = 0
    V1 = 1


# NOTE: Currently does not support token dropping like, because we decompress
# before use. And for token dropping to enable larger batch size, scheduler needs 
# to be changed, and do not decompress it.


'''
swap once:
For eviction, done in _make_space to force a drop.
For copy, always to GPU for now, and no prefix cache in GPU, so not a problem.
'''

class DeviceIndex:
    def __init__(self, compute_type: StorageComputeType, replica_id: int, stage_id: int, device_type: StorageDeviceType):
        self.compute_type = compute_type
        self.replica_id = replica_id
        self.stage_id = stage_id
        self.device_type = device_type
        self.local_backend_no = -1
        if device_type == StorageDeviceType.GPU:
            self.local_backend_no = 0
        elif device_type == StorageDeviceType.CPU:
            self.local_backend_no = 1
        elif device_type == StorageDeviceType.DISK:
            self.local_backend_no = 2
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
    def __hash__(self):
        return hash((self.compute_type, self.replica_id, self.stage_id, self.device_type))
    def __eq__(self, other):
        return self.compute_type == other.compute_type and \
                self.replica_id == other.replica_id and \
                self.stage_id == other.stage_id and \
                self.device_type == other.device_type
    

class ChannelManager:
    def __init__(self):
        self._map_to_channel: Dict[tuple, Tuple[float, Channel]] = {}
    def add_channel(self, from_device: DeviceIndex, to_device: DeviceIndex, thput: float, channel: Channel):
        assert (from_device, to_device) not in self._map_to_channel
        self._map_to_channel[((from_device.compute_type, from_device.replica_id, from_device.stage_id, from_device.device_type), 
                              (to_device.compute_type, to_device.replica_id, to_device.stage_id, to_device.device_type))] = \
                                (thput, channel)
    def get_channel(self, from_device: DeviceIndex, to_device: DeviceIndex) -> Tuple[float, Channel]:
        return self._map_to_channel[((from_device.compute_type, from_device.replica_id, from_device.stage_id, from_device.device_type), 
                              (to_device.compute_type, to_device.replica_id, to_device.stage_id, to_device.device_type))]


global_channel_manager = ChannelManager()

# Read and write does not contend, but inside read/write, contend once in end in common.
def build_local_channels_rw_no_contend(cache_engine_config: CacheEngineConfig, replica_id: int, stage_id: int):
    global global_channel_manager
    # With computation device.
    have_cpu = len(cache_engine_config.cpu_memory_size) > 0
    have_disk = len(cache_engine_config.disk_size) > 0
    gpu_read_channel = Channel()
    gpu_write_channel = Channel()
    cpu_to_disk_channel = Channel()
    disk_to_cpu_channel = Channel()
    gpu_device = DeviceIndex(StorageComputeType.CAN_COMPUTE, replica_id, stage_id, StorageDeviceType.GPU)
    cpu_device = DeviceIndex(StorageComputeType.CAN_COMPUTE, replica_id, stage_id, StorageDeviceType.CPU)
    disk_device = DeviceIndex(StorageComputeType.CAN_COMPUTE, replica_id, stage_id, StorageDeviceType.DISK)
    if have_cpu and have_disk:
        global_channel_manager.add_channel(cpu_device, disk_device, parse_thput(cache_engine_config.cpu_disk_thput), cpu_to_disk_channel)
        global_channel_manager.add_channel(disk_device, cpu_device, parse_thput(cache_engine_config.disk_cpu_thput), disk_to_cpu_channel)
    if have_cpu:
        global_channel_manager.add_channel(cpu_device, gpu_device, parse_thput(cache_engine_config.cpu_gpu_thput), gpu_read_channel)
        global_channel_manager.add_channel(gpu_device, cpu_device, parse_thput(cache_engine_config.gpu_cpu_thput), gpu_write_channel)
    if have_disk:
        global_channel_manager.add_channel(disk_device, gpu_device, parse_thput(cache_engine_config.disk_gpu_thput), gpu_read_channel)
        global_channel_manager.add_channel(gpu_device, disk_device, parse_thput(cache_engine_config.gpu_disk_thput), gpu_write_channel)

def build_ours_framework_evictor(cache_engine_config: CacheEngineConfig, kv_size_calculator):
    alpha = FixedOursAlpha(cache_engine_config.evictor_alpha)
    thputs = []
    storage_sizes = []
    cpu_size = 0
    if len(cache_engine_config.cpu_memory_size) > 0:
        cpu_size = parse_size(cache_engine_config.cpu_memory_size)
        cpu_thput = parse_thput(cache_engine_config.cpu_gpu_thput)
        thputs.append(cpu_thput)
        storage_sizes.append(cpu_size)
    disk_size = 0
    if len(cache_engine_config.disk_size) > 0:
        disk_size = parse_size(cache_engine_config.disk_size)
        disk_thput = parse_thput(cache_engine_config.disk_gpu_thput)
        thputs.append(disk_thput)
        storage_sizes.append(disk_size)
    full_size = cpu_size + disk_size
    assert full_size > 0
    compression_manager = get_compress_level_manager()
    min_ratio_level = None
    min_ratio = None
    for level, ratio in compression_manager.to_rate.items():
        if min_ratio is None or ratio < min_ratio:
            min_ratio = ratio
            min_ratio_level = level
    assert min_ratio_level is not None
    chunk_size = cache_engine_config.cache_chunk_size
    min_kv_chunk_size = kv_size_calculator.get_kv_size(chunk_size, min_ratio_level)
    item_size_for_all_full_compress = full_size // min_kv_chunk_size
    lfu_window_size = 3 * item_size_for_all_full_compress
    # NOTE: Currently just 3 * C.
    estimator = WindowLFUEstimator(lfu_window_size)
    chunk_byte_size = kv_size_calculator.get_kv_size(chunk_size, 0)
    # TODO: Optimize_on_hit.
    global_ours_framework_evictor = OursGlobalFrameworkEvictor(alpha, estimator, 
                                                               compression_manager,
                                                               thputs,
                                                               storage_sizes,
                                                               False,
                                                               cache_engine_config.store_policy,
                                                               chunk_byte_size
                                                               )
    return global_ours_framework_evictor

def build_ours_super_chunk_evictor(cache_engine_config: CacheEngineConfig, kv_size_calculator):
    alpha = FixedOursAlpha(cache_engine_config.evictor_alpha)
    thputs = []
    storage_sizes = []
    cpu_size = 0
    if len(cache_engine_config.cpu_memory_size) > 0:
        cpu_size = parse_size(cache_engine_config.cpu_memory_size)
        cpu_thput = parse_thput(cache_engine_config.cpu_gpu_thput)
        thputs.append(cpu_thput)
        storage_sizes.append(cpu_size)
    disk_size = 0
    if len(cache_engine_config.disk_size) > 0:
        disk_size = parse_size(cache_engine_config.disk_size)
        disk_thput = parse_thput(cache_engine_config.disk_gpu_thput)
        thputs.append(disk_thput)
        storage_sizes.append(disk_size)
    full_size = cpu_size + disk_size
    assert full_size > 0
    compression_manager = get_compress_level_manager()
    min_ratio_level = None
    min_ratio = None
    for level, ratio in compression_manager.to_rate.items():
        if min_ratio is None or ratio < min_ratio:
            min_ratio = ratio
            min_ratio_level = level
    assert min_ratio_level is not None
    chunk_size = cache_engine_config.cache_chunk_size
    min_kv_chunk_size = kv_size_calculator.get_kv_size(chunk_size, min_ratio_level)
    item_size_for_all_full_compress = full_size // min_kv_chunk_size
    lfu_window_size = 3 * item_size_for_all_full_compress
    # NOTE: Currently just 3 * C.
    estimator = WindowLFUEstimator(lfu_window_size)
    chunk_byte_size = kv_size_calculator.get_kv_size(chunk_size, 0)
    # TODO: Optimize_on_hit.
    global_ours_super_chunk_evictor = OursSuperChunkEvictor(alpha, 
                                                            estimator,
                                                            compression_manager,
                                                            thputs,
                                                            storage_sizes,
                                                            False,
                                                            cache_engine_config.store_policy,
                                                            chunk_byte_size)
    return global_ours_super_chunk_evictor

# Per (model x pipeline_stage).
# So every batch_stage should have one.
# For global stuff, like across models, across pipeline stages, prefetch.
# Do it in a future orchestrator, call it in scheduler.
class CacheEngine(BaseEntity):
    def __init__(self, cache_engine_config: CacheEngineConfig, replica_stage_scheduler) -> None:
        super().__init__()
        self._id = CacheEngine.generate_id()
        self._cache_log_str = cache_engine_config.cache_log
        self._cache_log_level = CacheLogLevel.DEFAULT
        if self._cache_log_str.lower() == "default":
            self._cache_log_level = CacheLogLevel.DEFAULT
        elif self._cache_log_str.lower() == "v1":
            self._cache_log_level = CacheLogLevel.V1
        else:
            raise ValueError(f"Cache log not recognized: {self._cache_log}")
        self._store_policy_str = cache_engine_config.store_policy
        self._evict_policy = cache_engine_config.eviction_policy
        self._ours_v1_token_thres = cache_engine_config.ours_v1_token_thres
        self._cpu_memory_size = 0
        if len(cache_engine_config.cpu_memory_size) > 0:
            self._cpu_memory_size = parse_size(cache_engine_config.cpu_memory_size)
            print(f"cpu memory size: {self._cpu_memory_size}")
        self._disk_size = 0
        if len(cache_engine_config.disk_size) > 0:
            self._disk_size = parse_size(cache_engine_config.disk_size)
            print(f"disk size: {self._disk_size}")
        self._full_chunk_only = cache_engine_config.full_chunk_only
        self._replica_stage_scheduler = replica_stage_scheduler
        self._replica_id = replica_stage_scheduler.replica_id
        self._stage_id = replica_stage_scheduler.stage_id
        self._kv_size_calculator = replica_stage_scheduler.kv_size_calculator
        self._disk_cpu_thput = cache_engine_config.disk_cpu_thput
        self._cpu_disk_thput = cache_engine_config.cpu_disk_thput
        self._cpu_gpu_thput = cache_engine_config.cpu_gpu_thput
        self._gpu_cpu_thput = cache_engine_config.gpu_cpu_thput
        self._disk_gpu_thput = cache_engine_config.disk_gpu_thput
        self._gpu_disk_thput = cache_engine_config.gpu_disk_thput
        self._contention_model = cache_engine_config.contention_model
        self._gpu_prefix_cache = cache_engine_config.gpu_prefix_cache
        self._chunk_size = cache_engine_config.cache_chunk_size
        self._gpu_compute_device: ComputationDevice = replica_stage_scheduler.gpu_compute_device
        self._cpu_compute_device: ComputationDevice = ComputationDevice()
        self._simulator = replica_stage_scheduler.simulator
        self._wasted_write_to_lower = 0
        self._wasted_compress = 0
        self._wasted_drop = 0
        self._effective_put_cnt = 0
        # TODO: GPU prefix cache not supported now.
        assert not self._gpu_prefix_cache
        # TODO: Only one rw-no-contend model.
        assert self._contention_model == "rw-no-contend"
        if self._contention_model == "rw-no-contend":
            build_local_channels_rw_no_contend(cache_engine_config, 
                                               replica_stage_scheduler.replica_id, 
                                               replica_stage_scheduler.stage_id)
            
        self._retrieve_chunk_cnt = 0
        self._hit_chunk_cnt = 0
        self._hit_in_cpu_cnt = 0
        self._hit_in_disk_cnt = 0
        
        self._gpu_dev_index = DeviceIndex(StorageComputeType.CAN_COMPUTE, 
                                          self._replica_id,
                                          self._stage_id,
                                          StorageDeviceType.GPU)
        self._cpu_dev_index = DeviceIndex(StorageComputeType.CAN_COMPUTE, 
                                          self._replica_id,
                                          self._stage_id,
                                          StorageDeviceType.CPU)
        self._disk_dev_index = DeviceIndex(StorageComputeType.CAN_COMPUTE, 
                                          self._replica_id,
                                          self._stage_id,
                                          StorageDeviceType.DISK)
        
        # TODO: Manage size here.
        # (backend, size)
        self._evictors = None
        self._storage_backends = None
        if self._gpu_prefix_cache:
            raise NotImplementedError("GPU prefix cache not supported now.")
        else:
            self._storage_backends: List[List[Optional[KVStorageBackEnd], Optional[int]]] = [[None, None]]
            self._evictors = [None]
            global_ours_framework_evictor = None
            global_ours_super_chunk_evictor = None
            if len(cache_engine_config.cpu_memory_size) > 0:
                self._storage_backends.append([KVStorageBackEnd(self), self._cpu_memory_size])
                if self._evict_policy.lower() == "oursframework":
                    if global_ours_framework_evictor is None:
                        global_ours_framework_evictor = build_ours_framework_evictor(cache_engine_config, self._kv_size_calculator)
                    self._evictors.append(global_ours_framework_evictor)
                elif self._evict_policy.lower() == "ourssuperchunk":
                    if global_ours_super_chunk_evictor is None:
                        global_ours_super_chunk_evictor = build_ours_super_chunk_evictor(cache_engine_config, self._kv_size_calculator)
                    self._evictors.append(global_ours_super_chunk_evictor)
                else:
                    self._evictors.append(self._get_evictor_by_name(self._evict_policy))
            else:
                self._storage_backends.append([None, None])
                self._evictors.append(None)
            if len(cache_engine_config.disk_size) > 0:
                self._storage_backends.append([KVStorageBackEnd(self), self._disk_size])
                if self._evict_policy.lower() == "oursframework":
                    if global_ours_framework_evictor is None:
                        global_ours_framework_evictor = build_ours_framework_evictor(cache_engine_config, self._kv_size_calculator)
                    self._evictors.append(global_ours_framework_evictor)
                elif self._evict_policy.lower() == "ourssuperchunk":
                    if global_ours_super_chunk_evictor is None:
                        global_ours_super_chunk_evictor = build_ours_super_chunk_evictor(cache_engine_config, self._kv_size_calculator)
                    self._evictors.append(global_ours_super_chunk_evictor)
                else:
                    self._evictors.append(self._get_evictor_by_name(self._evict_policy))
            else:
                self._storage_backends.append([None, None])
                self._evictors.append(None)
        sizes = [self._storage_backends[1][1], self._storage_backends[2][1]]
        # print(f"cpu size: {sizes[0]}, disk size: {sizes[1]}")
        self._debug_info = {}
        atexit.register(self.print_stats)

    def print_stats(self):
        print(f"hit rate: {self._hit_chunk_cnt / self._retrieve_chunk_cnt}")
        if self._hit_chunk_cnt > 0:
            print(f"hit in cpu rate: {self._hit_in_cpu_cnt / self._hit_chunk_cnt}")
            print(f"hit in disk rate: {self._hit_in_disk_cnt / self._hit_chunk_cnt}")
        # print(f"effective put cnt: {self._effective_put_cnt}")
        print(f"wasted item on write to lower: {self._wasted_write_to_lower}")
        print(f"wasted item on compress: {self._wasted_compress}")
        print(f"wasted item on drop: {self._wasted_drop}")
            

        
    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    def _get_evictor_by_name(self, evictor_name: str):
        if evictor_name.lower() == "lru":
            return LRUEvictor()
        elif evictor_name.lower() == "lfu":
            return LFUEvictor()
        elif evictor_name.lower() == "lfuv2":
            return LFUEvictorV2()
        elif evictor_name.lower() == "oursv1":
            return OurEvictorV1(self._ours_v1_token_thres)
        else:
            raise ValueError(f"Eviction policy not recognized: {self._evict_policy}")
            
        
    def _chunk_tokens(self,
                      tokens: list):
        retval = []
        token_cnt = len(tokens)
        for i in range(0, token_cnt, self._chunk_size):
            end_idx = min(i + self._chunk_size, token_cnt)
            chunk_token = tokens[i:end_idx]
            chunk_tuple = tuple(chunk_token)
            retval.append(chunk_tuple)
        if self._full_chunk_only:
            if token_cnt % self._chunk_size != 0:
                retval.pop()
        return tuple(retval)

    def _hash_tokens(
        self,
        tokens: tuple,
        prefix_hash: str,
    ) -> str:
        assert len(tokens) > 0
        bytes_array = None
        if type(tokens[0]) == int:
            bytes_array = bytearray(struct.pack("i" * len(tokens), *tokens))
        elif type(tokens[0]) == str:
            bytes_array = bytearray("".join(tokens).encode("utf-8"))
        else:
            raise ValueError("Unsupported token type")
        return hashlib.sha256(
             prefix_hash.encode("ascii") + bytes_array).hexdigest()
    
    # Update index --> do trans op --> change space.
    def _make_space(self, cur_time: float, need_size: int, backend_no: int) -> float:
        # print(f"make space called at {cur_time}, need_size: {need_size}, backend_no: {backend_no}")
        assert backend_no != 0
        evict_end_time = cur_time
        # Make space.
        assert self._storage_backends[backend_no][0] is not None
        while self._storage_backends[backend_no][1] < need_size:
            evictor: BaseEvictor = self._evictors[backend_no]
            # evict_op, evict_operand = evictor.evict(backend_no)
            evict_list = evictor.evict(backend_no)
            for evict_op, evict_operand in evict_list:
                # Update index.
                evicted_item: Optional[KVObjectMetadata] = None
                assert evict_op != EvictOpType.NONE
                if evict_op == EvictOpType.WRITE_TO_LOWER:
                    # print(f"WRITE_TO_LOWER {evict_operand.hash_value}\n\n\n")
                    evicted_item = evict_operand
                    '''
                    if evicted_item is None:
                        # print(f"Current size in backend_no: {backend_no} is {self._storage_backends[backend_no][1]}")
                        # print(f"Available hashes: {len(self._storage_backends[backend_no][0]._hash_to_chunk)}")
                        for kvindex in self._storage_backends[backend_no][0]._hash_to_chunk.values():
                            kv_st = kvindex.storage_info
                            compress = 0
                            status_dict = kv_st.copies.get(compress, None)
                            if status_dict is not None:
                                if status_dict[StorageInfoType.READY] is not None:
                                    print(f"READY")
                                elif status_dict[StorageInfoType.ARRIVING] is not None:
                                    print(f"ARRIVING")
                                else:
                                    print(f"UNKNOWN")
                    '''
                    assert evicted_item is not None, "WRITE_TO_LOWER must have evicted item."
                elif evict_op == EvictOpType.DROP:
                    evicted_item = evict_operand
                    assert evicted_item is not None, "DROP must have evicted item."
                elif evict_op == EvictOpType.COMPRESS:
                    evicted_item = evict_operand[0]
                    # print(f"evict compress copies: {evicted_item.storage_info.copies}")
                    assert evicted_item is not None, "COMPRESS must have evicted item."
                else:
                    raise ValueError(f"Unsupported evict op: {evict_op}")
                assert evicted_item is not None
                evict_make_space = None
                have_in_next_layer = backend_no < 2 and self._storage_backends[backend_no + 1][0] is not None and \
                self._storage_backends[backend_no + 1][0].lookup(evicted_item.hash_value, None, None) is not None
                have_next_layer = backend_no < 2 and self._storage_backends[backend_no + 1][0] is not None
                if evict_op == EvictOpType.WRITE_TO_LOWER and have_in_next_layer:
                    # Swap out once.
                    evict_op = EvictOpType.DROP
                if evict_op == EvictOpType.WRITE_TO_LOWER and not have_next_layer:
                    # Have to drop.
                    evict_op = EvictOpType.DROP
                evict_op_return_time = cur_time
                # print(f"remove from backend_no: {backend_no}, evicted_item: {evicted_item._id}")
                if evict_op == EvictOpType.WRITE_TO_LOWER:
                    assert self._storage_backends[backend_no][0].remove(evicted_item)
                    # print(f"remove from {backend_no}, size from {self._storage_backends[backend_no][1]} to {self._storage_backends[backend_no][1] + evicted_item.size}")
                    assert not have_in_next_layer
                    make_space_end = cur_time
                    if evicted_item.hit_cnt == 0:
                        self._wasted_write_to_lower += 1
                    if evicted_item.size > self._storage_backends[backend_no + 1][1]:
                        make_space_end = self._make_space(cur_time, evicted_item.size, backend_no + 1)
                        evict_op_return_time = max(evict_op_return_time, make_space_end)
                    from_device = None
                    to_device = None
                    if backend_no == 0:
                        from_device = self._gpu_dev_index
                        to_device = self._cpu_dev_index
                    elif backend_no == 1:
                        from_device = self._cpu_dev_index
                        to_device = self._disk_dev_index
                    else:
                        raise ValueError(f"Unsupported backend_no: {backend_no} and {backend_no + 1}")
                    thput, channel = global_channel_manager.get_channel(from_device , to_device)
                    launch_time, trans_time = channel.transmit(evicted_item.size, make_space_end, thput)
                    trans_end_time = launch_time + trans_time
                    trans_end_event = TransmissionEndEvent(trans_end_time, [])
                    new_kv_obj = KVObjectMetadata(evicted_item.prefix_hash,
                                                    evicted_item.hash_value,
                                                    evicted_item.prefix_token_len,
                                                    evicted_item.chunk_token_len,
                                                    evicted_item.size,
                                                    evicted_item.compression_level,
                                                    StorageInfoType.ARRIVING,
                                                    to_device,
                                                    trans_end_event)
                    # Mark on the next layer as arriving.
                    assert self._storage_backends[backend_no + 1][0].put(new_kv_obj)
                    # print(f"put to {backend_no + 1}, size from {self._storage_backends[backend_no + 1][1]} to {self._storage_backends[backend_no + 1][1] - new_kv_obj.size}")
                    # print(f"WRITE_TO_LOWER EVICT new_kv_obj: {new_kv_obj._id}")
                    next_evictor: BaseEvictor = self._evictors[backend_no + 1]
                    next_evictor.update_on_transfer(evicted_item, new_kv_obj)
                    trans_end_event.append_item(new_kv_obj)
                    global_simulator = self._simulator
                    # Always blocking.
                    global_simulator.add_events([trans_end_event])
                    evict_op_return_time = global_simulator.loop_until(trans_end_event)
                    self._storage_backends[backend_no + 1][1] -= new_kv_obj.size
                    evict_make_space = evicted_item.size
                elif evict_op == EvictOpType.COMPRESS:
                    compression_level = evict_operand[1]
                    assert compression_level != 0
                    assert compression_level > evicted_item.compression_level
                    # print(f"evict compress before transform: {evicted_item.storage_info.copies}")
                    # print("Compress in backend_no: ", backend_no)
                    if evicted_item.hit_cnt == 0:
                        self._wasted_compress += 1
                    transform_end_time, new_obj = self._transform(cur_time, evicted_item.compression_level, 
                                                                compression_level, evicted_item, backend_no, 
                                                                True, True, False)
                    evict_op_return_time = max(evict_op_return_time, transform_end_time)
                    evict_make_space = 0 # NOTE: Transform itself HAS modified the space!!
                elif evict_op == EvictOpType.DROP:
                    if evicted_item.hit_cnt == 0:
                        self._wasted_drop += 1
                    # print(f"remove from {backend_no}, size from {self._storage_backends[backend_no][1]} to {self._storage_backends[backend_no][1] + evicted_item.size}")
                    assert self._storage_backends[backend_no][0].remove(evicted_item)
                    evict_make_space = evicted_item.size
                else:
                    raise ValueError(f"Unsupported evict op: {evict_op}")
                assert type(evict_make_space) == int
                evict_end_time = max(evict_end_time, evict_op_return_time)
                # More free space.
                self._storage_backends[backend_no][1] += evict_make_space
        return evict_end_time



    # Make space, channel/compute, EndEvent, update index, udpate space.
    def _copy(self, cur_time: float, from_device: DeviceIndex, to_device: DeviceIndex, kv_obj: KVObjectMetadata,
              blocking: bool) -> float:
        need_size = kv_obj.size
        to_no = to_device.local_backend_no
        if to_no != 0:
            # Make space.
            # NOTE: If is 0, it is handled by scheduler.
            # Scheduler should manage the memory in GPU.
            self._make_space(cur_time, need_size, to_no)
        thput, channel = global_channel_manager.get_channel(from_device, to_device)
        # print(f"kv_chunk size is {kv_obj.size}, thput is {thput}")
        launch_time, trans_time = channel.transmit(kv_obj.size, cur_time, thput)
        trans_end_time = launch_time + trans_time
        trans_end_event = TransmissionEndEvent(trans_end_time, [])
        if to_device.device_type == StorageDeviceType.GPU:
            pass
            # print(f"{from_device.device_type}: {cur_time}, {launch_time}, {trans_time}, {trans_end_time}")
        # Update index.
        new_kv_obj = None
        if self._storage_backends[to_no][0] is not None:
            new_kv_obj = KVObjectMetadata(kv_obj.prefix_hash, 
                                          kv_obj.hash_value,
                                          kv_obj.prefix_token_len,
                                          kv_obj.chunk_token_len,
                                          kv_obj.size,
                                          kv_obj.compression_level,
                                          StorageInfoType.ARRIVING,
                                          to_device,
                                          trans_end_event
                                          )
            trans_end_event.append_item(new_kv_obj)
            # Update indexing.
            # NOTE: The one stored in backend and the one kv_obj sent to event
            # must be the same one.
            assert self._storage_backends[to_no][0].put(new_kv_obj)
            # Update size.
            # Less free space.
            # print(f"put to {to_no}, size from {self._storage_backends[to_no][1]} to {self._storage_backends[to_no][1] - new_kv_obj.size}")
            self._storage_backends[to_no][1] -= new_kv_obj.size
        global_simulator = self._simulator
        global_simulator.add_events([trans_end_event])
        if blocking:
            cur_time = global_simulator.loop_until(trans_end_event)
        return cur_time
    
    def _transform(self, cur_time: float, from_compress_level: int, 
    to_compress_level: int, kv_obj: KVObjectMetadata, 
                   backend_no: int,
                   replace_original: bool,
                   blocking: bool, temporary: bool) -> Tuple[float, KVObjectMetadata]:
        if from_compress_level == 0 and to_compress_level == 0:
            return cur_time, kv_obj.size
        assert blocking, "Now only support blocking transform."
        compress_level_manager = get_compress_level_manager()
        ratio = compress_level_manager.multiply_rate_from_to(from_compress_level, to_compress_level)
        if backend_no != 0 and not replace_original:
            assert blocking, "Now only support blocking transform."
            new_size = int(kv_obj.size * ratio)
            cur_time = self._make_space(cur_time, new_size, backend_no)
        transform_time = 0
        if to_compress_level == 0:
            transform_time = compress_level_manager.get_decompress_time(kv_obj.chunk_token_len, kv_obj.compression_level)
        else:
            transform_time = compress_level_manager.get_compress_time(kv_obj.chunk_token_len, kv_obj.compression_level)
        transform_launch_time = None
        if backend_no == 0:
            transform_launch_time, transform_time = self._gpu_compute_device.compute(transform_time, cur_time)
        else:
            transform_launch_time, transform_time = self._cpu_compute_device.compute(transform_time, cur_time)
        transform_end_time = transform_launch_time + transform_time
        transform_end_event = ComputeEndEvent(transform_end_time, [], None)
        device_idx = None
        if backend_no == 0:
            device_idx = self._gpu_dev_index
        elif backend_no == 1:
            device_idx = self._cpu_dev_index
        elif backend_no == 2:
            device_idx = self._disk_dev_index
        assert device_idx is not None
        if temporary:
            device_idx = None
        new_kv_obj = KVObjectMetadata(kv_obj.prefix_hash, kv_obj.hash_value, 
                                      kv_obj.prefix_token_len, kv_obj.chunk_token_len,
                                      int(kv_obj.size * ratio), to_compress_level, StorageInfoType.ARRIVING,
                                      device_idx,
                                      transform_end_event
                                      )
        if to_compress_level != 0:
            pass
            # print(f"Transform to size: {new_kv_obj.size}")
        # print(f"id from {kv_obj._id} to {new_kv_obj._id}, transform from {from_compress_level} to {to_compress_level}, size: {kv_obj.size}, new size: {new_kv_obj.size}")
        evictor: Optional[BaseEvictor] = self._evictors[backend_no]
        if evictor is not None:
            evictor.update_on_transform(kv_obj, new_kv_obj, cur_time) # Copy eviction data like frequency.
            # Note that _transform does not call update_on_put, becuase it is not a put access.
            # update_on_transform should put the new object into evictor.
        transform_end_event.append_item(new_kv_obj)
        # Update index and space.
        if replace_original:
            if backend_no != 0:
                assert self._storage_backends[backend_no][0] is not None
                # print(f"{kv_obj.storage_info.copies}\n\n")
                # print(f"Remove size: {kv_obj.size}")
                # print(f"remove from {backend_no}, size from {self._storage_backends[backend_no][1]} to {self._storage_backends[backend_no][1] + kv_obj.size}")
                # print(f"Removing {kv_obj.hash_value}")
                assert self._storage_backends[backend_no][0].remove(kv_obj)
                self._storage_backends[backend_no][1] += kv_obj.size
        if backend_no != 0 and not temporary:
            assert self._storage_backends[backend_no][0] is not None
            # print(f"Put size: {new_kv_obj.size}")
            assert self._storage_backends[backend_no][0].put(new_kv_obj)
            # print(f"put to {backend_no}, size from {self._storage_backends[backend_no][1]} to {self._storage_backends[backend_no][1] - new_kv_obj.size}")
            self._storage_backends[backend_no][1] -= new_kv_obj.size
        global_simulator = self._simulator
        global_simulator.add_events([transform_end_event])
        if blocking:
            cur_time = global_simulator.loop_until(transform_end_event)
        return cur_time, new_kv_obj
        


    # Make space, channel/compute, EndEvent, update index, udpate space, evictor_update.
    def retrieve(
        self,
        current_time: float,
        tokens: list,
        skip_leading_tokens: int,
        blocking: bool,
    ) -> Tuple[int, float, float]:
        # Currently retrieve is always blocking, until ready to execute.
        # Return how many tokens are fetched, and time.
        assert blocking, "Now only support blocking retrieve."
        assert skip_leading_tokens == 0, "Now no GPU prefix cache, no skipping."
        # print(f"retrieve called at {current_time}")
        # print(f"token number: {len(tokens)}")
        gpu_kv_objs = []
        cpu_kv_objs = []
        disk_kv_objs = []
        hit_token_cnt = 0
        chunk_tuple = self._chunk_tokens(tokens)
        # print(f"chunks number: {len(chunk_tuple)}")
        current_hash = ""
        compress_level_manager = get_compress_level_manager()
        min_quality = compress_level_manager.to_quality[0]
        # Should not trigger evict.
        retrieved_chunks = []
        from_devices = []
        self._retrieve_chunk_cnt += len(chunk_tuple)
        # print(f"retrieve called with {len(chunk_tuple)} chunks")
        # print(f"\n\ncpu size now: {self._storage_backends[1][1]}")
        idx = 0
        for chunk in chunk_tuple:
            idx += 1
            current_hash = self._hash_tokens(chunk, current_hash)
            # print(f"Retrieve hash {current_hash}")
            found_chunk = False
            for storage_no, storage_pair in enumerate(self._storage_backends):
                storage_backend = storage_pair[0]
                if storage_backend is None:
                    continue
                lookup_result = storage_backend.lookup(current_hash, None, StorageInfoType.READY)
                if lookup_result is None:
                    if self._cache_log_level > CacheLogLevel.DEFAULT:
                        pass
                    continue
                assert len(lookup_result) > 0
                min_compression_level = None
                selected_kv_obj: Optional[KVObjectMetadata] = None
                for kv_obj_metadata in lookup_result:
                    if min_compression_level is None:
                        min_compression_level = kv_obj_metadata.compression_level
                        selected_kv_obj = kv_obj_metadata
                    else:
                        if kv_obj_metadata.compression_level < min_compression_level:
                            min_compression_level = kv_obj_metadata.compression_level
                            selected_kv_obj = kv_obj_metadata
                assert selected_kv_obj is not None
                selected_kv_obj.inc_hit_cnt()
                retrieved_chunks.append(selected_kv_obj)
                if storage_no == 0:
                    gpu_kv_objs.append(selected_kv_obj)
                    from_devices.append(self._gpu_dev_index)
                elif storage_no == 1:
                    cpu_kv_objs.append(selected_kv_obj)
                    from_devices.append(self._cpu_dev_index)
                    self._hit_in_cpu_cnt += 1
                elif storage_no == 2:
                    disk_kv_objs.append(selected_kv_obj)
                    from_devices.append(self._disk_dev_index)
                    self._hit_in_disk_cnt += 1
                else:
                    raise ValueError(f"Unsupported storage_no: {storage_no}")
                found_chunk = True
                self._hit_chunk_cnt += 1
                the_quality = compress_level_manager.to_quality[selected_kv_obj.compression_level]
                min_quality = min(min_quality, the_quality)
                break
            if not found_chunk:
                # print(f"not found chunk {idx - 1}")
                break
            else:
                hit_token_cnt += len(chunk)



        # Move all the chunks to GPU.
        # Do decompression(if there is such an overhead).
        # NOTE: Currently do it in GPU, and replace the original.
        # NOTE: Currently only update evictor of CPU and Disk on those kv_objs.
        assert len(gpu_kv_objs) == 0
        cpu_evictor: Optional[BaseEvictor] = self._evictors[1]
        disk_evictor: Optional[BaseEvictor] = self._evictors[2]
        evictor_space_list = []
        for level in range(1, len(self._evictors)):
            if self._evictors[level] is not None:
                evictor_space_list.append(self._storage_backends[level][1])
        # print(f"before evictor, cpu size now: {self._storage_backends[1][1]}")
        if cpu_evictor is not None:
            # print(f"retrieve from cpu chunk cnt: {len(cpu_kv_objs)}")
            cpu_evictor.update_on_get(cpu_kv_objs, current_time, evictor_space_list)
        if disk_evictor is not None:
            # print(f"retrieve from disk chunk cnt: {len(disk_kv_objs)}")
            disk_evictor.update_on_get(disk_kv_objs, current_time, evictor_space_list)
        max_time = current_time
        # print("\n\n")
        for chunk_id, kv_obj in enumerate(retrieved_chunks):
            from_device = None
            to_device = self._gpu_dev_index
            from_device = from_devices[chunk_id]
            move_end_time = self._copy(current_time, from_device, to_device, kv_obj, blocking)
            time_es = move_end_time - current_time
            # print(f"From {from_device.local_backend_no} to {to_device.local_backend_no}, time: {time_es}")
            # Do it in too device, do not decompress in CPU.
            if kv_obj.compression_level != 0:
                # Note that this decompress happens in GPU and in the setting without GPU cache,
                # it will not affect storage at all.
                decompress_end_time, _ = self._transform(move_end_time, kv_obj.compression_level, 0, kv_obj, 
                                                    to_device.local_backend_no, True, blocking, False)
            else:
                decompress_end_time = move_end_time
            max_time = max(max_time, decompress_end_time)
        # print(f"out of retrieve, cpu size now: {self._storage_backends[1][1]}")
        return hit_token_cnt, max_time, min_quality


    def store(self,
              current_time: float,
              tokens: list,
              skip_existing: bool,
              blocking: bool) -> float:
        for evictor in self._evictors:
            if evictor is not None:
                evictor.begin_store()
        # print(f"store called at {current_time}")
        # print(f"store called with {len(tokens)} tokens")
        if self._cpu_memory_size == 0 and self._disk_size == 0:
            # print("No storage backend, store skipped.")
            return current_time
        return_time = current_time
        # print(f"token number: {len(tokens)}")
        chunk_tuple = self._chunk_tokens(tokens)
        # print(f"store chunks number: {len(chunk_tuple)}")
        # print(f"chunks number: {len(chunk_tuple)}")
        current_hash = ""
        total_need_size = 0
        # print(f"one req chunk number: {len(chunk_tuple)}")
        max_previous_level = 0
        following_token_number = sum([len(chunk) for chunk in chunk_tuple])
        skippped_kv_objs = []
        # print(f"\n\n\nstore called with {len(chunk_tuple)} chunks")
        for chunk_id, chunk in enumerate(chunk_tuple):
            prefix_hash = current_hash
            prefix_token_len = chunk_id * self._chunk_size
            current_hash = self._hash_tokens(chunk, prefix_hash)
            # print(f"Store hash {current_hash}")
            # print(f"Store hash {current_hash}")
            self._debug_info[current_hash] = (tokens[0], chunk_id)
            skip = False
            if skip_existing:
                for storage_id_no, storage_pair in enumerate(self._storage_backends):
                    if storage_pair[0] is not None:
                        lookup_result = storage_pair[0].lookup(current_hash, None, None)
                        if lookup_result is not None:
                            skip = True
                            kv_obj_lookup_list = list(lookup_result)
                            min_compress_obj = None
                            min_compress_level_here = None
                            for lookup_obj in kv_obj_lookup_list:
                                if min_compress_level_here is None or lookup_obj.compression_level < min_compress_level_here:
                                    min_compress_level_here = lookup_obj.compression_level
                                    min_compress_obj = lookup_obj
                            if min_compress_level_here > max_previous_level:
                                max_previous_level = min_compress_level_here
                            skippped_kv_objs.append(min_compress_obj)
                            # TODO: Check here.
                            # I think the necessary part should have been updated before in get.
                            if self._evictors[storage_id_no] is not None:
                                evictor: BaseEvictor = self._evictors[storage_id_no]
                                evictor.update_on_skipped_store([min_compress_obj], current_time)
            if skip:
                # print("store one chunk skipped\n")
                continue
            # NOTE: Store policy.
            self._effective_put_cnt += 1
            ori_kv_size = self._kv_size_calculator.get_kv_size(len(chunk), 0)
            kv_query = KVObjectQuery(prefix_hash, current_hash, prefix_token_len, len(chunk), ori_kv_size)
            evictor_space_list = []
            for level in range(1, len(self._evictors)):
                if self._evictors[level] is not None:
                    evictor_space_list.append(self._storage_backends[level][1])
            # NOTE: Currently always call the hightest storage level.
            following_size = self._kv_size_calculator.get_kv_size(following_token_number, 0)
            # print(f"Asking about the {chunk_id} chunk")
            store_info = self._evictors[1].get_store_info([kv_query], current_time, max_previous_level, following_size)
            if len(store_info) > 1:
                # NOTE: For now, only called one time in one store.
                # Should apply these operations to previously cached items.
                assert len(store_info) == 2
                compress_chunk_cnt, compress_level = store_info[0]
                if compress_level > max_previous_level:
                    max_previous_level = compress_level
                reduced_start = len(skippped_kv_objs) - compress_chunk_cnt
                assert reduced_start >= 0
                compress_kv_chunks = skippped_kv_objs[reduced_start:]
                for compress_item in compress_kv_chunks:
                    com_backend_no = compress_item.device_idx.local_backend_no
                    assert compress_level > compress_item.compression_level
                    prev_transform_end_time, prev_new_obj = self._transform(current_time, compress_item.compression_level, 
                                                                compress_level, compress_item, com_backend_no, 
                                                                True, True, False)
                    # No need to set arriving, no transmit at all.
                    return_time = max(return_time, prev_transform_end_time)
                
            store_compress_tuple = store_info[-1]
            store_to_no, store_compress_level = store_compress_tuple
            kv_size = None
            kv_obj = None
            to_device = self._cpu_dev_index if store_to_no == 0 else self._disk_dev_index
            if store_compress_level != 0:
                # TODO: The transform and make_space step are both blocking here.
                # Make it non-blocking later.
                ori_kv_size = self._kv_size_calculator.get_kv_size(len(chunk), 0)
                ori_kv_obj = KVObjectMetadata(prefix_hash, current_hash, prefix_token_len, 
                                              len(chunk), ori_kv_size, 0, StorageInfoType.READY, to_device, 
                                              None)
                # NOTE: Not replacing original, and blocking now.
                # temporary to make it not store to 0 device.
                transform_end_time, new_obj = self._transform(current_time, 0, store_compress_level, ori_kv_obj, 
                                                        0, False, True, True)
                assert transform_end_time >= current_time
                current_time = transform_end_time
                assert new_obj is not None
                assert new_obj.status == StorageInfoType.READY, f"new_obj status: {new_obj.status}"
                new_obj.update_status(StorageInfoType.ARRIVING)
                new_obj.update_device_idx(to_device)
                # Store new obj.
                kv_size = new_obj.size
                kv_obj = new_obj
            else:
                kv_size = self._kv_size_calculator.get_kv_size(len(chunk), 0)
                kv_obj = KVObjectMetadata(prefix_hash, current_hash, prefix_token_len,
                                          len(chunk), kv_size, 0, StorageInfoType.ARRIVING, 
                                          to_device,
                                          None)
            assert kv_obj is not None
            assert kv_obj.status == StorageInfoType.ARRIVING
            assert kv_obj.size == kv_size
            total_need_size += kv_size
            assert type(kv_size) == int
            # NOTE: evictor backend no 0 is 1 in cache engine backend.
            if store_to_no == 0 and self._cpu_compute_device == 0:
                to_device = self._disk_dev_index
            to_no = to_device.local_backend_no
            assert to_no != 0
            # TODO: Now this step is blocking. 
            # Make an evict end event, and make it return an put event.
            make_space_end = self._make_space(current_time, kv_size, to_no)
            assert self._storage_backends[to_no][1] >= kv_size
            assert self._storage_backends[to_no][0] is not None
            thput, channel = global_channel_manager.get_channel(self._gpu_dev_index, to_device)
            launch_time, trans_time = channel.transmit(kv_size, make_space_end, thput)
            trans_end_time = launch_time + trans_time
            trans_end_event = TransmissionEndEvent(trans_end_time, [kv_obj])
            kv_obj.update_associated_event(trans_end_event)
            assert kv_obj.associated_event is not None
            # print(f"store one chunk not skipped, {kv_obj._id}\n")
            # update_on_put.
            if self._evictors[to_no] is not None:
                evictor: BaseEvictor = self._evictors[to_no]
                evictor.update_on_put([kv_obj], current_time)
            assert self._storage_backends[to_no][0].put(kv_obj)
            # print(f"put to {to_no}, size from {self._storage_backends[to_no][1]} to {self._storage_backends[to_no][1] - kv_size}")
            # update space
            self._storage_backends[to_no][1] -= kv_size
            assert self._storage_backends[to_no][1] >= 0
            global_simulator = self._simulator
            global_simulator.add_events([trans_end_event])
            store_one_chunk_host_end_time = current_time
            if blocking:
                assert kv_obj.associated_event is not None
                # print(f"store one chunk {kv_obj._id}")
                return_time = global_simulator.loop_until(trans_end_event)
                store_one_chunk_host_end_time = return_time
            return_time = max(return_time, store_one_chunk_host_end_time)
            following_token_number -= len(chunk)
        # print(f"put {total_need_size}")
        # print(f"cpu size from {oringal_cpu_size} to {self._storage_backends[1][1]}\n\n")
        return return_time
            
            
        

    # Prefix lookup.
    def lookup(self,
               tokens: list) -> int:
        # Return number of tokens found.
        chunk_tuple = self._chunk_tokens(tokens)
        current_hash = ""
        lookup_cnt = 0
        for chunk in chunk_tuple:
            current_hash = self._hash_tokens(chunk, current_hash)
            # NOTE: Hide all the policy behind this API.
            # Current policy is lookup always lookup all compression levels.
            # And only look for READY ones.
            found_chunk = False
            for storage_pair in self._storage_backends:
                storage_backend = storage_pair[0]
                if storage_backend is None:
                    continue
                else:
                    lookup_result = storage_backend.lookup(current_hash, None, StorageInfoType.READY)
                    if lookup_result is None:
                        continue
                    lookup_cnt += len(chunk)
                    found_chunk = True
                    break
            if not found_chunk:
                break
        return lookup_cnt

    @property
    def kv_size_calculator(self):
        return self._kv_size_calculator

