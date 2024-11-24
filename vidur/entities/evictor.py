from abc import ABC, abstractmethod
from typing import List, Optional
from vidur.utils.itemlist import ItemList, ItemListWrapper
from vidur.utils.itemheap import ItemHeap, ItemHeapWrapper
from vidur.types import EvictOpType
from collections import deque
from enum import Enum
import copy


class StorePolicy:
    FAST_DEVICE = 0

class BaseEvictor(ABC):
    def __init__(self):
        pass
    # Always return in the format of operation, other operands.
    @abstractmethod
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list = None):
        # Use these chunks from storage.
        # Can trigger cache movements.
        pass
    @abstractmethod
    def update_on_put(self, chunk_kv: list, timepoint: float):
        # These chunks are put into storage.
        # Will not trigger cache movements.
        pass
    @abstractmethod
    def evict(self, local_backend_no: int):
        pass
    @abstractmethod
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        pass
    @abstractmethod
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        pass

    @abstractmethod
    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        # Instruct cache engine to put into cache.
        pass
    @abstractmethod
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        pass
    def begin_store(self):
        pass

class LRUEvictor(BaseEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list = None):
        for kv in chunk_kv:
            prefix_token_len = kv.prefix_token_len
            if kv.evictor_data is None:
                kv.set_evictor_data(ItemHeapWrapper(kv, (timepoint, -prefix_token_len)))
                self._item_heap.push_heap(kv.evictor_data)
            kv = kv.evictor_data
            kv.set_score((timepoint, -prefix_token_len))
            self._item_heap.update_on_keychange(kv)
        return EvictOpType.NONE, None
    def update_on_put(self, chunk_kv: list, timepoint: float):
        return self.update_on_get(chunk_kv, timepoint)
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        raise NotImplementedError("LFU does not transform.")
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        assert to_kv_obj.evictor_data is None
        assert from_kv_obj.evictor_data is not None, f"{from_kv_obj._id} evictor data is None."
        new_score = (from_kv_obj.evictor_data.score[0], from_kv_obj.evictor_data.score[1])
        to_kv_obj.set_evictor_data(ItemHeapWrapper(to_kv_obj, new_score))
        self._item_heap.push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]
    def evict(self, local_backend_no: int):
        # local_backend_no not used here, since it is per device evictor.
        return [(EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap())]
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        pass


class LFUEvictor(BaseEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list = None):
        # Frequency then recency, then prefix length.
        # Min get popped first.
        # (f, r, -prefix_token_len)
        for kv in chunk_kv:
            prefix_token_len = kv.prefix_token_len
            # Note that evictor_data should not accessible from evictor data structure after poped.
            # So gc will collect them after all of them kv item removed.
            if kv.evictor_data is None:
                kv.set_evictor_data(ItemHeapWrapper(kv, (0, timepoint, -prefix_token_len)))
                self._item_heap.push_heap(kv.evictor_data)
            kv = kv.evictor_data
            kv.set_score((kv.score[0] + 1, timepoint, -prefix_token_len))
            self._item_heap.update_on_keychange(kv)
        return EvictOpType.NONE, None
    def update_on_put(self, chunk_kv: list, timepoint: float):
        return self.update_on_get(chunk_kv, timepoint)
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        raise NotImplementedError("LFU does not transform.")
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        assert to_kv_obj.evictor_data is None
        assert from_kv_obj.evictor_data is not None, f"{from_kv_obj._id} evictor data is None."
        new_score = (from_kv_obj.evictor_data.score[0],
                     from_kv_obj.evictor_data.score[1], from_kv_obj.evictor_data.score[2])
        to_kv_obj.set_evictor_data(ItemHeapWrapper(to_kv_obj, new_score))
        self._item_heap.push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]
    def evict(self, local_backend_no: int):
        return [(EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap())]
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        pass
    
class LFUEvictorV2(LFUEvictor):
    # Consider on demand compression.
    # Actually compress then evict if evicting consequentively.
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        assert from_kv_obj.evictor_data is not None
        assert to_kv_obj.evictor_data is None
        frequency = from_kv_obj.evictor_data.score[0]
        prefix_token_len = from_kv_obj.prefix_token_len
        new_score = (frequency, timepoint, -prefix_token_len)
        to_kv_obj.set_evictor_data(ItemHeapWrapper(to_kv_obj, new_score))
        self._item_heap.push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    # This actually does a compress and evict if calling evict twice.
    def evict(self, local_backend_no: int):
        kv_obj = self._item_heap.pop_heap()
        if kv_obj.compression_level == 0:
            return [(EvictOpType.COMPRESS, (kv_obj, 1))]
        else:
            return [(EvictOpType.WRITE_TO_LOWER, kv_obj)]

class LFUEvictorAllCompress(LFUEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        # 0 layer, compress to 1.
        return [(0, 1) for _ in range(len(chunk_kv_query))]

class OurEvictorV1(BaseEvictor):
    def __init__(self, threshold: int):
        super().__init__()
        self._threshold = threshold
        self._heaps: List[ItemHeap] = [ItemHeap(), ItemHeap()]
    # NOTE: Call update before decode && after decode for those frequency to be correct.
    # NOTE: I think do need to inherent frequency? Check this.
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list = None):
        for kv in chunk_kv:
            prefix_token_len = kv.prefix_token_len
            compression_level = kv.compression_level
            if kv.evictor_data is None:
                kv.set_evictor_data(ItemHeapWrapper(kv, (0, timepoint, -prefix_token_len)))
                self._heaps[compression_level].push_heap(kv.evictor_data)
            kv = kv.evictor_data
            kv.set_score((kv.score[0] + 1, timepoint, -prefix_token_len))
            self._heaps[compression_level].update_on_keychange(kv)
        return EvictOpType.NONE, None
    
    def update_on_put(self, chunk_kv: List, timepoint: float):
        return self.update_on_get(chunk_kv, timepoint)

    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        # From kv obj must have been inside heap, so it must have evictor data.
        # To kv obj must be new, cos we do not compress it twice in this policy.
        assert from_kv_obj.evictor_data is not None
        assert to_kv_obj.evictor_data is None
        frequency = from_kv_obj.evictor_data.score[0]
        prefix_token_len = from_kv_obj.prefix_token_len
        new_score = (frequency, timepoint, -prefix_token_len)
        to_kv_obj.set_evictor_data(ItemHeapWrapper(to_kv_obj, new_score))
        self._heaps[to_kv_obj.compression_level].push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        assert to_kv_obj.evictor_data is None
        assert from_kv_obj.evictor_data is not None
        new_score = (from_kv_obj.evictor_data.score[0],
                     from_kv_obj.evictor_data.score[1], from_kv_obj.evictor_data.score[2])
        to_kv_obj.set_evictor_data(ItemHeapWrapper(to_kv_obj, new_score))
        self._heaps[to_kv_obj.compression_level].push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]

    def evict(self, local_backend_no: int):
        if self._heaps[0].size() > 0:
            top_item = self._heaps[0].top()
            if top_item.prefix_token_len >= self._threshold:
                return [(EvictOpType.COMPRESS, (self._heaps[0].pop_heap(), 1))]
            else:
                return [(EvictOpType.WRITE_TO_LOWER, self._heaps[0].pop_heap())]
        else:
            assert self._heaps[0].size() == 0
            assert self._heaps[1].size() > 0
            return [(EvictOpType.WRITE_TO_LOWER, self._heaps[1].pop_heap())]
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        pass
        


class BaseEstimator:
    def __init__(self) -> None:
        pass
    @abstractmethod
    def access(self, key):
        pass
    @abstractmethod
    def get(self, key):
        pass

# Estimator only gives f. It does not manage the heap.
class WindowLFUEstimator(BaseEstimator):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self._request_queue = deque()
        self._content_to_cnt = {}
        assert window_size > 0
        self._window_size = window_size
    def access(self, content_hash):
        assert len(self._request_queue) <= self._window_size
        if len(self._request_queue) == self._window_size:
            poped_hash = self._request_queue.popleft()
            assert poped_hash in self._content_to_cnt
            self._content_to_cnt[poped_hash] -= 1
            if self._content_to_cnt[poped_hash] == 0:
                del self._content_to_cnt[poped_hash]
        if content_hash not in self._content_to_cnt:
            self._content_to_cnt[content_hash] = 1
        else:
            self._content_to_cnt[content_hash] += 1
        self._request_queue.append(content_hash)
    def get(self, content_hash):
        return self._content_to_cnt.get(content_hash, 0)
        
class ItemEstimatorWrapper(ItemHeapWrapper):
    def __init__(self, item, estimator, prefix_token_len: int):
        super().__init__(item, 0)
        self._estimator = estimator
        self._prefix_token_len = prefix_token_len
    @property
    def score(self):
        # score is overwritten
        return (self._estimator.get(self.item.hash_value), -self._prefix_token_len)

class BaseOursAlpha:
    def __init__(self):
        pass
    @abstractmethod
    def alpha(self) -> float:
        pass
    @abstractmethod
    def update(self, aux):
        pass

class FixedOursAlpha(BaseOursAlpha):
    def __init__(self, initial_alpha: float):
        super().__init__()
        self._alpha = initial_alpha
    def alpha(self) -> float:
        return self._alpha
    def update(self, aux):
        pass
        

# Alpha is fixed, we can get optimal opeation for every L in advance.
# Assume only storing full chunks.
# NOTE: Only used in fast device with a lower storage.
# For lowest level, fall back to LFU.
# This is like a template, we can configure alpha and estimator.
class OursGlobalFrameworkEvictor(BaseEvictor):
    def __init__(self, alpha: BaseOursAlpha, estimator: BaseEstimator, 
                 compression_manager,
                 thputs: List[float],
                 storage_sizes: List[int],
                 optimize_on_hit: bool,
                 store_policy: str,
                 chunk_byte_size: int):
        super().__init__()
        assert len(thputs) > 0
        assert chunk_byte_size > 0
        self._alpha = alpha
        self._compression_manager = compression_manager
        self._estimator = estimator
        self._thputs = thputs
        self._storage_sizes = storage_sizes
        self._optimize_on_hit = optimize_on_hit
        self._store_policy_str = store_policy.lower()
        if self._store_policy_str == "fast_device":
            self._store_policy = StorePolicy.FAST_DEVICE
        else:
            raise NotImplementedError(f"Not implemented store policy {store_policy}")
        # Ours operation include eviction && compress to higher.
        self._alpha_thresholds = []


        self._compact_list = self.construct_compact_list(compression_manager, thputs, chunk_byte_size)
        self._level_no_to_delay_and_quality = self.construct_level_no_to_delay_and_quality(self._compact_list)
        self._optimal_ops = self.update_optimal_ops(self._alpha.alpha(), self._compact_list)
        self._store_compress_level = self.construct_store_compress_level(self._alpha.alpha(), self._compact_list)
        # Select the least predicted f from estimator.
        self._heaps = self.construct_heaps(len(thputs), compression_manager.get_level_set())
        self._evict_called_cnt = 0
        import atexit
        atexit.register(self.print_stats)

        self._alpha_thresholds.sort()
        # print(f"alpha_thresholds: {self._alpha_thresholds}")
        self._recommended_alphas = []
        if len(self._alpha_thresholds) > 0:
            self._recommended_alphas.append(self._alpha_thresholds[0] / 2)
        for i in range(1, len(self._alpha_thresholds) - 1):
            alpha_val = self._alpha_thresholds[i]
            next_alpha_val = self._alpha_thresholds[i + 1]
            self._recommended_alphas.append((alpha_val + next_alpha_val)/2)
        if len(self._alpha_thresholds) > 0:
            self._recommended_alphas.append(self._alpha_thresholds[-1] + 1.0)
        self._recommended_alphas = [str(rec_a) for rec_a in self._recommended_alphas]
        # print("recommended alphas:")
        # print(" ".join(self._recommended_alphas))

    def print_stats(self):
        # print(f"Evict called {self._evict_called_cnt} times.")
        pass
    def construct_store_compress_level(self, alpha: float, global_compact_list: list):
        store_compress_level = []
        for compact_list in global_compact_list:
            max_u = None
            max_l = None
            for level_no, delay_fast, delay_slow, quality in compact_list:
                utility = self._item_utility_ratio(alpha, delay_fast, quality)
                if max_u is None or utility > max_u:
                    max_u = utility
                    max_l = level_no
            store_compress_level.append(max_l)
        # print(f"store_compress_level: {store_compress_level}")
        for compact_list in global_compact_list:
            for i  in range(0, len(compact_list)):
                d1 = compact_list[i][1]
                q1 = compact_list[i][3]
                for j in range(i + 1, len(compact_list)):
                    d2 = compact_list[j][1]
                    q2 = compact_list[j][3]
                    assert d1 > d2
                    assert q1 > q2
                    thres_alpha = (q1 - q2) / (d1 - d2)
                    self._alpha_thresholds.append(thres_alpha)
        return store_compress_level

    def construct_level_no_to_delay_and_quality(self, global_compact_list: list):
        level_no_to_delay_and_quality = []
        for compact_list in global_compact_list:
            temp_dict = {}
            for level_no, delay_fast, delay_slow, quality in compact_list:
                temp_dict[level_no] = (delay_fast, delay_slow, quality)
            level_no_to_delay_and_quality.append(temp_dict)
        return level_no_to_delay_and_quality

    def construct_compact_list(self, compression_manager, thputs: List[float], chunk_byte_size: int):
        global_compact_list = []
        for local_backend_no in range(len(thputs)):
            # NOTE: This is local_backend_no - 1
            compact_list = []
            level_no_set = set()
            fast_thput = thputs[local_backend_no]
            slow_thput = None if local_backend_no == len(thputs) - 1 else thputs[local_backend_no + 1]
            assert fast_thput > 0.0
            if slow_thput is not None:
                assert slow_thput > 0.0
                assert slow_thput < fast_thput
            for item in compression_manager.get_all():
                level_no, ratio, quality, encode_cost, decode_cost = item
                assert encode_cost >= 0.0
                assert decode_cost >= 0.0
                new_chunk_byte_size = chunk_byte_size * ratio
                new_delay_fast = new_chunk_byte_size / fast_thput
                new_delay_slow = new_chunk_byte_size / slow_thput if slow_thput is not None else None
                # print(f"Zero point alpha for {local_backend_no} compression level {level_no} is {quality/new_delay_fast}")
                if new_delay_slow is not None:
                    assert new_delay_fast < new_delay_slow
                compact_list.append((level_no, new_delay_fast, new_delay_slow, quality))
                assert level_no not in level_no_set
                level_no_set.add(level_no)
            compact_list.sort(key=lambda x: x[0])
            print(f"compact_list {compact_list}")
            for i in range(1, len(compact_list)):
                assert compact_list[i][1] < compact_list[i-1][1]
                if compact_list[i][2] is not None:
                    assert compact_list[i][2] < compact_list[i-1][2]
                assert compact_list[i][3] < compact_list[i-1][3]
            
            global_compact_list.append(compact_list)
        return global_compact_list


    def _item_utility_ratio(self, alpha: float, delay: float, quality: float):
        return -alpha * delay + quality
    
    def _item_utility(self, alpha: float, frequency, delay: float, quality: float):
        return frequency * self._item_utility_ratio(alpha, delay, quality)

    def update_optimal_ops(self, alpha: float, global_compact_list: list):
        print(f"alpha in oursframework: {alpha}")
        max_quality_drop = self._compression_manager.get_max_quality_drop()
        optimal_ops_list = []
        for evictor_backend_no, compact_list in enumerate(global_compact_list):
            optimal_ops = {}
            for i in range(len(compact_list)):
                level_no, delay_fast, delay_slow, quality = compact_list[i]
                assert level_no is not None
                assert delay_fast is not None
                assert delay_fast > 0.0
                if delay_slow is None:
                    assert evictor_backend_no == len(global_compact_list) - 1
                evict_score = None
                if delay_slow is not None:
                    evict_score = self._item_utility_ratio(alpha, delay_slow, quality)
                assert level_no not in optimal_ops
                max_compress_score = None
                max_compress_level = None
                max_compress_level_quality = None
                for j in range(i + 1, len(compact_list)):
                    level_no_j, delay_fast_j, delay_slow_j, quality_j = compact_list[j]
                    if delay_slow is not None:
                        alpha_thres = (quality - quality_j) / (delay_slow - delay_fast_j)
                    self._alpha_thresholds.append(alpha_thres)
                    # level_no should have been sorted.
                    assert level_no_j > level_no
                    assert delay_fast_j < delay_fast
                    if delay_slow_j is None:
                        assert delay_slow is None
                    else:
                        assert delay_slow_j < delay_slow
                    assert quality_j < quality
                    this_score = self._item_utility_ratio(alpha, delay_fast_j, quality_j)
                    if max_compress_score is None:
                        max_compress_score = this_score
                        max_compress_level = level_no_j
                        max_compress_level_quality = quality_j
                    else:
                        if this_score > max_compress_score:
                            max_compress_score = this_score
                            max_compress_level = level_no_j
                            max_compress_level_quality = quality_j
                if max_compress_level is not None:
                    if evict_score is not None:
                        if evict_score > max_compress_score:
                            optimal_ops[level_no] = -1
                        else:
                            # Check again for how much quality is dropped by ratio.
                            ratio_drop = (quality - max_compress_level_quality) / max_quality_drop
                            if ratio_drop > 0.5:
                                optimal_ops[level_no] = -1
                            else: 
                                optimal_ops[level_no] = max_compress_level
                    else:
                        assert delay_slow is None
                        optimal_ops[level_no] = max_compress_level
                else:
                    # Write to lower or drop.
                    optimal_ops[level_no] = -1
            optimal_ops_list.append(optimal_ops)
        print(f"optimal_ops_list: {optimal_ops_list}")
        return optimal_ops_list
    
    def construct_heaps(self, backend_num: int, level_no_set: set):
        return [{l_no: ItemHeap() for l_no in level_no_set} for _ in range(backend_num)]

    # NOTE: The only state change to evictor is and should be heaps.
    # TODO: Check immediate increase score situations in store, prefer to do that.
    # But now there should not be such an occasion at all.
    def evict(self, local_backend_no: int):
        self._evict_called_cnt += 1
        # It is a global evictor.
        assert local_backend_no > 0, f"Not supporting managing local backend {local_backend_no}"
        # print(f"Evicting from {local_backend_no}")
        evictor_backend_no = local_backend_no - 1
        # print(f"Evicting from evictor backend no {evictor_backend_no}")
        compact_list = self._compact_list[evictor_backend_no]
        if evictor_backend_no == len(self._thputs) - 1:
            # If in the last layer, always prefer compress.
            # This is consistent with optimal_ops.
            max_compress_score = None
            selected_to_compress_level = None
            evict_item_compress_level = None
            level_cnt = 0
            for level_no, delay_fast, delay_slow, quality in compact_list:
                level_cnt += 1
                assert delay_slow is None
                assert delay_fast is not None
                assert quality is not None
                heap = self._heaps[evictor_backend_no][level_no]
                if heap.size() == 0:
                    continue
                level_best_op = self._optimal_ops[evictor_backend_no][level_no]
                if level_best_op < 0:
                    assert level_best_op == -1
                    assert level_cnt == len(compact_list), f"Always prefer compress in the last layer."
                    # Always prefer compress, so must be impossible to compress.
                    evict_item_compress_level = level_no
                else:
                    assert level_best_op > level_no
                    to_delay_fast, to_delay_slow, to_quality = \
                    self._level_no_to_delay_and_quality[evictor_backend_no][level_best_op]
                    assert to_delay_slow is None
                    this_score = self._item_utility(self._alpha.alpha(),
                                                    self._estimator.get(heap.top().hash_value),
                                                    to_delay_fast, 
                                                    to_quality)
                    if max_compress_score is None or this_score > max_compress_score:
                        max_compress_score = this_score
                        selected_to_compress_level = level_best_op
                        evict_item_compress_level = level_no
            if selected_to_compress_level is None:
                assert evict_item_compress_level is not None, f"evict in last layer and no item to evict."
                return EvictOpType.DROP, self._heaps[evictor_backend_no][evict_item_compress_level].pop_heap()
            else:
                return EvictOpType.COMPRESS, \
                (self._heaps[evictor_backend_no][evict_item_compress_level].pop_heap(), selected_to_compress_level)

        assert evictor_backend_no < len(self._thputs) - 1
        # print("Not in the last layer.")
        min_score_drop = None
        current_evict_item_compress_level = None
        selected_to_compress_level = None
        for level_no, delay_fast, delay_slow, quality in compact_list:
            heap = self._heaps[evictor_backend_no][level_no]
            if heap.size() == 0:
                continue
            level_best_op = self._optimal_ops[evictor_backend_no][level_no]
            ori_score = self._item_utility(self._alpha.alpha(),
                                           self._estimator.get(heap.top().hash_value),
                                           delay_fast,
                                           quality)
            # print(f"Level {level_no} best op {level_best_op} in {evictor_backend_no}")
            if level_best_op < 0:
                assert level_best_op == -1
                this_score = self._item_utility(self._alpha.alpha(),
                                                self._estimator.get(heap.top().hash_value),
                                                delay_slow, 
                                                quality)
                score_drop = ori_score - this_score
                assert score_drop >= 0, f"score drop {score_drop} ori_score {ori_score} this_score {this_score}, for fixed alpha, should have drop > 0."
                if min_score_drop is None or score_drop < min_score_drop:
                    min_score_drop = score_drop
                    current_evict_item_compress_level = level_no
                    selected_to_compress_level = None
            else:
                assert level_best_op > level_no
                to_delay_fast, to_delay_slow, to_quality = \
                self._level_no_to_delay_and_quality[evictor_backend_no][level_best_op]
                this_score = self._item_utility(self._alpha.alpha(),
                                                self._estimator.get(heap.top().hash_value),
                                                to_delay_fast, 
                                                to_quality)
                score_drop = ori_score - this_score
                assert score_drop >= 0, f"score drop {score_drop} ori_score {ori_score} this_score {this_score}, for fixed alpha, should have drop > 0."
                if min_score_drop is None or score_drop < min_score_drop:
                    min_score_drop = score_drop
                    current_evict_item_compress_level = level_no
                    selected_to_compress_level = level_best_op
        assert current_evict_item_compress_level is not None
        if selected_to_compress_level is not None:
            return [(EvictOpType.COMPRESS, \
            (self._heaps[evictor_backend_no][current_evict_item_compress_level].pop_heap(), selected_to_compress_level))]
        else:
            assert current_evict_item_compress_level is not None
            return [(EvictOpType.WRITE_TO_LOWER, self._heaps[evictor_backend_no][current_evict_item_compress_level].pop_heap())]
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        pass

    '''
    # NOTE: Now only for update_on_hit.
    # Keep the original compression level to reduce search space.
    class TryPutFast:
        class TryPutOpType(Enum):
            INSERT = 0
            REMOVE = 1
        def __init__(self, evictor, space_list) -> None:
            self._results = None
            self._already_false = False
            self._idx = -1
            # (op_type, device_level, compress_level, evictor_data)
            self._revert_queue = []
            self._kv_list = None
            self._evictor = evictor
            self._space_list = space_list
            self._fake_obj_set = set()
        def _begin_run(self, kv_chunks_check_list: list):
            self._results = [False] * len(kv_chunks_check_list)
            self._kv_list = kv_chunks_check_list
            # kv_chunks should either be all in slow device.
            assert all([kv.device_idx.local_backend_no == 2 for kv in kv_chunks_check_list])
            self._idx = 0
            self._already_false = False
        def _step(self) -> bool:
            if self._idx >= len(self._results) or self._already_false:
                return False
            # Insert into fast device.
            target_kv_obj = self._kv_list[self._idx]
            fake_kv_obj = copy.deepcopy(target_kv_obj)
            fake_evictor_data = ItemEstimatorWrapper(fake_kv_obj, self._evictor._estimator, fake_kv_obj.prefix_token_len)
            fake_kv_obj.set_evictor_data(fake_evictor_data)
            self._evictor._heaps[0][fake_kv_obj.compression_level].push_heap(fake_evictor_data)
            assert fake_kv_obj not in self._fake_obj_set
            self._fake_obj_set.add(fake_kv_obj)
            self._revert_queue.append((self.TryPutOpType.INSERT, 0, fake_kv_obj.compression_level, fake_evictor_data))
            # Try to make space for.
            need_space_size = fake_kv_obj.size
            if need_space_size <= self._space_list[0]:
                self._results[self._idx] = True
                self._idx += 1
                return True
            while need_space_size > self._space_list[0]:
                evict_op_type, evict_item = self._evictor.evict(0)
                if evict_item in self._fake_obj_set:
                    self._already_false = True
                    return False
                if evict_op_type == EvictOpType.WRITE_TO_LOWER:
                    self._revert_queue.append((self.TryPutOpType.REMOVE, 0, evict_item.compression_level, evict_item))
                    self._revert_queue.append((self.TryPutOpType.INSERT, 1, evict_item.compression_level, evict_item))
                    need_space_size -= evict_item.size
                else:
                    assert evict_op_type == EvictOpType.COMPRESS
                    self._revert_queue.append((self.TryPutOpType.REMOVE, 0, evict_item[0].compression_level, evict_item[0]))
                    self._revert_queue.append((self.TryPutOpType.INSERT, 0, evict_item[1], evict_item[0]))
                    need_space_size -= evict_item[0].size

            self._idx += 1
        def _run_to_end(self):
            while self._step():
                pass
        def _restore_heap(self):
            raise NotImplementedError("Not implemented.")
        def result(self, kv_chunks_check_list: list):
            self._begin_run(kv_chunks_check_list)
            self._run_to_end()
            self._restore_heap()
            return self._results
    '''
    

    # NOTE: Can return NONE or SWAP, fetch is a special case of swap.
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list = None):
        # 1. Update frequency of every kv_obj, update their score.
        # Assume that they are updated all at once, then 
        # the first chunk is always more possible to get fetched.
        # So update in sequential order.
        assert space_list is not None
        swap_out = []
        swap_in = []
        kv_chunks_to_check_swap = []
        for kv in chunk_kv:
            # kv is kv_obj, not kv_query.
            device_idx = kv.device_idx
            local_backend_no = device_idx.local_backend_no
            evictor_backend_no = local_backend_no - 1
            assert evictor_backend_no in [0, 1]
            assert local_backend_no in [1, 2]
            assert kv.evictor_data is not None
            self._estimator.access(kv.hash_value)
            self._heaps[evictor_backend_no][kv.compression_level].update_on_keychange(kv.evictor_data)
            if local_backend_no == 2:
                # Consider putting into fast device.
                kv_chunks_to_check_swap.append(kv)

        if self._optimize_on_hit:
            # TODO: Optimize placements here.
            # Dry run AS A WHOLE to optimize placements by swapping.
            # Revert the changes to the heap before returning and let the cache engine finally issue them.
            raise NotImplementedError("Optimize on hit is not implemented.")

        if len(swap_in) == 0:
            assert len(swap_out) == 0
            return EvictOpType.NONE, None
        else:
            return EvictOpType.SWAP, (swap_out, swap_in)
    
    # NOTE: Cache engine will call get_store_info
    # then call evict to make space.
    # then call put to backend, then update_on_put.
    # And note that skipped chunks should have been updated in get.
    def update_on_put(self, chunk_kv: list, timepoint: float):
        # Should have constructed kv_obj according to the query about 
        # where to put them.
        for kv in chunk_kv:
            device_idx = kv.device_idx
            local_backend_no = device_idx.local_backend_no
            evictor_backend_no = local_backend_no - 1
            assert evictor_backend_no in [0, 1], f"{evictor_backend_no} not in [0, 1]"
            assert local_backend_no in [1, 2]
            self._estimator.access(kv.hash_value)
            if kv.evictor_data is None:
                kv.set_evictor_data(ItemEstimatorWrapper(kv, self._estimator, kv.prefix_token_len))
                # print(f"Push to heap {evictor_backend_no} {kv.compression_level} in put.")
                self._heaps[evictor_backend_no][kv.compression_level].push_heap(kv.evictor_data)
            else:
                self._heaps[evictor_backend_no][kv.compression_level].update_on_keychange(kv.evictor_data)
        return EvictOpType.NONE, None
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        assert from_kv_obj.evictor_data is not None
        assert to_kv_obj.evictor_data is None
        # frequency is naturally inherited by the same hash_value.
        to_kv_obj.set_evictor_data(ItemEstimatorWrapper(to_kv_obj, self._estimator, to_kv_obj.prefix_token_len))
        to_device_idx = to_kv_obj.device_idx
        local_backend_no = to_device_idx.local_backend_no
        evictor_backend_no = local_backend_no - 1
        assert evictor_backend_no in [0, 1]
        # print(f"Push to heap {evictor_backend_no} {to_kv_obj.compression_level} in transfer/transform.")
        self._heaps[evictor_backend_no][to_kv_obj.compression_level].push_heap(to_kv_obj.evictor_data)
        return EvictOpType.NONE, None
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        # We do not use timepoint here, the same.
        return self.update_on_transfer(from_kv_obj, to_kv_obj)

    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        if self._store_policy == StorePolicy.FAST_DEVICE:
            # NOTE: Now always put to fast device, but not always full version.
            # Store to fast device for now.
            cl = self._store_compress_level[0]
            '''
            # Check fast device for its space.
            fast_space = space_list[0]
            total_size = sum([kv.size for kv in chunk_kv_query])
            if total_size > fast_space:
                # print(f"Total size {total_size} > fast space {fast_space}")
                the_level = cl
                # Compress to before evict directly.
                for level in range(cl, len(self._level_no_to_delay_and_quality[0])):
                    if self._optimal_ops[0][level] < 0:
                        the_level = level
                        break
                # print(f"Store to fast device, compress to {the_level} > {cl}")
                return [(0, the_level) for _ in range(len(chunk_kv_query))]
            else:
                return [(0, cl) for _ in range(len(chunk_kv_query))]
            '''
            return [(0, cl) for _ in range(len(chunk_kv_query))]
        else:
            # TODO: Use a similar method with optimize_on_hit.
            raise NotImplementedError(f"Not implemented store policy {self._store_policy_str}")

class ItemSuperChunkWrapper(ItemHeapWrapper):
    # 1. Every update, update every chunk in the super chunk.
    # 2. Still index and access by chunk level, just summarize the frequency.
    def __init__(self, super_chunk):
        super().__init__(super_chunk, 0)
    @property
    def score(self):
        return self.item.score

# TODO: A problem on how we combine f with score_drop_ratio to get the final utility.
# TODO: How to search the best efficiently.
# Make a min heap of best utility drop for every layer. And update on every change.
class SuperChunk:
    '''
    Every chunk in the super chunk is BELIEVED to be accessed together.
    So we record one frequency for the super chunk.
    And we can divide it later.
    Before we divide it, frequency and quality should both be the same.
    '''
    def __init__(self, thput, evicted_thput, estimator, alpha: BaseOursAlpha, 
                 compression_levels, compress_level: int, total_size: int, content_hash_list: list, super_chunk_len: int):
        assert thput > 0.0
        self._total_size = total_size
        self._content_hash_list = content_hash_list
        # items can be replaced, but hash will not.
        self._thput = thput
        self._evicted_thput = evicted_thput
        self._super_chunk_len = super_chunk_len
        assert evicted_thput is None or evicted_thput > 0.0
        self._delay = self._total_size / thput
        self._compression_levels = compression_levels
        # A list of (ratio, quality, encode_cost, decode_cost)
        self._compress_level = compress_level
        quality_should_be = self._compression_levels[self._compress_level][1]
        self._quality = quality_should_be
        self._alpha = alpha
        self._estimator = estimator
        self._best_op, self._best_drop_ratio = self._get_best_drop_ratio()
        self._the_wrapper = ItemSuperChunkWrapper(self)
    @property
    def wrapper(self):
        return self._the_wrapper
    @property
    def total_size(self):
        return self._total_size
    @property
    def compression_level(self):
        return self._compress_level
    @property
    def best_op(self):
        return self._best_op
    @property
    def utility(self):
        freq = self.frequency
        return freq * self.utility_ratio
    @property
    def utility_ratio(self):
        alpha_value = self._alpha.alpha()
        return - alpha_value * self._delay + self._quality
    @property
    def super_chunk_len(self):
        return self._super_chunk_len
    @property
    def content_hash_list(self):
        return self._content_hash_list
    def _get_best_drop_ratio(self):
        current_u_ratio = self.utility_ratio
        evicted_delay = None if self._evicted_thput is None else self._total_size / self._evicted_thput
        evicted_u_ratio = None if evicted_delay is None else - self._alpha.alpha() * evicted_delay + self._quality
        best_drop = None if evicted_u_ratio is None else current_u_ratio - evicted_u_ratio
        best_compress_level = None
        best_is_evict = evicted_u_ratio is not None
        # Then compress levels.
        if self._compress_level < len(self._compression_levels) - 1:
            for idx in range(self._compress_level + 1, len(self._compression_levels)):
                ratio, quality, encode_cost, decode_cost = self._compression_levels[idx]
                level_no = idx
                assert encode_cost >= 0.0
                assert decode_cost >= 0.0
                assert level_no > self._compress_level
                compressed_total_size = self._total_size * ratio
                compressed_delay = compressed_total_size / self._thput
                compressed_quality = quality
                compressed_u_ratio = - self._alpha.alpha() * compressed_delay + compressed_quality
                compressed_drop = current_u_ratio - compressed_u_ratio
                if best_drop is None or compressed_drop < best_drop:
                    # NOTE: best_drop is min drop.
                    # For last layer, always prefer compress.
                    best_drop = compressed_drop
                    best_compress_level = level_no
                    best_is_evict = False
        # The actual drop will be best_drop_ratio * freq.
        if not best_is_evict:
            if best_compress_level is None:
                return tuple([EvictOpType.DROP]), best_drop
            else:
                return tuple([EvictOpType.COMPRESS, best_compress_level]), best_drop
        else:
            return tuple([EvictOpType.WRITE_TO_LOWER]), best_drop
        
    @property
    def score(self):
        return self._best_drop_ratio * self.frequency

    def get_first_hash(self):
        assert len(self._content_hash_list) > 0
        return self._content_hash_list[0]
    @property
    def frequency(self):
        return self._estimator.get(self.get_first_hash())
    
    def reinit_on_kv_tuple(self, new_length, new_size):
        assert new_length > 0
        self._content_hash_list = self._content_hash_list[:new_length]
        self._super_chunk_len = new_length
        self._total_size = new_size
        self._delay = self._total_size / self._thput
        self._best_op, self._best_drop_ratio = self._get_best_drop_ratio()

    def reinit_on_compress(self, new_compress_level: int, new_total_size: int):
        assert new_compress_level != self._compress_level
        self._compress_level = new_compress_level
        self._total_size = new_total_size
        self._delay = self._total_size / self._thput
        # items might have not be changed.
        self._quality = self._compression_levels[self._compress_level][1]
        self._best_op, self._best_drop_ratio = self._get_best_drop_ratio()

    def reinit_on_write_to_lower_or_drop(self, next_thput):
        self._thput = self._evicted_thput
        self._evicted_thput = next_thput
        if self._thput is None:
            assert self._evicted_thput is None
            return
        self._delay = self._total_size / self._thput
        self._best_op, self._best_drop_ratio = self._get_best_drop_ratio()
    
    def get_and_divide(self, differ_or_trunc_idx: int):
        # It is caused by a reuse in the middle.
        # differ_idx is where it differs or truncates.
        # Do not call it if reuse exceeds(extend) or just full.
        # Only remap the second chunks.
        assert differ_or_trunc_idx > 0 and differ_or_trunc_idx < self._super_chunk_len
        # All items in the chunk is the same level.
        assert self._total_size % self._super_chunk_len == 0
        ori_total_size = self._total_size
        ori_len = self._super_chunk_len
        unit_size = self._total_size // self._super_chunk_len
        divided_size = unit_size * differ_or_trunc_idx
        next_hash_list = self._content_hash_list[differ_or_trunc_idx:]
        self.reinit_on_kv_tuple(differ_or_trunc_idx, divided_size)
        next_total_size = ori_total_size - divided_size
        next_len = ori_len - differ_or_trunc_idx
        # print(f"ori_len {ori_len} next_len {next_len}")
        second_super_chunk = SuperChunk(self._thput, self._evicted_thput, self._estimator, self._alpha,
                                        self._compression_levels, self._compress_level, next_total_size, 
                                        next_hash_list, next_len)
        # NOTE: Remap outside.
        return second_super_chunk
    
    def full_hit(self):
        for hash_value in self._content_hash_list:
            self._estimator.access(hash_value)
    def extend_in_place(self, extend_list: list):
        assert len(extend_list) > 0
        self.full_hit()
        # NOTE: Remap outside.
        new_length = self._super_chunk_len + len(extend_list)
        assert self._compress_level == extend_list[0].compression_level, f"{self._compress_level} != {extend_list[0].compression_level}"
        assert self._total_size % self._super_chunk_len == 0
        unit_size = self._total_size // self._super_chunk_len
        new_size = unit_size * new_length
        self.reinit_on_kv_tuple(new_length, new_size)
        for kv in extend_list:
            self._content_hash_list.append(kv.hash_value)
        




class OursSuperChunkEvictor(BaseEvictor):
    '''
    Super chunk is several chunk that are always reused together.
    This is to incoperate multi-turn conversation session-level 
    reuse with RAG/system prompt chunk-level reuse.

    Use a partitioning algorithm for chunk-superchunk.
    1. Maintain a chunk --> super chunk(a list of chunks, idx) mapping.
    2. Update super chunk in update_on_get, 
    check if the tail one is the last chunk of a super chunk, if not, 
    divide that super chunk.
    3. Can be extended when used together.
    4. On extend or divide, update delay, quality, frequency and optimal_ops, and score_ratio.
    '''
    # Evict candidates are super chunks.
    def __init__(self, alpha: BaseOursAlpha, estimator: BaseEstimator, 
                 compression_manager,
                 thputs: List[float],
                 storage_sizes: List[int],
                 optimize_on_hit: bool,
                 store_policy: str,
                 chunk_byte_size: int):
        self._alpha = alpha
        self._estimator = estimator
        self._compression_manager = compression_manager
        self._thputs = thputs
        self._storage_sizes = storage_sizes
        self._optimize_on_hit = optimize_on_hit
        self._store_policy_str = store_policy.lower()
        self._chunk_byte_size = chunk_byte_size
        self._heaps: List[ItemHeap] = [ItemHeap() for _ in range(len(thputs))]
        self._last_cached_super_chunk = None
        self._compression_levels = []
        get_all_list: list = self._compression_manager.get_all()
        get_all_list.sort()
        for idx, tp in enumerate(get_all_list):
            level_no, ratio, quality, encode_cost, decode_cost = tp
            assert level_no == idx
            self._compression_levels.append((ratio, quality, encode_cost, decode_cost))

        self._from_hash_to_kv_obj = [] # Update on new hash or replace of obj.
        self._pending_operation_counter = 0
        self._is_first_op_on_evict = True
        for _ in range(len(thputs)):
            self._from_hash_to_kv_obj.append({})
    def update_on_get(self, chunk_kv: list, timepoint: float, space_list=None):
        if len(chunk_kv) == 0:
            return EvictOpType.NONE, None
        swap_out = []
        swap_in = []
        kv_chunks_to_check_swap = []
        max_compress_level_now = None
        kv_chunks_by_super_chunk = []
        assert_local_backend_no_same = None
        # print(f"update_on_get with {len(chunk_kv)} chunks.")
        for kv in chunk_kv:
            # print(f"update_on_get filling in [{kv.device_idx.local_backend_no - 1}][{kv.hash_value}]")
            self._from_hash_to_kv_obj[kv.device_idx.local_backend_no - 1][kv.hash_value] = kv
            # kv is kv_obj, not kv_query.
            device_idx = kv.device_idx
            local_backend_no = device_idx.local_backend_no
            evictor_backend_no = local_backend_no - 1
            assert evictor_backend_no in [0, 1]
            assert local_backend_no in [1, 2]
            assert kv.evictor_data is not None
            if max_compress_level_now is None or kv.compression_level > max_compress_level_now:
                max_compress_level_now = kv.compression_level
            if assert_local_backend_no_same is None:
                assert_local_backend_no_same = local_backend_no
            else:
                assert assert_local_backend_no_same == local_backend_no
            if local_backend_no == 2:
                # Consider putting into fast device.
                kv_chunks_to_check_swap.append(kv)
            super_chunk: SuperChunk = kv.evictor_data.item
            kv_chunks_by_super_chunk.append(super_chunk)
            # print(f"Super chunk {super_chunk.super_chunk_len} {super_chunk.total_size}.")
        idx = 0
        while idx < len(kv_chunks_by_super_chunk):
            super_chunk: SuperChunk = kv_chunks_by_super_chunk[idx]
            next_idx = idx + 1
            while next_idx < len(kv_chunks_by_super_chunk) and kv_chunks_by_super_chunk[next_idx] == super_chunk:
                next_idx += 1
            # [idx, next_idx) is one super chunk.
            real_len = next_idx - idx
            assert real_len > 0
            super_chunk_len = super_chunk.super_chunk_len
            assert super_chunk_len > 0
            assert real_len <= super_chunk_len, "Not possible to extend on get."
            new_chunk = None
            local_backend_no = chunk_kv[idx].device_idx.local_backend_no
            remap_last_kv = None
            if super_chunk_len == real_len:
                # Full hit.
                super_chunk.full_hit()
            else:
                # print(f"Divide super chunk {super_chunk_len} to {real_len} and {super_chunk_len - real_len}")
                new_chunk = super_chunk.get_and_divide(real_len)
                for hash_value in new_chunk.content_hash_list:
                    # print(f"[{local_backend_no - 1}][{hash_value}]")
                    remap_kv = self._from_hash_to_kv_obj[local_backend_no - 1][hash_value]
                    remap_kv.set_evictor_data(new_chunk.wrapper)
                    remap_last_kv = remap_kv
            # Update heap.
            evictor_backend_no = local_backend_no - 1
            # For every super chunk, update.
            first_kv = chunk_kv[idx]
            self._heaps[evictor_backend_no].update_on_keychange(first_kv.evictor_data)
            if new_chunk is not None:
                assert remap_last_kv is not None
                last_kv = remap_last_kv
                assert first_kv.evictor_data.item != last_kv.evictor_data.item
                assert last_kv.evictor_data.item == new_chunk
                # print(f"Push to heap {evictor_backend_no} {last_kv.evictor_data.item} in get.")
                self._heaps[evictor_backend_no].push_heap(last_kv.evictor_data)
            idx = next_idx
        if self._optimize_on_hit:
            # TODO: Optimize placements here.
            # Dry run AS A WHOLE to optimize placements by swapping.
            # Revert the changes to the heap before returning and let the cache engine finally issue them.
            raise NotImplementedError("Optimize on hit is not implemented.")

        if len(swap_in) == 0:
            assert len(swap_out) == 0
            return EvictOpType.NONE, None
        else:
            return EvictOpType.SWAP, (swap_out, swap_in)
    
    def update_on_skipped_store(self, chunk_kv: list, timepoint: float):
        # print(f"update_on_skipped_store with {len(chunk_kv)} chunks.")
        for kv in chunk_kv:
            # print(f"update_on_skipped_store filling in [{kv.device_idx.local_backend_no - 1}][{kv.hash_value}]")
            self._from_hash_to_kv_obj[kv.device_idx.local_backend_no - 1][kv.hash_value] = kv
        if len(chunk_kv) > 0:
            self._last_cached_super_chunk = chunk_kv[-1].evictor_data.item
            self._last_super_chunk_evictor_backend_no = chunk_kv[-1].device_idx.local_backend_no - 1
    def update_on_put(self, chunk_kv: list, timepoint: float):
        # NOTE: The hit ones should have been skipped.
        if len(chunk_kv) == 0:
            return EvictOpType.NONE, None
        # print(f"update_on_put with {len(chunk_kv)} chunks.")
        ori_heap_size = self._heaps[chunk_kv[0].device_idx.local_backend_no - 1].size()
        super_chunk = None
        for kv in chunk_kv:
            # print(f"update_on_put filling in [{kv.device_idx.local_backend_no - 1}][{kv.hash_value}]")
            self._from_hash_to_kv_obj[kv.device_idx.local_backend_no - 1][kv.hash_value] = kv
        if self._last_cached_super_chunk is None:
            # No skipped && first put.
            compression_level = chunk_kv[0].compression_level
            # Make sure of this in get_store_info.
            assert all(kv.compression_level == compression_level for kv in chunk_kv)
            local_backend_no = chunk_kv[0].device_idx.local_backend_no
            evictor_backend_no = local_backend_no - 1
            assert evictor_backend_no in [0, 1], f"{evictor_backend_no} not in [0, 1]"
            assert local_backend_no in [1, 2]
            next_thput = None if evictor_backend_no >= len(self._thputs) else self._thputs[evictor_backend_no]
            total_size = sum([kv.size for kv in chunk_kv])
            super_chunk = SuperChunk(self._thputs[evictor_backend_no], next_thput, 
                                     self._estimator, self._alpha, self._compression_levels, 
                                     compression_level, total_size, [kv.hash_value for kv in chunk_kv], len(chunk_kv))
            for kv in chunk_kv:
                kv.set_evictor_data(super_chunk.wrapper)
            # print(f"Push to heap {evictor_backend_no} {super_chunk} in put.")
            self._heaps[evictor_backend_no].push_heap(chunk_kv[0].evictor_data)
        else:
            super_chunk: SuperChunk = self._last_cached_super_chunk
            super_chunk.extend_in_place(chunk_kv)
            # Remap
            for kv in chunk_kv:
                kv.set_evictor_data(super_chunk.wrapper)
            self._heaps[chunk_kv[0].device_idx.local_backend_no - 1].update_on_keychange(chunk_kv[0].evictor_data)
        assert super_chunk is not None
        self._last_cached_super_chunk = super_chunk
        self._last_super_chunk_evictor_backend_no = chunk_kv[-1].device_idx.local_backend_no - 1
        now_heap_size = self._heaps[chunk_kv[0].device_idx.local_backend_no - 1].size()
        # print(f"In put, {chunk_kv[0].device_idx.local_backend_no - 1} heap size {ori_heap_size} --> {now_heap_size}")
        return EvictOpType.NONE, None

    def begin_store(self):
        self._last_cached_super_chunk = None
        self._last_super_chunk_evictor_backend_no = None

    def get_best_compress(self, full_size: int, compress_level_from: int, thput: float):
        best_ratio = None
        best_level = None
        for compress_level in range(compress_level_from, len(self._compression_levels)):
            ratio, quality, encode_cost, decode_cost = self._compression_levels[compress_level]
            assert encode_cost >= 0.0
            assert decode_cost >= 0.0
            compressed_size = full_size * ratio
            utility_ratio = - self._alpha.alpha() * (compressed_size / thput) + quality
            if best_ratio is None or utility_ratio > best_ratio:
                best_ratio = utility_ratio
                best_level = compress_level
        assert best_level is not None
        return best_level

            

    def get_store_info(self, chunk_kv_query: list, timepoint: float, previous_max_level, following_space) -> list:
        # Always store to the same layer as before or fast.
        ret_list = []
        store_to_layer = 0
        p_full = 0
        if self._last_cached_super_chunk is not None:
            p_full = int(self._last_cached_super_chunk.total_size / \
                            self._compression_levels[self._last_cached_super_chunk.compression_level][0])
            if self._last_cached_super_chunk.compression_level > previous_max_level:
                # print(f"max_previous updated from {previous_max_level} to {self._last_cached_super_chunk.compression_level}")
                previous_max_level = self._last_cached_super_chunk.compression_level
        full_size = p_full + following_space
        evictor_backend_no = 0 if self._last_super_chunk_evictor_backend_no is None else self._last_super_chunk_evictor_backend_no
        best_level = self.get_best_compress(full_size, previous_max_level, self._thputs[evictor_backend_no])
        assert best_level >= previous_max_level
        if self._last_cached_super_chunk is not None:
            assert self._last_super_chunk_evictor_backend_no is not None
            store_to_layer = self._last_super_chunk_evictor_backend_no
            if best_level > self._last_cached_super_chunk.compression_level:
                ret_list.append((self._last_cached_super_chunk.super_chunk_len, best_level))
                # print(f"Backward op: {self._last_cached_super_chunk.super_chunk_len} x {self._last_cached_super_chunk.compression_level} --> {best_level}\n")
        ret_list.append((store_to_layer, best_level))
        # print(f"get_store_info {ret_list}")
        if best_level != 0:
            pass
            # print(f"Store to layer {store_to_layer} with {best_level} compression, with max_prev is {previous_max_level}")
        return ret_list
    
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        assert from_kv_obj.evictor_data is not None
        assert to_kv_obj.evictor_data is None
        to_kv_obj.set_evictor_data(from_kv_obj.evictor_data)
        to_device_idx = to_kv_obj.device_idx
        local_backend_no = to_device_idx.local_backend_no
        evictor_backend_no = local_backend_no - 1
        assert evictor_backend_no in [0, 1]
        # print(f"push to heap {evictor_backend_no} {to_kv_obj.evictor_data.item} in transfer.")
        if self._is_first_op_on_evict:
            # Currently all transform or all transfer.
            ori_size = self._heaps[evictor_backend_no].size()
            assert self._pending_operation_counter > 0
            self._is_first_op_on_evict = False
            # print(f"Push to heap {evictor_backend_no} {to_kv_obj.evictor_data.item} in transfer.")
            self._heaps[evictor_backend_no].push_heap(to_kv_obj.evictor_data)
            now_size = self._heaps[evictor_backend_no].size()
            # print(f"In transfer and first op in evict, {evictor_backend_no} heap size {ori_size} --> {now_size}")
        # print(f"update_on_transfer filling in [{evictor_backend_no}][{to_kv_obj.hash_value}]")
        self._from_hash_to_kv_obj[evictor_backend_no][to_kv_obj.hash_value] = to_kv_obj
        self._pending_operation_counter -= 1
        return EvictOpType.NONE, None
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        assert from_kv_obj.evictor_data is not None
        assert to_kv_obj.evictor_data is None
        to_kv_obj.set_evictor_data(from_kv_obj.evictor_data)
        to_device_idx = to_kv_obj.device_idx
        local_backend_no = to_device_idx.local_backend_no
        evictor_backend_no = local_backend_no - 1
        assert evictor_backend_no in [0, 1]
        # print(f"push to heap {evictor_backend_no} {to_kv_obj.evictor_data.item} in transform from {from_kv_obj.compression_level} to {to_kv_obj.compression_level}.")
        if self._is_first_op_on_evict:
            # Currently all transform or all transfer.
            ori_size = self._heaps[evictor_backend_no].size()
            assert self._pending_operation_counter > 0
            self._is_first_op_on_evict = False
            # print(f"Push to heap {evictor_backend_no} {to_kv_obj.evictor_data.item} in transform.")
            self._heaps[evictor_backend_no].push_heap(to_kv_obj.evictor_data)
            now_size = self._heaps[evictor_backend_no].size()
            # print(f"In transform and first op in evict, {evictor_backend_no} heap size {ori_size} --> {now_size}")
        # print(f"update_on_transform filling in [{evictor_backend_no}][{to_kv_obj.hash_value}]")
        self._from_hash_to_kv_obj[evictor_backend_no][to_kv_obj.hash_value] = to_kv_obj
        self._pending_operation_counter -= 1
        return EvictOpType.NONE, None
    
    def evict(self, local_backend_no: int):
        # print(f"\n\n\nEvict called with {local_backend_no}")
        assert self._pending_operation_counter == 0
        self._is_first_op_on_evict = True
        assert local_backend_no > 0, f"Not supporting managing local backend {local_backend_no}"
        ori_size = self._heaps[local_backend_no - 1].size()
        evictor_backend_no = local_backend_no - 1
        assert evictor_backend_no in [0, 1]
        assert self._heaps[evictor_backend_no].size() > 0
        evict_super_chunk_wrapper = self._heaps[evictor_backend_no].pop_heap(True)
        evict_super_chunk = evict_super_chunk_wrapper.item
        assert evict_super_chunk is not None
        if evict_super_chunk == self._last_cached_super_chunk:
            restore_pinned_super_chunk_wrapper = evict_super_chunk_wrapper
            assert self._heaps[evictor_backend_no].size() > 0, f"For now we assume size > 2 x one request,"\
                " for we have to pin some chunks for simplicity."
            evict_super_chunk = self._heaps[evictor_backend_no].pop_heap()
            assert evict_super_chunk != restore_pinned_super_chunk_wrapper.item, f"{evict_super_chunk}"
            # print("Restore pinned super chunk.")
            self._heaps[evictor_backend_no].push_heap(restore_pinned_super_chunk_wrapper)
            # print(f"From {restore_pinned_super_chunk_wrapper.item} to {evict_super_chunk}")
        # Organize into list.
        best_op = evict_super_chunk.best_op
        op_type = best_op[0]
        op_aux = None
        if len(best_op) > 1:
            assert op_type == EvictOpType.COMPRESS
            # print(f"Eviction compress to {best_op[1]}")
            op_aux = best_op[1]
        ret_list = []
        # NOTE: reinit on compress and write_to_lower in advance here.
        if op_type == EvictOpType.COMPRESS:
            assert evict_super_chunk.total_size % evict_super_chunk.super_chunk_len == 0
            unit_size = evict_super_chunk.total_size // evict_super_chunk.super_chunk_len
            unit_compressed_size = int(unit_size * self._compression_levels[op_aux][0])
            new_total_size = unit_compressed_size * evict_super_chunk.super_chunk_len
            # print(f"new compress level {op_aux} new total size {new_total_size}")
            evict_super_chunk.reinit_on_compress(op_aux, new_total_size)
        else:
            assert op_type == EvictOpType.WRITE_TO_LOWER
            next_thput = None if evictor_backend_no >= len(self._thputs) else self._thputs[evictor_backend_no]
            evict_super_chunk.reinit_on_write_to_lower_or_drop(next_thput)
        # Remove.
        # print(f"Pop from heap {evictor_backend_no} {evict_super_chunk} in evict.")
        # print(f"hash len {len(evict_super_chunk.content_hash_list)}")
        for hash_value in evict_super_chunk.content_hash_list:
            # print(f"Popping from [{evictor_backend_no}][{hash_value}]")
            kv_obj = self._from_hash_to_kv_obj[evictor_backend_no].pop(hash_value)
            assert kv_obj is not None
            if op_type == EvictOpType.COMPRESS:
                assert op_aux is not None
                ret_list.append((EvictOpType.COMPRESS, (kv_obj, op_aux)))
            else:
                assert op_aux is None
                ret_list.append((op_type, kv_obj))
        self._pending_operation_counter = len(ret_list)
        now_size = self._heaps[local_backend_no - 1].size()
        # print(f"Evict, {local_backend_no - 1} heap size {ori_size} --> {now_size}")
        return ret_list