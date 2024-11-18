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
    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
        # Instruct cache engine to put into cache.
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
    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]
    def evict(self, local_backend_no: int):
        # local_backend_no not used here, since it is per device evictor.
        return EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap()


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
    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]
    def evict(self, local_backend_no: int):
        return EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap()
    
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
            return EvictOpType.COMPRESS, (kv_obj, 1)
        else:
            return EvictOpType.WRITE_TO_LOWER, kv_obj

class LFUEvictorAllCompress(LFUEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
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
    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
        return [(0, 0) for _ in range(len(chunk_kv_query))]

    def evict(self, local_backend_no: int):
        if self._heaps[0].size() > 0:
            top_item = self._heaps[0].top()
            if top_item.prefix_token_len >= self._threshold:
                return EvictOpType.COMPRESS, (self._heaps[0].pop_heap(), 1)
            else:
                return EvictOpType.WRITE_TO_LOWER, self._heaps[0].pop_heap()
        else:
            assert self._heaps[0].size() == 0
            assert self._heaps[1].size() > 0
            return EvictOpType.WRITE_TO_LOWER, self._heaps[1].pop_heap()
        


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
        print(f"store_compress_level: {store_compress_level}")
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
                    else:
                        if this_score > max_compress_score:
                            max_compress_score = this_score
                            max_compress_level = level_no_j
                if max_compress_level is not None:
                    if evict_score is not None:
                        if evict_score > max_compress_score:
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
            return EvictOpType.COMPRESS, \
            (self._heaps[evictor_backend_no][current_evict_item_compress_level].pop_heap(), selected_to_compress_level)
        else:
            assert current_evict_item_compress_level is not None
            return EvictOpType.WRITE_TO_LOWER, self._heaps[evictor_backend_no][current_evict_item_compress_level].pop_heap()

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

    def get_store_info(self, chunk_kv_query: list, timepoint: float, space_list = None) -> list:
        assert space_list is not None
        if self._store_policy == StorePolicy.FAST_DEVICE:
            # NOTE: Now always put to fast device, but not always full version.
            # Store to fast device for now.
            cl = self._store_compress_level[0]
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
            return [(0, cl) for _ in range(len(chunk_kv_query))]
        else:
            # TODO: Use a similar method with optimize_on_hit.
            raise NotImplementedError(f"Not implemented store policy {self._store_policy_str}")
        