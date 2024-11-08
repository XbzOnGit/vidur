from abc import ABC, abstractmethod
from typing import List
from vidur.utils.itemlist import ItemList, ItemListWrapper
from vidur.utils.itemheap import ItemHeap, ItemHeapWrapper
from vidur.types import EvictOpType



class BaseEvictor(ABC):
    def __init__(self):
        pass
    # Always return in the format of operation, other operands.
    @abstractmethod
    def update_on_get(self, chunk_kv: list, timepoint: float):
        # Use these chunks from storage.
        pass
    @abstractmethod
    def update_on_put(self, chunk_kv: list, timepoint: float):
        # These chunks are put into storage.
        pass
    @abstractmethod
    def evict(self):
        pass
    @abstractmethod
    def update_on_transform(self, from_kv_obj, to_kv_obj, timepoint: float):
        pass
    @abstractmethod
    def update_on_transfer(self, from_kv_obj, to_kv_obj):
        pass


class LRUEvictor(BaseEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def update_on_get(self, chunk_kv: list, timepoint: float):
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
        
    def evict(self):
        return EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap()


class LFUEvictor(BaseEvictor):
    def __init__(self):
        super().__init__()
        self._item_heap = ItemHeap()
    def update_on_get(self, chunk_kv: list, timepoint: float):
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
        
    def evict(self):
        return EvictOpType.WRITE_TO_LOWER, self._item_heap.pop_heap()
    

class OurEvictorV1(BaseEvictor):
    def __init__(self, threshold: int):
        super().__init__()
        self._threshold = threshold
        self._heaps: List[ItemHeap] = [ItemHeap(), ItemHeap()]
    # NOTE: Call update before decode && after decode for those frequency to be correct.
    # NOTE: I think do need to inherent frequency? Check this.
    def update_on_get(self, chunk_kv: list, timepoint: float):
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

    def evict(self):
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
        
