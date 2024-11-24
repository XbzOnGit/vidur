from typing import List
# Min heap.
item_idx = 0
class ItemHeapWrapper:
    def __init__(self, item, score):
        self._item = item
        self._score = score
        self._index = None
        global item_idx
        self._id = item_idx
        item_idx += 1
    @property
    def item(self):
        return self._item
    @property
    def score(self):
        return self._score
    @property
    def index(self):
        return self._index
    def set_index(self, index):
        self._index = index
    def set_score(self, score):
        self._score = score
    def __lt__(self, other):
        return self._score < other._score
    def __le__(self, other):
        return self._score <= other._score
    def __eq__(self, other):
        return self._score == other._score
    def __ne__(self, other):
        return self._score != other._score
    def __gt__(self, other):
        return self._score > other._score
    def __ge__(self, other):
        return self._score >= other._score

class ItemHeap:
    def __init__(self):
        self._heap: List[ItemHeapWrapper] = []
        self._check_set = set()
    def push_heap(self, item: ItemHeapWrapper):
        # print("Pushing item ", item.item)
        assert item.item not in self._check_set
        self._check_set.add(item.item)
        self._heap.append(item)
        item.set_index(len(self._heap) - 1)
        self._sift_up(len(self._heap) - 1)
        assert item.index is not None
    def pop_heap(self, original_form=False):
        if len(self._heap) == 0:
            return None
        self._check_set.remove(self._heap[0].item)
        self.swap(0, len(self._heap) - 1)
        item = self._heap.pop()
        item.set_index(None)
        self._sift_down(0)
        if original_form:
            return item
        return item.item
    def top(self):
        if len(self._heap) == 0:
            return None
        return self._heap[0].item
    def remove(self, item: ItemHeapWrapper):
        index = item.index
        self.swap(index, len(self._heap) - 1)
        self._heap.pop()
        item.set_index(None)
        self._sift_down(index)
    def update_on_keychange(self, item: ItemHeapWrapper):
        index = item.index
        if index is None:
            self.push_heap(item)
            return
        parent = (index - 1) // 2
        if index > 0 and self._heap[parent].score > self._heap[index].score:
            self._sift_up(index)
        else:
            self._sift_down(index)
    def swap(self, index1, index2):
        item1 = self._heap[index1]
        item2 = self._heap[index2]
        self._heap[index1] = item2
        self._heap[index2] = item1
        item1.set_index(index2)
        item2.set_index(index1)
    def size(self):
        return len(self._heap)
    def _sift_up(self, index):
        while index > 0:
            parent = (index - 1) // 2
            if self._heap[parent].score <= self._heap[index].score:
                break
            self.swap(parent, index)
            index = parent
    def _sift_down(self, index):
        while index * 2 + 1 < len(self._heap):
            left = index * 2 + 1
            right = index * 2 + 2
            smallest = left
            if right < len(self._heap) and self._heap[right].score < self._heap[left].score:
                smallest = right
            if self._heap[index].score <= self._heap[smallest].score:
                break
            self.swap(index, smallest)
            index = smallest
