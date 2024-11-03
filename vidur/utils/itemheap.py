from typing import List
# Min heap.
class ItemHeapWrapper:
    def __init__(self, item, score):
        self._item = item
        self._score = score
        self._index = None
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

class ItemHeap:
    def __init__(self):
        self._heap: List[ItemHeapWrapper] = []
    def push_heap(self, item, score):
        wrapped_item = ItemHeapWrapper(item, score)
        self._heap.append(wrapped_item)
        wrapped_item.set_index(len(self._heap) - 1)
        self._sift_up(len(self._heap) - 1)
    def pop_heap(self):
        if len(self._heap) == 0:
            return None
        self.swap(0, len(self._heap) - 1)
        item = self._heap.pop()
        item.index = None
        self._sift_down(0)
        return item.item
    def top(self):
        if len(self._heap) == 0:
            return None
        return self._heap[0].item
    def remove(self, item):
        index = item.index
        self.swap(index, len(self._heap) - 1)
        self._heap.pop()
        item.index = None
        self._sift_down(index)
    def update_on_keychange(self, item):
        index = item.index
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
