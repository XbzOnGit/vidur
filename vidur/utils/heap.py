class Heap:
    def __init__(self, key_attr: str, position_attr: str):
        self._key_attr = key_attr
        self._position_attr = position_attr
        self._heap = []
    def _swap_in_heap(self, item1, item2):
        index1 = getattr(item1, self._position_attr)
        index2 = getattr(item2, self._position_attr)
        setattr(item1, self._position_attr, index2)
        setattr(item2, self._position_attr, index1)
        self._heap[index1] = item2
        self._heap[index2] = item1
    def _sift_up_heap(self, item, unconditional: bool = False):
        index = getattr(item, self._position_attr)
        while index > 0:
            parent_index = (index - 1) // 2
            parent = self._heap[parent_index]
            if not unconditional and getattr(parent, self._key_attr) <= getattr(item, self._key_attr):
                break
            self._swap_in_heap(item, parent)
            index = parent_index
    def _sift_down_heap(self, item, unconditional: bool = False):
        index = getattr(item, self._position_attr)
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            left_child = None
            right_child = None
            if left_child_index < len(self._heap):
                left_child = self._heap[left_child_index]
            if right_child_index < len(self._heap):
                right_child = self._heap[right_child_index]
            if left_child is None and right_child is None:
                # None x None.
                break
            if left_child is not None and right_child is not None:
                # y x y
                if getattr(left_child, self._key_attr) <= getattr(right_child, self._key_attr):
                    min_child = left_child
                    min_child_index = left_child_index
                else:
                    min_child = right_child
                    min_child_index = right_child_index
            else:
                # Cannot be None x y, so must be y x None.
                # Because right child cannot be there if left child is not there.
                assert left_child is not None
                assert right_child is None
                min_child = left_child
                min_child_index = left_child_index

            if not unconditional and getattr(item, self._key_attr) <= getattr(min_child, self._key_attr):
                break
            self._swap_in_heap(item, min_child)
            index = min_child_index
    def insert(self, item):
        setattr(item, self._position_attr, len(self._heap))
        self._heap.append(item)
        self._sift_up_heap(item)
        # print(f"Position after insert {getattr(item, self._position_attr)}")

    def _pop_root_from_heap(self):
        root = self._heap[0]
        last = self._heap.pop()
        if len(self._heap) > 0:
            self._heap[0] = last
            setattr(last, self._position_attr, 0)
            self._sift_down_heap(last)
        return root

    def delete(self, item):
        index = getattr(item, self._position_attr)
        assert index >= 0
        self._sift_up_heap(item, unconditional=True)
        self._pop_root_from_heap()

    def modify_heap(self, item):
        index = getattr(item, self._position_attr)
        assert index >= 0
        parent_index = (index - 1) // 2
        parent = self._heap[parent_index]
        if getattr(parent, self._key_attr) > getattr(item, self._key_attr):
            # If originally in shape, then originally children >= self >= parent.
            # Then self gets smaller, cannot sift down for sure, now sift up.
            self._sift_up_heap(item)
        else:
            # Cannot sift up for sure, now try to sift down.
            self._sift_down_heap(item)
    def get_min(self):
        return self._heap[0] if len(self._heap) > 0 else None
    
    def size(self):
        return len(self._heap)
        
